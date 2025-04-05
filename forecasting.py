import torch
from torch import nn
import torch.nn.functional as F
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os

class MineDataForecaster:
    def __init__(self, model_path='huggingface/time-series-transformer-tourism-monthly', 
                 sequence_length=24, prediction_length=1, checkpoint_dir='./checkpoints'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=sequence_length,
            input_size=6,  # co, co2, temp, humidity, pm25, pm10
            num_time_features=4,  # hour, day_of_week, month, year
            d_model=64,
            n_heads=8
        )
        
        self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        self.feature_scaler = StandardScaler()
        
    def prepare_sequence(self, data, multi_step=False):
        """Prepare sequences with enhanced time features"""
        features = ['co_level', 'co2_level', 'temperature', 
                   'humidity', 'pm25', 'pm10']
        
        scaled_data = self.feature_scaler.fit_transform(data[features])
        data[features] = scaled_data
        
        sequences = []
        targets = []
        timestamps = []
        
        step_size = self.prediction_length if multi_step else 1
        for i in range(0, len(data) - self.sequence_length - self.prediction_length + 1, step_size):
            seq = data[features].iloc[i:i+self.sequence_length].values
            target = data[features].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length].values
            time_feature = data['timestamp'].iloc[i:i+self.sequence_length].values
            sequences.append(seq)
            targets.append(target)
            timestamps.append(self._encode_timestamp(time_feature))
            
        return (np.array(sequences), np.array(timestamps)), np.array(targets)
    
    def _encode_timestamp(self, timestamps):
        """Enhanced timestamp encoding with multiple features"""
        time_features = np.zeros((len(timestamps), 4))
        for i, ts in enumerate(timestamps):
            dt = pd.Timestamp(ts)
            time_features[i, 0] = dt.hour / 24.0  # Hour of day
            time_features[i, 1] = dt.dayofweek / 6.0  # Day of week (0-6)
            time_features[i, 2] = (dt.month - 1) / 11.0  # Month (0-11)
            time_features[i, 3] = (dt.year - 2000) / 100.0  # Year (normalized)
        return time_features
    
    def train(self, historical_data, epochs=10, batch_size=32, validation_split=0.2):
        """Training with checkpointing"""
        (X, timestamps), y = self.prepare_sequence(historical_data, multi_step=True)
        
        split_idx = int(len(X) * (1 - validation_split))
        train_dataset = TensorDataset(
            torch.FloatTensor(X[:split_idx]),
            torch.FloatTensor(timestamps[:split_idx]),
            torch.FloatTensor(y[:split_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X[split_idx:]),
            torch.FloatTensor(timestamps[split_idx:]),
            torch.FloatTensor(y[split_idx:])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_time, batch_y in train_loader:
                batch_X, batch_time, batch_y = (batch_X.to(self.device), 
                                              batch_time.to(self.device), 
                                              batch_y.to(self.device))
                
                outputs = self.model(
                    past_values=batch_X,
                    past_time_features=batch_time
                ).prediction_outputs
                
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            val_loss = self._validate(val_loader, criterion)
            scheduler.step(val_loss)
            
            train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_time, batch_y in val_loader:
                batch_X, batch_time, batch_y = (batch_X.to(self.device), 
                                              batch_time.to(self.device), 
                                              batch_y.to(self.device))
                outputs = self.model(
                    past_values=batch_X,
                    past_time_features=batch_time
                ).prediction_outputs
                val_loss += criterion(outputs, batch_y).item()
        return val_loss / len(val_loader)
    
    def predict(self, sequence, timestamp, steps=1):
        """Multi-step prediction"""
        self.model.eval()
        predictions = []
        current_seq = sequence.copy()
        current_time = timestamp.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                seq_tensor = torch.FloatTensor(current_seq[-self.sequence_length:]).unsqueeze(0).to(self.device)
                time_tensor = torch.FloatTensor(self._encode_timestamp(current_time[-self.sequence_length:])).unsqueeze(0).to(self.device)
                
                outputs = self.model(
                    past_values=seq_tensor,
                    past_time_features=time_tensor
                )
                pred = outputs.prediction_outputs.cpu().numpy()[0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                if steps > 1:
                    current_seq = np.concatenate([current_seq, pred])
                    # Generate next timestamp (assuming hourly data)
                    last_time = pd.Timestamp(current_time[-1])
                    next_time = last_time + pd.Timedelta(hours=1)
                    current_time = np.append(current_time, next_time)
        
        predictions = np.array(predictions)
        if steps == 1:
            return predictions[0], outputs.prediction_outputs.std(dim=-1).cpu().numpy()[0]
        return predictions, outputs.prediction_outputs.std(dim=-1).cpu().numpy()[0]
    
    def evaluate(self, test_data):
        """Evaluate with multi-step support"""
        (X_test, timestamps), y_test = self.prepare_sequence(test_data, multi_step=True)
        predictions = []
        
        for seq, time in zip(X_test, timestamps):
            pred, _ = self.predict(seq, time)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        y_test = y_test.squeeze(1) if self.prediction_length == 1 else y_test
        
        features = ['co_level', 'co2_level', 'temperature', 
                   'humidity', 'pm25', 'pm10']
        metrics = {}
        
        for i, feature in enumerate(features):
            metrics[feature] = {
                'MAE': mean_absolute_error(y_test[:, :, i].flatten(), predictions[:, :, i].flatten()),
                'RMSE': np.sqrt(mean_squared_error(y_test[:, :, i].flatten(), predictions[:, :, i].flatten())),
                'R2': r2_score(y_test[:, :, i].flatten(), predictions[:, :, i].flatten())
            }
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.scale_ = np.std(data, axis=0)
        return (data - self.mean_) / self.scale_
    
    def transform(self, data):
        return (data - self.mean_) / self.scale_
    
    def inverse_transform(self, data):
        return (data * self.scale_) + self.mean_
