
import torch
from torch import nn
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

class MineDataForecaster:
    def __init__(self, model_path='meta-llama/Llama-2-7b'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=6  # co, co2, temp, humidity, pm25, pm10
        ).to(self.device)
        
    def prepare_sequence(self, data, sequence_length=24):
        """Prepare sequences for time series prediction"""
        features = ['co_level', 'co2_level', 'temperature', 
                   'humidity', 'pm25', 'pm10']
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[features].iloc[i:i+sequence_length].values
            target = data[features].iloc[i+sequence_length].values
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def train(self, historical_data, epochs=10):
        """Train the forecasting model"""
        X, y = self.prepare_sequence(historical_data)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(X), 32):
                batch_X = torch.FloatTensor(X[i:i+32]).to(self.device)
                batch_y = torch.FloatTensor(y[i:i+32]).to(self.device)
                
                outputs = self.model(batch_X).logits
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X):.4f}")
    
    def predict(self, sequence):
        """Make predictions for the next time step"""
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            prediction = self.model(sequence_tensor).logits
            return prediction.cpu().numpy()[0]
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        X_test, y_test = self.prepare_sequence(test_data)
        predictions = []
        
        for sequence in X_test:
            pred = self.predict(sequence)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'R2': r2_score(y_test, predictions)
        }
        
        return metrics
