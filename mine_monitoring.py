import threading
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import serial
from digi.xbee.devices import XBeeDevice
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import RPi.GPIO as GPIO
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.text import MIMEText
import json
import pjsua2 as pj
import torch
from torch import nn
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import os
import logging

# Configuration
CONFIG = {
    'DB_PATH': os.path.join(os.path.dirname(__file__), 'mine_data.db'),
    'USE_TRANSFORMER': False,  # Toggle for Transformer model (default to LinearRegression)
    'SERIAL_PORT': '/dev/ttyUSB0',
    'BAUD_RATE': 9600,
    'GPIO_LED_PIN': 18,
    'GPIO_BUZZER_PIN': 23,
    'GPIO_VENTILATION_PIN': 24
}

# Configure logging
logging.basicConfig(
    filename='mine_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Thread-safe database lock
db_lock = threading.Lock()

# --- Data Collection Section ---
class MineDataCollector:
    def __init__(self, db_path=CONFIG['DB_PATH'], use_sensors=False, port=CONFIG['SERIAL_PORT'], baud_rate=CONFIG['BAUD_RATE']):
        self.db_path = db_path
        self.use_sensors = use_sensors
        self.port = port
        self.baud_rate = baud_rate
        self.calibration_factors = {
            'co_level': 0.1, 'co2_level': 0.2, 'temperature': 1.0,
            'humidity': 1.0, 'pm25': 1.0, 'pm10': 1.0
        }
        self.setup_database()
        if use_sensors:
            try:
                self.xbee = XBeeDevice(self.port, self.baud_rate)
                self.xbee.open()
                logging.info("XBee device initialized successfully")
            except Exception as e:
                logging.error(f"XBee initialization failed: {e}")
                self.use_sensors = False

    def setup_database(self):
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    timestamp DATETIME, location_id TEXT,
                    co_level FLOAT, co2_level FLOAT, temperature FLOAT,
                    humidity FLOAT, pm25 FLOAT, pm10 FLOAT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_log (
                    timestamp DATETIME, sensor TEXT,
                    raw_value FLOAT, calibrated_value FLOAT, calibration_factor FLOAT
                )
            ''')
            conn.commit()
            conn.close()

    def log_calibration(self, sensor, raw_value, calibrated_value, factor):
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            data = {
                'timestamp': datetime.now(), 'sensor': sensor,
                'raw_value': raw_value, 'calibrated_value': calibrated_value,
                'calibration_factor': factor
            }
            df = pd.DataFrame([data])
            df.to_sql('calibration_log', conn, if_exists='append', index=False)
            conn.close()

    def get_sensor_reading(self):
        if not self.use_sensors:
            return self.simulate_sensor_reading()
        try:
            packet = self.xbee.read_data(timeout=5)
            if packet:
                data = packet.data.decode().strip().split(',')
                raw_values = [float(x) for x in data]
                calibrated = {}
                for i, (key, factor) in enumerate(self.calibration_factors.items()):
                    raw = raw_values[i]
                    calib_value = raw * factor
                    calibrated[key] = calib_value
                    self.log_calibration(key, raw, calib_value, factor)
                return calibrated
            else:
                raise Exception("No data received from XBee")
        except Exception as e:
            logging.error(f"Failed to read sensor data from XBee: {e}")
            return self.simulate_sensor_reading()

    def simulate_sensor_reading(self):
        sim_reading = {
            'co_level': random.uniform(0, 50), 'co2_level': random.uniform(300, 1000),
            'temperature': random.uniform(15, 35), 'humidity': random.uniform(30, 90),
            'pm25': random.uniform(0, 35), 'pm10': random.uniform(0, 150)
        }
        for key in sim_reading:
            self.log_calibration(key, sim_reading[key] / self.calibration_factors[key],
                               sim_reading[key], self.calibration_factors[key])
        return sim_reading

    def collect_data(self, location_id, duration_seconds=60, interval_seconds=5):
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            start_time = time.time()
            end_time = start_time + duration_seconds
            while time.time() < end_time:
                readings = self.get_sensor_reading()
                timestamp = datetime.now()
                data = {'timestamp': timestamp, 'location_id': location_id, **readings}
                df = pd.DataFrame([data])
                df.to_sql('sensor_data', conn, if_exists='append', index=False)
                print(f"Recorded data at {timestamp}: {readings}")
                time.sleep(interval_seconds)
            conn.close()

    def get_historical_data(self, start_date=None, end_date=None):
        with db_lock:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM sensor_data"
            if start_date and end_date:
                query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
            df = pd.read_sql_query(query, conn)
            conn.close()
        return df

# --- Alarm System Section ---
class MineAlarmSystem:
    def __init__(self, emergency_system=None):
        self.thresholds = {
            'co_level': 50.0, 'co2_level': 1000.0, 'temperature': 30.0,
            'humidity': 85.0, 'pm25': 35.0, 'pm10': 150.0
        }
        GPIO.setmode(GPIO.BCM)
        self.led_pin = CONFIG['GPIO_LED_PIN']
        self.buzzer_pin = CONFIG['GPIO_BUZZER_PIN']
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.model_trained = False
        self.emergency_system = emergency_system

    def check_thresholds(self, readings):
        alerts = []
        for key, value in readings.items():
            if key in self.thresholds and value > self.thresholds[key]:
                alerts.append(f"{key} exceeds threshold: {value:.2f}")
        if alerts and self.emergency_system:
            self.emergency_system.send_alert("\n".join(alerts))
        return alerts

    def trigger_alarm(self, duration=2):
        GPIO.output(self.led_pin, GPIO.HIGH)
        GPIO.output(self.buzzer_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.led_pin, GPIO.LOW)
        GPIO.output(self.buzzer_pin, GPIO.LOW)

    def train_anomaly_detector(self, historical_data):
        features = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
        X = historical_data[features].values
        self.isolation_forest.fit(X)
        self.model_trained = True

    def check_anomalies(self, readings):
        if not self.model_trained:
            return False
        features = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
        X = np.array([[readings[f] for f in features]])
        prediction = self.isolation_forest.predict(X)
        if prediction[0] == -1 and self.emergency_system:
            self.emergency_system.send_alert(f"Anomaly detected: {readings}")
        return prediction[0] == -1

    def cleanup(self):
        GPIO.cleanup()

# --- Emergency Response Section ---
class EmergencyResponse:
    def __init__(self):
        self.ventilation_pin = CONFIG['GPIO_VENTILATION_PIN']
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ventilation_pin, GPIO.OUT)
        self.config = {
            'emergency_contacts': ['supervisor@mine.com'],
            'phone_numbers': ['sip:supervisor@sip.mine.com'],
            'evacuation_zones': ['shaft_1', 'shaft_2', 'ventilation_area'],
            'ventilation_threshold': {'co_level': 30.0, 'co2_level': 800.0}
        }
        self.emergency_active = False
        self.evacuation_in_progress = False
        self._setup_sip()

    def _setup_sip(self):
        self.lib = pj.Lib()
        self.lib.init()
        self.transport = self.lib.create_transport(pj.TransportType.UDP)
        self.lib.start()
        self.acc = self.lib.create_account(pj.AccountConfig("sip.mine.com", "alert-system", "password"))

    def activate_ventilation(self):
        GPIO.output(self.ventilation_pin, GPIO.HIGH)
        print("[EMERGENCY] Ventilation system activated")

    def deactivate_ventilation(self):
        GPIO.output(self.ventilation_pin, GPIO.LOW)
        print("[EMERGENCY] Ventilation system deactivated")

    def send_alert(self, message, level='warning'):
        for contact in self.config['emergency_contacts']:
            msg = MIMEText(message)
            msg['Subject'] = f'MINE ALERT [{level.upper()}]: Emergency Situation'
            msg['From'] = 'mine-monitoring@mine.com'
            msg['To'] = contact
            try:
                with smtplib.SMTP('smtp.mine.com', 587) as server:
                    server.starttls()
                    server.login('alert-system', 'password')
                    server.send_message(msg)
                print(f"Email alert sent to {contact}")
            except Exception as e:
                print(f"Email failed: {e}")
        for number in self.config['phone_numbers']:
            try:
                call = self.acc.make_call(number, pj.CallOpParam())
                print(f"VoIP call initiated to {number}")
                time.sleep(5)
                call.hangup()
            except Exception as e:
                print(f"VoIP call failed: {e}")

    def initiate_evacuation(self, danger_zones):
        if not self.evacuation_in_progress:
            self.evacuation_in_progress = True
            message = f"EMERGENCY EVACUATION REQUIRED in zones: {', '.join(danger_zones)}"
            self.send_alert(message, level='critical')
            print("[EMERGENCY] Evacuation procedures initiated")
            threading.Thread(target=self.monitor_evacuation).start()

    def monitor_evacuation(self):
        pass

    def assess_situation(self, sensor_data, predictions, anomalies):
        responses_triggered = []
        if (sensor_data['co_level'] > self.config['ventilation_threshold']['co_level'] or
            sensor_data['co2_level'] > self.config['ventilation_threshold']['co2_level']):
            self.activate_ventilation()
            responses_triggered.append('ventilation')
        if anomalies:
            self.send_alert(f"Anomaly detected: {json.dumps(sensor_data)}")
            responses_triggered.append('alert')
        if predictions is not None:
            dangerous_predictions = any(
                predictions[0][i] > self.config['ventilation_threshold'][key]
                for i, key in enumerate(['co_level', 'co2_level'])
            )
            if dangerous_predictions:
                self.initiate_evacuation(['shaft_1'])
                responses_triggered.append('evacuation')
        return responses_triggered

    def cleanup(self):
        self.deactivate_ventilation()
        GPIO.cleanup(self.ventilation_pin)
        self.lib.destroy()

# --- Forecasting Section ---
class MineDataForecaster:
    def __init__(self, use_transformer=CONFIG['USE_TRANSFORMER'], sequence_length=12, prediction_length=1, checkpoint_dir='./checkpoints'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_transformer = use_transformer
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        if use_transformer:
            config = TimeSeriesTransformerConfig(
                prediction_length=prediction_length, context_length=sequence_length,
                input_size=6, num_time_features=4, d_model=32, n_heads=4
            )
            self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        else:
            self.lr_model = LinearRegression()
        self.feature_scaler = StandardScaler()

    def prepare_sequence(self, data, multi_step=False):
        features = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
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
        if self.use_transformer:
            return (np.array(sequences), np.array(timestamps)), np.array(targets)
        else:
            return np.array([seq.flatten() for seq in sequences]), np.array([t.flatten() for t in targets])

    def _encode_timestamp(self, timestamps):
        time_features = np.zeros((len(timestamps), 4))
        for i, ts in enumerate(timestamps):
            dt = pd.Timestamp(ts)
            time_features[i, 0] = dt.hour / 24.0
            time_features[i, 1] = dt.dayofweek / 6.0
            time_features[i, 2] = (dt.month - 1) / 11.0
            time_features[i, 3] = (dt.year - 2000) / 100.0
        return time_features

    def train(self, historical_data, epochs=5 if self.use_transformer else 1, batch_size=16, validation_split=0.2):
        X, y = self.prepare_sequence(historical_data, multi_step=True)
        if self.use_transformer:
            split_idx = int(len(X[0]) * (1 - validation_split))
            train_dataset = TensorDataset(
                torch.FloatTensor(X[0][:split_idx]), torch.FloatTensor(X[1][:split_idx]),
                torch.FloatTensor(y[:split_idx])
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X[0][split_idx:]), torch.FloatTensor(X[1][split_idx:]),
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
                    batch_X, batch_time, batch_y = (batch_X.to(self.device), batch_time.to(self.device), batch_y.to(self.device))
                    outputs = self.model(past_values=batch_X, past_time_features=batch_time).prediction_outputs
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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
        else:
            self.lr_model.fit(X, y)
            print("Linear Regression model trained.")

    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_time, batch_y in val_loader:
                batch_X, batch_time, batch_y = (batch_X.to(self.device), batch_time.to(self.device), batch_y.to(self.device))
                outputs = self.model(past_values=batch_X, past_time_features=batch_time).prediction_outputs
                val_loss += criterion(outputs, batch_y).item()
        return val_loss / len(val_loader)

    def predict(self, sequence, timestamp=None, steps=1):
        if self.use_transformer:
            self.model.eval()
            predictions = []
            current_seq = sequence.copy()
            current_time = timestamp.copy()
            with torch.no_grad():
                for _ in range(steps):
                    seq_tensor = torch.FloatTensor(current_seq[-self.sequence_length:]).unsqueeze(0).to(self.device)
                    time_tensor = torch.FloatTensor(self._encode_timestamp(current_time[-self.sequence_length:])).unsqueeze(0).to(self.device)
                    outputs = self.model(past_values=seq_tensor, past_time_features=time_tensor)
                    pred = outputs.prediction_outputs.cpu().numpy()[0]
                    predictions.append(pred)
                    if steps > 1:
                        current_seq = np.concatenate([current_seq, pred])
                        last_time = pd.Timestamp(current_time[-1])
                        next_time = last_time + pd.Timedelta(hours=1)
                        current_time = np.append(current_time, next_time)
            predictions = np.array(predictions)
            return predictions[0] if steps == 1 else predictions
        else:
            sequence_flat = sequence.flatten().reshape(1, -1)
            pred = self.lr_model.predict(sequence_flat)[0]
            return pred.reshape(self.prediction_length, 6)

    def evaluate(self, test_data):
        X, y = self.prepare_sequence(test_data, multi_step=True)
        predictions = []
        if self.use_transformer:
            for seq, time in zip(X[0], X[1]):
                pred = self.predict(seq, time)
                predictions.append(pred)
        else:
            for seq in X:
                pred = self.predict(seq)
                predictions.append(pred)
        predictions = np.array(predictions)
        y_test = y.squeeze(1) if self.prediction_length == 1 else y
        features = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
        metrics = {}
        for i, feature in enumerate(features):
            metrics[feature] = {
                'MAE': mean_absolute_error(y[:, :, i].flatten(), predictions[:, :, i].flatten()),
                'RMSE': np.sqrt(mean_squared_error(y[:, :, i].flatten(), predictions[:, :, i].flatten())),
                'R2': r2_score(y[:, :, i].flatten(), predictions[:, :, i].flatten())
            }
        return metrics

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'val_loss': val_loss}
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
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

# --- Web Dashboard Section ---
app = Flask(__name__)
app.secret_key = 'your-secret-key'
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

def get_sensor_data():
    with db_lock:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        df = pd.read_sql_query("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
    return df

def get_recent_alerts():
    with db_lock:
        conn = sqlite3.connect(CONFIG['DB_PATH'])
        df = pd.read_sql_query("""
            SELECT timestamp, location_id, co_level, co2_level, temperature, humidity, pm25, pm10
            FROM sensor_data 
            WHERE co_level > 50 OR co2_level > 1000 OR temperature > 30 OR humidity > 85 OR pm25 > 35 OR pm10 > 150
            ORDER BY timestamp DESC LIMIT 10
        """, conn)
        conn.close()
    return df.to_dict(orient='records')

@app.route('/')
@login_required
def index():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            login_user(User(1))
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/sensor-data')
@login_required
def sensor_data():
    df = get_sensor_data()
    data = {
        'co_level': {'values': df['co_level'].tolist(), 'timestamps': df['timestamp'].tolist()},
        'co2_level': {'values': df['co2_level'].tolist(), 'timestamps': df['timestamp'].tolist()},
        'temperature': {'values': df['temperature'].tolist(), 'timestamps': df['timestamp'].tolist()},
        'humidity': {'values': df['humidity'].tolist(), 'timestamps': df['timestamp'].tolist()},
        'pm25': {'values': df['pm25'].tolist(), 'timestamps': df['timestamp'].tolist()},
        'pm10': {'values': df['pm10'].tolist(), 'timestamps': df['timestamp'].tolist()}
    }
    return jsonify(data)

@app.route('/api/alerts')
@login_required
def alerts():
    alerts = get_recent_alerts()
    return jsonify(alerts)

# --- System Management Section ---
class MineMonitoringSystem:
    def __init__(self):
        self.collector = MineDataCollector(use_sensors=True)
        self.emergency_system = EmergencyResponse()
        self.alarm_system = MineAlarmSystem(self.emergency_system)
        self.forecaster = MineDataForecaster()
        self.running = False

    def simulate_optical_fiber(self, data):
        logging.info(f"Simulated optical fiber transmission: {data}")
        return data

    def start_data_collection(self):
        while self.running:
            for location in ['mine_shaft_1', 'mine_shaft_2', 'ventilation_area']:
                readings = self.collector.get_sensor_reading()
                logging.info(f"Collected readings from {location}: {readings}")
                self.collector.collect_data(location, duration_seconds=10, interval_seconds=2)
                readings = self.simulate_optical_fiber(readings)
                alerts = self.alarm_system.check_thresholds(readings)
                if alerts:
                    logging.warning(f"Threshold alerts at {location}: {alerts}")
                    self.alarm_system.trigger_alarm()
                is_anomaly = self.alarm_system.check_anomalies(readings)
                if is_anomaly:
                    logging.warning(f"Anomaly detected at {location}: {readings}")
                    self.alarm_system.trigger_alarm(duration=1)
                historical_data = self.collector.get_historical_data()
                predictions = None
                if len(historical_data) > 24:
                    sequence = historical_data[-24:][['co_level', 'co2_level', 'temperature',
                                                    'humidity', 'pm25', 'pm10']].values
                    predictions = self.forecaster.predict(sequence)
                    logging.info(f"Predictions for {location}: {predictions}")
                responses = self.emergency_system.assess_situation(readings, predictions, is_anomaly)
                if responses:
                    logging.critical(f"Emergency responses triggered at {location}: {responses}")
                time.sleep(5)

    def train_models(self):
        logging.info("Training models with historical data...")
        historical_data = self.collector.get_historical_data()
        if len(historical_data) > 24:
            self.alarm_system.train_anomaly_detector(historical_data)
            logging.info("Anomaly detector trained")
            self.forecaster.train(historical_data)
            logging.info("Forecasting model trained")
            metrics = self.forecaster.evaluate(historical_data)
            logging.info(f"Forecasting Model Metrics: {metrics}")

    def start(self):
        self.running = True
        logging.info("Starting mine monitoring system...")
        self.train_models()
        collection_thread = threading.Thread(target=self.start_data_collection)
        collection_thread.daemon = True
        collection_thread.start()
        app.run(host='0.0.0.0', port=8080)

    def stop(self):
        self.running = False
        logging.info("Shutting down mine monitoring system...")
        self.alarm_system.cleanup()
        self.emergency_system.cleanup()

if __name__ == "__main__":
    system = MineMonitoringSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()
