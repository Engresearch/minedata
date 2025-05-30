#!/usr/bin/env python3

import threading
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import serial
from digi.xbee.devices import XBeeDevice
from flask import Flask, render_template
from flask_socketio import SocketIO
import RPi.GPIO as GPIO
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
import json
import pjsua2 as pj
import torch
from torch import nn
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import socket
import re
import schedule
from filelock import FileLock

# Configuration
CONFIG = {
    'DB_PATH': 'mine_data.db',
    'SERIAL_PORT': '/dev/ttyUSB0',
    'BAUD_RATE': 9600,
    'TCP_HOST': '0.0.0.0',
    'TCP_PORT': 5000,
    'FEEDBACK_IP': '192.168.1.20',
    'FEEDBACK_PORT': 5001,
    'GPIO_LED_PIN': 18,
    'GPIO_BUZZER_PIN': 23,
    'GPIO_VENTILATION_PIN': 24,
    'LOG_FILE': 'mine_monitoring.log',
    'LOG_INTERVAL': 30,
    'CSV_PATH': 'sensor_data.csv'
}

# Thresholds
THRESHOLDS = {
    'co_level': 50.0, 'co2_level': 1000.0, 'temperature': 30.0,
    'humidity': 85.0, 'pm25': 35.0, 'pm10': 150.0
}

VENTILATION_THRESHOLDS = {'co_level': 30.0, 'co2_level': 800.0}

# Sensor validation ranges
VALIDATION_RANGES = {
    'co_level': (0, 1000), 'co2_level': (0, 5000), 'temperature': (-20, 60),
    'humidity': (0, 100), 'pm25': (0, 500), 'pm10': (0, 1000)
}

# Regex patterns for TCP data parsing
PATTERNS = {
    'SensorID': r'SensorID:\s*(\w+)',
    'co_level': r'MQ7\s*AirQua=(\d+)\s*PPM',
    'co2_level': r'MQ135\s*AirQua=(\d+)\s*PPM',
    'humidity': r'Humidity:\s*([\d.]+)\s*%',
    'temperature': r'Temperature:\s*([\d.]+)\s*C',
    'pm25': r'P2\.5:\s*([\d.]+)',
    'pm10': r'P10:\s*([\d.]+)'
}

# Configure logging
logging.basicConfig(
    filename=CONFIG['LOG_FILE'],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Thread-safe database and file locks
db_lock = threading.Lock()
csv_lock = FileLock(CONFIG['CSV_PATH'] + ".lock")

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(CONFIG['GPIO_LED_PIN'], GPIO.OUT)
GPIO.setup(CONFIG['GPIO_BUZZER_PIN'], GPIO.OUT)
GPIO.setup(CONFIG['GPIO_VENTILATION_PIN'], GPIO.OUT)
GPIO.output(CONFIG['GPIO_VENTILATION_PIN'], GPIO.LOW)

# Flask and SocketIO app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# --- Data Collection Section ---
class MineDataCollector:
    def __init__(self, db_path=CONFIG['DB_PATH'], use_sensors=True, port=CONFIG['SERIAL_PORT'], baud_rate=CONFIG['BAUD_RATE']):
        self.db_path = db_path
        self.use_sensors = use_sensors
        self.port = port
        self.baud_rate = baud_rate
        self.calibration_factors = {
            'co_level': 0.1, 'co2_level': 0.2, 'temperature': 1.0,
            'humidity': 1.0, 'pm25': 1.0, 'pm10': 1.0
        }
        self.xbee = None
        self.setup_database()
        if use_sensors:
            try:
                self.xbee = XBeeDevice(self.port, self.baud_rate)
                self.xbee.open()
                logger.info("XBee device initialized successfully")
            except Exception as e:
                logger.error(f"XBee initialization failed: {e}")
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

    def validate_readings(self, readings):
        """Validate sensor readings against physical ranges."""
        valid = True
        errors = []
        for key, value in readings.items():
            if key in VALIDATION_RANGES:
                min_val, max_val = VALIDATION_RANGES[key]
                if not (min_val <= value <= max_val):
                    errors.append(f"{key} out of range: {value} (expected {min_val}-{max_val})")
                    valid = False
        if not valid:
            logger.error(f"Invalid readings: {errors}")
        return valid, errors

    def get_sensor_reading(self):
        if not self.use_sensors:
            return self.simulate_sensor_reading()
        try:
            packet = self.xbee.read_data(timeout=5)
            if packet:
                data = packet.data.decode().strip().split(',')
                raw_values = [float(x) for x in data]
                # Simulate redundant sensors for CO and CO2
                co_readings = [raw_values[0] * (1 + random.uniform(-0.1, 0.1)) for _ in range(2)]
                co2_readings = [raw_values[1] * (1 + random.uniform(-0.1, 0.1)) for _ in range(2)]
                calibrated = {}
                calibrated['co_level'] = np.median(co_readings) * self.calibration_factors['co_level']
                calibrated['co2_level'] = np.median(co2_readings) * self.calibration_factors['co2_level']
                for i, key in enumerate(['temperature', 'humidity', 'pm25', 'pm10'], 2):
                    raw = raw_values[i]
                    calib_value = raw * self.calibration_factors[key]
                    calibrated[key] = calib_value
                    self.log_calibration(key, raw, calib_value, self.calibration_factors[key])
                # Log calibration for median CO and CO2
                self.log_calibration('co_level', np.median(co_readings), calibrated['co_level'], self.calibration_factors['co_level'])
                self.log_calibration('co2_level', np.median(co2_readings), calibrated['co2_level'], self.calibration_factors['co2_level'])
                return calibrated
            else:
                raise Exception("No data received from XBee")
        except Exception as e:
            logger.error(f"Failed to read sensor data from XBee: {e}")
            return self.simulate_sensor_reading()

    def simulate_sensor_reading(self):
        sim_reading = {
            'co_level': random.uniform(0, 60), 'co2_level': random.uniform(300, 1200),
            'temperature': random.uniform(15, 40), 'humidity': random.uniform(30, 90),
            'pm25': random.uniform(0, 50), 'pm10': random.uniform(0, 200)
        }
        # Simulate redundancy for CO and CO2
        co_readings = [sim_reading['co_level'] * (1 + random.uniform(-0.1, 0.1)) for _ in range(2)]
        co2_readings = [sim_reading['co2_level'] * (1 + random.uniform(-0.1, 0.1)) for _ in range(2)]
        sim_reading['co_level'] = np.median(co_readings)
        sim_reading['co2_level'] = np.median(co2_readings)
        for key in sim_reading:
            raw_value = sim_reading[key] / self.calibration_factors[key]
            self.log_calibration(key, raw_value, sim_reading[key], self.calibration_factors[key])
        return sim_reading

    def write_to_csv(self, data):
        """Write sensor data to CSV with thread-safe locking."""
        column_mapping = {
            'timestamp': 'Timestamp',
            'location_id': 'SensorID',
            'co_level': 'MQ7_AirQuality',
            'co2_level': 'MQ135_AirQuality',
            'temperature': 'Temperature_C',
            'humidity': 'Humidity',
            'pm25': 'PM2_5',
            'pm10': 'PM10'
        }
        csv_data = {csv_col: data[db_col] for db_col, csv_col in column_mapping.items()}
        df = pd.DataFrame([csv_data])
        with csv_lock:
            if os.path.exists(CONFIG['CSV_PATH']):
                df.to_csv(CONFIG['CSV_PATH'], mode='a', header=False, index=False)
            else:
                df.to_csv(CONFIG['CSV_PATH'], mode='w', header=True, index=False)
        logger.info(f"Appended data to {CONFIG['CSV_PATH']}: {csv_data}")

    def import_csv_to_db(self, csv_path=CONFIG['CSV_PATH']):
        """Import sensor data from CSV to SQLite database."""
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return
            df = pd.read_csv(csv_path)
            column_mapping = {
                'Timestamp': 'timestamp',
                'SensorID': 'location_id',
                'MQ7_AirQuality': 'co_level',
                'MQ135_AirQuality': 'co2_level',
                'Temperature_C': 'temperature',
                'Humidity': 'humidity',
                'PM2_5': 'pm25',
                'PM10': 'pm10'
            }
            required_columns = set(column_mapping.keys())
            csv_columns = set(df.columns)
            if not required_columns.issubset(csv_columns):
                missing = required_columns - csv_columns
                logger.error(f"Missing CSV columns: {missing}")
                return
            df = df.rename(columns=column_mapping)
            numeric_cols = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['timestamp', 'location_id'] + numeric_cols)
            if df.empty:
                logger.warning("No valid data to import after cleaning")
                return
            with db_lock:
                conn = sqlite3.connect(self.db_path)
                df.to_sql('sensor_data', conn, if_exists='append', index=False)
                conn.close()
            logger.info(f"Imported {len(df)} rows from {csv_path} to database")
        except Exception as e:
            logger.error(f"CSV import failed: {e}")

    def collect_data(self, location_id, duration_seconds=10, interval_seconds=2):
        start_time = time.time()
        end_time = start_time + duration_seconds
        while time.time() < end_time and system.running:
            readings = self.get_sensor_reading()
            valid, errors = self.validate_readings(readings)
            if not valid:
                logger.error(f"Skipping invalid data for {location_id}: {errors}")
                time.sleep(interval_seconds)
                continue
            timestamp = datetime.now()
            data = {'timestamp': timestamp, 'location_id': location_id, **readings}
            # Write to SQLite
            with db_lock:
                conn = sqlite3.connect(self.db_path)
                df = pd.DataFrame([data])
                df.to_sql('sensor_data', conn, if_exists='append', index=False)
                conn.close()
            # Write to CSV
            self.write_to_csv(data)
            logger.info(f"Recorded data at {timestamp} for {location_id}: {readings}")
            # Emit to dashboard
            socketio.emit('sensor_data', get_sensor_data().to_dict(orient='records'))
            socketio.emit('alerts', get_recent_alerts())
            socketio.emit('gauges', get_gauge_data())
            historical_data = self.get_historical_data()
            if len(historical_data) >= 24:
                sequence = historical_data[-24:][['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']].values
                timestamp = historical_data[-24:]['timestamp'].values
                pred, _ = system.forecaster.predict(sequence, timestamp, steps=1)
                pred_data = {
                    'co_level': float(pred[0][0]), 'co2_level': float(pred[0][1]), 'temperature': float(pred[0][2]),
                    'humidity': float(pred[0][3]), 'pm25': float(pred[0][4]), 'pm10': float(pred[0][5])
                }
                socketio.emit('predictions', pred_data)
            time.sleep(interval_seconds)

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
        self.thresholds = THRESHOLDS
        self.led_pin = CONFIG['GPIO_LED_PIN']
        self.buzzer_pin = CONFIG['GPIO_BUZZER_PIN']
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
        logger.info("Anomaly detector trained")

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
        GPIO.cleanup([self.led_pin, self.buzzer_pin])

# --- Emergency Response Section ---
class EmergencyResponse:
    def __init__(self):
        self.ventilation_pin = CONFIG['GPIO_VENTILATION_PIN']
        self.config = {
            'emergency_contacts': ['supervisor@mine.com'],
            'phone_numbers': ['sip:supervisor@sip.mine.com'],
            'evacuation_zones': ['mine_shaft_1', 'mine_shaft_2', 'ventilation_area'],
            'ventilation_threshold': VENTILATION_THRESHOLDS
        }
        self.emergency_active = False
        self.evacuation_in_progress = False
        self._setup_sip()

    def _setup_sip(self):
        try:
            self.lib = pj.Lib()
            self.lib.init()
            self.transport = self.lib.create_transport(pj.TransportType.UDP)
            self.lib.start()
            self.acc = self.lib.create_account(pj.AccountConfig("sip.mine.com", "alert-system", "password"))
            logger.info("SIP client initialized")
        except Exception as e:
            logger.error(f"SIP initialization failed: {e}")

    def activate_ventilation(self):
        GPIO.output(self.ventilation_pin, GPIO.HIGH)
        logger.info("[EMERGENCY] Ventilation system activated")

    def deactivate_ventilation(self):
        GPIO.output(self.ventilation_pin, GPIO.LOW)
        logger.info("[EMERGENCY] Ventilation system deactivated")

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
                logger.info(f"Email alert sent to {contact}")
            except Exception as e:
                logger.error(f"Email failed: {e}")
        for number in self.config['phone_numbers']:
            try:
                call = self.acc.make_call(number, pj.CallOpParam())
                logger.info(f"VoIP call initiated to {number}")
                time.sleep(5)
                call.hangup()
            except Exception as e:
                logger.error(f"VoIP call failed: {e}")

    def initiate_evacuation(self, danger_zones):
        if not self.evacuation_in_progress:
            self.evacuation_in_progress = True
            message = f"EMERGENCY EVACUATION REQUIRED in zones: {', '.join(danger_zones)}"
            self.send_alert(message, level='critical')
            logger.critical("[EMERGENCY] Evacuation procedures initiated")
            threading.Thread(target=self.monitor_evacuation).start()

    def monitor_evacuation(self):
        pass  # Placeholder for RFID tracking

    def assess_situation(self, sensor_data, predictions, anomalies):
        responses_triggered = []
        if (sensor_data.get('co_level', 0) > self.config['ventilation_threshold']['co_level'] or
                sensor_data.get('co2_level', 0) > self.config['ventilation_threshold']['co2_level']):
            self.activate_ventilation()
            responses_triggered.append('ventilation')
        if anomalies:
            self.send_alert(f"Anomaly detected: {json.dumps(sensor_data)}")
            responses_triggered.append('alert')
        if predictions is not None:
            dangerous_predictions = any(
                predictions[0][i] > self.config['ventilation_threshold'].get(key, float('inf'))
                for i, key in enumerate(['co_level', 'co2_level'])
            )
            if dangerous_predictions:
                self.initiate_evacuation(['mine_shaft_1'])
                responses_triggered.append('evacuation')
        return responses_triggered

    def cleanup(self):
        self.deactivate_ventilation()
        GPIO.cleanup(self.ventilation_pin)
        try:
            self.lib.destroy()
            logger.info("SIP client destroyed")
        except:
            pass

# --- Forecasting Section ---
class MineDataForecaster:
    def __init__(self, sequence_length=24, prediction_length=1, checkpoint_dir='./checkpoints'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length, context_length=sequence_length,
            input_size=6, num_time_features=4, d_model=64, n_heads=8
        )
        self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        self.feature_scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=200)
        self.rf_trained = False
        self.data_buffers = {}

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
        return (np.array(sequences), np.array(timestamps)), np.array(targets)

    def _encode_timestamp(self, timestamps):
        time_features = np.zeros((len(timestamps), 4))
        for i, ts in enumerate(timestamps):
            dt = pd.Timestamp(ts)
            time_features[i, 0] = dt.hour / 24.0
            time_features[i, 1] = dt.dayofweek / 6.0
            time_features[i, 2] = (dt.month - 1) / 11.0
            time_features[i, 3] = (dt.year - 2000) / 100.0
        return time_features

    def train(self, historical_data, epochs=10, batch_size=32, validation_split=0.2):
        logger.info("Training forecasting models...")
        # Train Transformer
        (X, timestamps), y = self.prepare_sequence(historical_data, multi_step=True)
        split_idx = int(len(X[0]) * (1 - validation_split))
        train_dataset = TensorDataset(
            torch.FloatTensor(X[0][:split_idx]), torch.FloatTensor(timestamps[:split_idx]),
            torch.FloatTensor(y[:split_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X[0][split_idx:]), torch.FloatTensor(timestamps[split_idx:]),
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
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
        # Train Random Forest
        X_rf = historical_data[['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']].values
        y_rf = np.where(
            (X_rf[:, 0] > THRESHOLDS['co_level']) | (X_rf[:, 1] > THRESHOLDS['co2_level']) |
            (X_rf[:, 2] > THRESHOLDS['temperature']) | (X_rf[:, 3] > THRESHOLDS['humidity']) |
            (X_rf[:, 4] > THRESHOLDS['pm25']) | (X_rf[:, 5] > THRESHOLDS['pm10']), 1, 0
        )
        self.rf_model.fit(X_rf, y_rf)
        self.rf_trained = True
        logger.info("Random Forest model trained")

    def retrain_models(self, historical_data):
        """Retrain all models and return performance metrics."""
        if len(historical_data) < 24:
            logger.warning("Insufficient data for retraining (<24 records)")
            return None
        logger.info(f"Retraining models with {len(historical_data)} records")
        # Retrain Isolation Forest (via alarm system)
        system.alarm_system.train_anomaly_detector(historical_data)
        # Retrain Random Forest and Transformer
        self.train(historical_data, epochs=5)  # Reduced epochs for retraining
        # Evaluate models
        metrics = self.evaluate(historical_data)
        logger.info(f"Retraining complete. Metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_time, batch_y in val_loader:
                batch_X, batch_time, batch_y = (batch_X.to(self.device), batch_time.to(self.device), batch_y.to(self.device))
                outputs = self.model(past_values=batch_X, past_time_features=batch_time).prediction_outputs
                val_loss += criterion(outputs, batch_y).item()
        return val_loss / len(val_loader)

    def predict(self, sequence, timestamp, steps=1):
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
        return predictions[0], outputs.prediction_outputs.std(dim=-1).cpu().numpy()[0]

    def predict_rf(self, sensor_data, sensor_id):
        if sensor_id not in self.data_buffers:
            self.data_buffers[sensor_id] = []
        self.data_buffers[sensor_id].append(sensor_data)
        if len(self.data_buffers[sensor_id]) > 10:
            self.data_buffers[sensor_id].pop(0)
        df = pd.DataFrame(self.data_buffers[sensor_id])
        features = df.drop(columns=['timestamp', 'location_id']).mean().to_dict()
        X = np.array([[features[k] for k in ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']]])
        if self.rf_trained:
            prediction = self.rf_model.predict(X)[0]
            probability = self.rf_model.predict_proba(X)[0][1]
        else:
            alerts = [k for k, v in sensor_data.items() if k in THRESHOLDS and v > THRESHOLDS[k]]
            prediction = 1 if alerts else 0
            probability = 1.0 if alerts else 0.0
        return prediction, probability

    def evaluate(self, test_data):
        (X_test, timestamps), y_test = self.prepare_sequence(test_data, multi_step=True)
        predictions = []
        for seq, time in zip(X_test, timestamps):
            pred, _ = self.predict(seq, time)
            predictions.append(pred)
        predictions = np.array(predictions)
        y_test = y_test.squeeze(1)
        features = ['co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10']
        metrics = {}
        for i, feature in enumerate(features):
            metrics[feature] = {
                'MAE': mean_absolute_error(y_test[:, i].flatten(), predictions[:, i].flatten()),
                'RMSE': np.sqrt(mean_squared_error(y_test[:, i].flatten(), predictions[:, i].flatten())),
                'R2': r2_score(y_test[:, i].flatten(), predictions[:, i].flatten())
            }
        return metrics

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'val_loss': val_loss}
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

# --- TCP Server Section ---
class MineServer:
    def __init__(self, alarm_system, emergency_system, forecaster):
        self.alarm_system = alarm_system
        self.emergency_system = emergency_system
        self.forecaster = forecaster
        self.data_buffer = []

    def parse_tcp_data(self, raw_data):
        parsed_data = {}
        for key, pattern in PATTERNS.items():
            match = re.search(pattern, raw_data)
            if match:
                value = match.group(1)
                parsed_data[key] = float(value) if key != 'SensorID' else value
        if 'SensorID' in parsed_data and len(parsed_data) > 1:
            parsed_data['timestamp'] = datetime.now()
            parsed_data['location_id'] = parsed_data.pop('SensorID')
            parsed_data = {k: parsed_data[k] for k in ['timestamp', 'location_id', 'co_level', 'co2_level', 'temperature', 'humidity', 'pm25', 'pm10'] if k in parsed_data}
            return parsed_data
        logger.warning(f"Failed to parse TCP data: {raw_data}")
        return None

    def save_to_db(self):
        if not self.data_buffer:
            return
        with db_lock:
            conn = sqlite3.connect(CONFIG['DB_PATH'])
            df = pd.DataFrame(self.data_buffer)
            df.to_sql('sensor_data', conn, if_exists='append', index=False)
            conn.close()
        system.collector.write_to_csv(self.data_buffer[-1])
        logger.info(f"Saved {len(self.data_buffer)} rows to database and CSV")
        socketio.emit('sensor_data', get_sensor_data().to_dict(orient='records'))
        socketio.emit('alerts', get_recent_alerts())
        socketio.emit('gauges', get_gauge_data())
        self.data_buffer.clear()

    def send_feedback(self, command):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(command.encode(), (CONFIG['FEEDBACK_IP'], CONFIG['FEEDBACK_PORT']))
            sock.close()
            logger.info(f"Sent feedback: {command}")
        except Exception as e:
            logger.error(f"Feedback error: {e}")

    def handle_client(self, connection, client_address):
        try:
            while True:
                raw_data = connection.recv(1024).decode('utf-8').strip()
                if raw_data:
                    logger.info(f"Received from {client_address}: {raw_data}")
                    parsed_data = self.parse_tcp_data(raw_data)
                    if parsed_data:
                        valid, errors = system.collector.validate_readings(parsed_data)
                        if not valid:
                            logger.error(f"Skipping invalid TCP data: {errors}")
                            continue
                        self.data_buffer.append(parsed_data)
                        alerts = self.alarm_system.check_thresholds(parsed_data)
                        is_anomaly = self.alarm_system.check_anomalies(parsed_data)
                        prediction, probability = self.forecaster.predict_rf(parsed_data, parsed_data['location_id'])
                        command = "GREEN"
                        if probability < 0.3:
                            command = "GREEN"
                        elif 0.3 <= probability < 0.7:
                            command = "YELLOW"
                        else:
                            command = "RED_BUZZER"
                            self.alarm_system.trigger_alarm()
                            self.emergency_system.assess_situation(parsed_data, None, is_anomaly)
                        self.send_feedback(f"{parsed_data['location_id']}:{command}")
                        logger.info(f"Location {parsed_data['location_id']} - Prediction: {'Warning' if prediction else 'Safe'}, Probability: {probability:.2f}")
                else:
                    logger.info(f"Client {client_address} disconnected")
                    break
        except Exception as e:
            logger.error(f"Client error {client_address}: {e}")
        finally:
            connection.close()

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((CONFIG['TCP_HOST'], CONFIG['TCP_PORT']))
        server_socket.listen(5)
        logger.info(f"TCP server listening on {CONFIG['TCP_HOST']}:{CONFIG['TCP_PORT']}")
        last_save_time = time.time()
        try:
            while system.running:
                server_socket.settimeout(1.0)
                try:
                    connection, client_address = server_socket.accept()
                    threading.Thread(target=self.handle_client, args=(connection, client_address)).start()
                except socket.timeout:
                    pass
                if time.time() - last_save_time >= CONFIG['LOG_INTERVAL']:
                    self.save_to_db()
                    last_save_time = time.time()
        except Exception as e:
            logger.error(f"TCP server error: {e}")
        finally:
            server_socket.close()

# --- Web Dashboard Section ---
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

def get_gauge_data():
    df = get_sensor_data().tail(1)
    if df.empty:
        return {}
    latest = df.iloc[0]
    gauges = {}
    for key, threshold in THRESHOLDS.items():
        gauges[key] = {
            'value': latest[key],
            'max': threshold * 1.5,
            'threshold': threshold
        }
    return gauges

@app.route('/')
def index():
    return render_template('dashboard.html')

# --- System Management Section ---
class MineMonitoringSystem:
    def __init__(self):
        self.collector = MineDataCollector(use_sensors=True)
        self.emergency_system = EmergencyResponse()
        self.alarm_system = MineAlarmSystem(self.emergency_system)
        self.forecaster = MineDataForecaster()
        self.server = MineServer(self.alarm_system, self.emergency_system, self.forecaster)
        self.running = False

    def simulate_optical_fiber(self, data):
        logger.info(f"Simulated optical fiber transmission: {data}")
        return data

    def schedule_retraining(self):
        """Schedule weekly model retraining every Sunday at midnight."""
        def retrain_job():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            historical_data = self.collector.get_historical_data(
                start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
                end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
            )
            metrics = self.forecaster.retrain_models(historical_data)
            if metrics:
                logger.info("Weekly retraining completed successfully")
            else:
                logger.warning("Weekly retraining skipped due to insufficient data")
        schedule.every().sunday.at("00:00").do(retrain_job)
        threading.Thread(target=self.run_scheduler, daemon=True).start()

    def run_scheduler(self):
        while self.running:
            schedule.run_pending()
            time.sleep(60)

    def start_data_collection(self):
        while self.running:
            for location in ['mine_shaft_1', 'mine_shaft_2', 'ventilation_area']:
                readings = self.collector.get_sensor_reading()
                readings = self.simulate_optical_fiber(readings)
                self.collector.collect_data(location, duration_seconds=10, interval_seconds=2)
                alerts = self.alarm_system.check_thresholds(readings)
                if alerts:
                    logger.warning(f"Threshold alerts at {location}: {alerts}")
                    self.alarm_system.trigger_alarm()
                is_anomaly = self.alarm_system.check_anomalies(readings)
                if is_anomaly:
                    logger.warning(f"Anomaly detected at {location}: {readings}")
                    self.alarm_system.trigger_alarm(duration=1)
                historical_data = self.collector.get_historical_data()
                predictions = None
                if len(historical_data) >)self.sequence_length:
                    sequence = historical_data[-self.sequence_length:][['co_level', 'co2_level', 'temperature',
                                                    'humidity', 'pm25', 'pm10']].values
                    timestamp = historical_data[-self.sequence_length:]['timestamp'].values
                    predictions, _ = self.forecaster.predict(sequence, timestamp)
                    logger.info(f"Predictions for {location}: {predictions}")
                responses = self.emergency_system.assess_situation(readings, predictions, is_anomaly)
                if responses:
                    logger.critical(f"Emergency responses triggered at {location}: {responses}")
                time.sleep(5)

    def train_models(self):
        logger.info("Training models with historical data...")
        historical_data = self.collector.get_historical_data()
        if len(historical_data) > 24:
            self.alarm_system.train_anomaly_detector(historical_data)
            self.forecaster.train(historical_data)
            metrics = self.forecaster.evaluate(historical_data)
            logger.info(f"Initial Model Metrics: {metrics}")

    def start(self):
        self.running = True
        logger.info("Starting mine monitoring system...")
        self.collector.import_csv_to_db()
        self.train_models()
        self.schedule_retraining()
        threading.Thread(target=self.start_data_collection, daemon=True).start()
        threading.Thread(target=self.server.start, daemon=True).start()
        socketio.run(app, host='0.0.0.0', port=8080)

    def stop(self):
        self.running = False
        logger.info("Shutting down mine monitoring system...")
        self.alarm_system.cleanup()
        self.emergency_system.cleanup()

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

if __name__ == "__main__":
    system = MineMonitoringSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()
