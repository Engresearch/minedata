
import RPi.GPIO as GPIO
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import time
from datetime import datetime

class MineAlarmSystem:
    def __init__(self):
        # Define thresholds
        self.thresholds = {
            'co_level': 50.0,      # ppm
            'co2_level': 1000.0,   # ppm
            'temperature': 30.0,    # Celsius
            'humidity': 85.0,       # %
            'pm25': 35.0,          # µg/m³
            'pm10': 150.0          # µg/m³
        }
        
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self.led_pin = 18
        self.buzzer_pin = 23
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        
        # Initialize anomaly detector
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.model_trained = False
    
    def check_thresholds(self, readings):
        """Check if any readings exceed thresholds"""
        alerts = []
        for key, value in readings.items():
            if key in self.thresholds and value > self.thresholds[key]:
                alerts.append(f"{key} exceeds threshold: {value:.2f}")
        return alerts
    
    def trigger_alarm(self, duration=2):
        """Trigger LED and buzzer"""
        GPIO.output(self.led_pin, GPIO.HIGH)
        GPIO.output(self.buzzer_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.led_pin, GPIO.LOW)
        GPIO.output(self.buzzer_pin, GPIO.LOW)
    
    def train_anomaly_detector(self, historical_data):
        """Train the Isolation Forest model"""
        features = ['co_level', 'co2_level', 'temperature', 
                   'humidity', 'pm25', 'pm10']
        X = historical_data[features].values
        self.isolation_forest.fit(X)
        self.model_trained = True
    
    def check_anomalies(self, readings):
        """Check for anomalies in current readings"""
        if not self.model_trained:
            return False
            
        features = ['co_level', 'co2_level', 'temperature', 
                   'humidity', 'pm25', 'pm10']
        X = np.array([[readings[f] for f in features]])
        prediction = self.isolation_forest.predict(X)
        return prediction[0] == -1  # -1 indicates anomaly
    
    def cleanup(self):
        """Cleanup GPIO"""
        GPIO.cleanup()
