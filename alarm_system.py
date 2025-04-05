import RPi.GPIO as GPIO
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import time

class MineAlarmSystem:
    def __init__(self, emergency_system=None):
        self.thresholds = {
            'co_level': 50.0,
            'co2_level': 1000.0,
            'temperature': 30.0,
            'humidity': 85.0,
            'pm25': 35.0,
            'pm10': 150.0
        }
        GPIO.setmode(GPIO.BCM)
        self.led_pin = 18
        self.buzzer_pin = 23
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
