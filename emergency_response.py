
import threading
import smtplib
import json
from email.mime.text import MIMEText
import RPi.GPIO as GPIO
from datetime import datetime

class EmergencyResponse:
    def __init__(self):
        # GPIO setup for ventilation control
        self.ventilation_pin = 24
        GPIO.setup(self.ventilation_pin, GPIO.OUT)
        
        # Load configuration
        self.config = {
            'emergency_contacts': ['supervisor@mine.com'],
            'evacuation_zones': ['shaft_1', 'shaft_2', 'ventilation_area'],
            'ventilation_threshold': {
                'co_level': 30.0,
                'co2_level': 800.0
            }
        }
        
        # Initialize status
        self.emergency_active = False
        self.evacuation_in_progress = False
    
    def activate_ventilation(self):
        """Activate emergency ventilation system"""
        GPIO.output(self.ventilation_pin, GPIO.HIGH)
        print("[EMERGENCY] Ventilation system activated")
    
    def deactivate_ventilation(self):
        """Deactivate emergency ventilation system"""
        GPIO.output(self.ventilation_pin, GPIO.LOW)
        print("[EMERGENCY] Ventilation system deactivated")
    
    def send_alert(self, message, level='warning'):
        """Send email alerts to emergency contacts"""
        for contact in self.config['emergency_contacts']:
            msg = MIMEText(message)
            msg['Subject'] = f'MINE ALERT [{level.upper()}]: Emergency Situation'
            msg['From'] = 'mine-monitoring@mine.com'
            msg['To'] = contact
            
            try:
                with smtplib.SMTP('smtp.mine.com', 587) as server:
                    server.starttls()
                    server.login('alert-system', 'password')  # Use environment variables in production
                    server.send_message(msg)
                print(f"Alert sent to {contact}")
            except Exception as e:
                print(f"Failed to send alert: {e}")
    
    def initiate_evacuation(self, danger_zones):
        """Initiate evacuation procedures"""
        if not self.evacuation_in_progress:
            self.evacuation_in_progress = True
            message = f"EMERGENCY EVACUATION REQUIRED in zones: {', '.join(danger_zones)}"
            self.send_alert(message, level='critical')
            
            # Activate emergency lighting and sirens (implement with actual GPIO)
            print("[EMERGENCY] Evacuation procedures initiated")
            
            # Start monitoring evacuation progress in separate thread
            threading.Thread(target=self.monitor_evacuation).start()
    
    def monitor_evacuation(self):
        """Monitor evacuation progress"""
        # In real implementation, this would track miners' locations
        # and ensure everyone has reached safety zones
        pass
    
    def assess_situation(self, sensor_data, predictions, anomalies):
        """Assess situation and trigger appropriate responses"""
        responses_triggered = []
        
        # Check immediate dangers from sensor data
        if (sensor_data['co_level'] > self.config['ventilation_threshold']['co_level'] or
            sensor_data['co2_level'] > self.config['ventilation_threshold']['co2_level']):
            self.activate_ventilation()
            responses_triggered.append('ventilation')
        
        # Check anomalies
        if anomalies:
            message = f"Anomaly detected: {json.dumps(sensor_data)}"
            self.send_alert(message)
            responses_triggered.append('alert')
        
        # Check predictions
        dangerous_predictions = any(
            predictions[i] > self.config['ventilation_threshold'][key]
            for i, key in enumerate(['co_level', 'co2_level'])
        )
        
        if dangerous_predictions:
            self.initiate_evacuation(['shaft_1'])  # Specify zones based on predictions
            responses_triggered.append('evacuation')
        
        return responses_triggered
    
    def cleanup(self):
        """Cleanup GPIO and active procedures"""
        self.deactivate_ventilation()
        GPIO.cleanup(self.ventilation_pin)
