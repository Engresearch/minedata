import threading
import smtplib
import json
from email.mime.text import MIMEText
import RPi.GPIO as GPIO
from datetime import datetime
import pjsua2 as pj

class EmergencyResponse:
    def __init__(self):
        self.ventilation_pin = 24
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
        # Email alert
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

        # VoIP call
        for number in self.config['phone_numbers']:
            try:
                call = self.acc.make_call(number, pj.CallOpParam())
                print(f"VoIP call initiated to {number}")
                time.sleep(5)  # Simulate call duration
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
        pass  # Implement with RFID tracking if available

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
