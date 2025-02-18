
import threading
from main import MineDataCollector
from alarm_system import MineAlarmSystem
from emergency_response import EmergencyResponse
from forecasting import MineDataForecaster
from dashboard import app
import time

class MineMonitoringSystem:
    def __init__(self):
        self.collector = MineDataCollector(use_sensors=True)
        self.alarm_system = MineAlarmSystem()
        self.emergency_system = EmergencyResponse()
        self.forecaster = MineDataForecaster()
        self.running = False
        
    def start_data_collection(self):
        while self.running:
            for location in ['mine_shaft_1', 'mine_shaft_2', 'ventilation_area']:
                # Get current readings
                readings = self.collector.get_sensor_reading()
                
                # Store data
                self.collector.collect_data(location, duration_seconds=10, interval_seconds=2)
                
                # Check for threshold violations
                alerts = self.alarm_system.check_thresholds(readings)
                if alerts:
                    print(f"ALERTS: {alerts}")
                    self.alarm_system.trigger_alarm()
                
                # Check for anomalies
                is_anomaly = self.alarm_system.check_anomalies(readings)
                if is_anomaly:
                    print("ANOMALY DETECTED!")
                    self.alarm_system.trigger_alarm(duration=1)
                
                # Get predictions if enough historical data exists
                historical_data = self.collector.get_historical_data()
                predictions = None
                if len(historical_data) > 24:
                    sequence = historical_data[-24:][['co_level', 'co2_level', 'temperature', 
                                                    'humidity', 'pm25', 'pm10']].values
                    predictions = self.forecaster.predict(sequence)
                
                # Assess emergency responses
                responses = self.emergency_system.assess_situation(readings, predictions, is_anomaly)
                if responses:
                    print(f"Emergency responses triggered: {', '.join(responses)}")
                
                time.sleep(5)  # Wait before next location
    
    def train_models(self):
        print("Training models with historical data...")
        historical_data = self.collector.get_historical_data()
        
        if len(historical_data) > 24:
            # Train anomaly detector
            self.alarm_system.train_anomaly_detector(historical_data)
            print("Anomaly detector trained")
            
            # Train forecaster
            self.forecaster.train(historical_data)
            print("Forecasting model trained")
            
            # Evaluate forecasting model
            metrics = self.forecaster.evaluate(historical_data)
            print("Forecasting Model Metrics:")
            print(f"MAE: {metrics['MAE']:.4f}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"R2: {metrics['R2']:.4f}")
    
    def start(self):
        self.running = True
        
        # Train models with existing data
        self.train_models()
        
        # Start data collection in a separate thread
        collection_thread = threading.Thread(target=self.start_data_collection)
        collection_thread.daemon = True
        collection_thread.start()
        
        # Start the dashboard
        app.run(host='0.0.0.0', port=8080)
    
    def stop(self):
        self.running = False
        self.alarm_system.cleanup()
        self.emergency_system.cleanup()

if __name__ == "__main__":
    system = MineMonitoringSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()
