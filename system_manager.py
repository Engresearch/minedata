import threading
from main import MineDataCollector
from alarm_system import MineAlarmSystem
from emergency_response import EmergencyResponse
from forecasting import MineDataForecaster
from dashboard import app
import time
import logging

# Configure logging
logging.basicConfig(
    filename='mine_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
