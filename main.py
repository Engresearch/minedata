
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import random  # For demo data generation

class MineDataCollector:
    def __init__(self, db_path='mine_data.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for sensor data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                timestamp DATETIME,
                location_id TEXT,
                co_level FLOAT,
                co2_level FLOAT,
                temperature FLOAT,
                humidity FLOAT,
                pm25 FLOAT,
                pm10 FLOAT
            )
        ''')
        conn.commit()
        conn.close()

    def __init__(self, db_path='mine_data.db', use_sensors=False):
        self.db_path = db_path
        self.use_sensors = use_sensors
        self.setup_database()
        if self.use_sensors:
            try:
                import board
                import adafruit_bmp280
                from smbus2 import SMBus
                
                # Initialize I2C
                i2c = board.I2C()
                self.temp_sensor = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)
                self.gas_bus = SMBus(1)  # Use I2C bus 1 on Raspberry Pi
            except Exception as e:
                print(f"Warning: Sensor initialization failed: {e}")
                self.use_sensors = False

    def get_sensor_reading(self):
        """Get real sensor readings from connected hardware"""
        try:
            if not self.use_sensors:
                return self.simulate_sensor_reading()
            
            # Read from actual sensors
            temp = self.temp_sensor.temperature
            
            # Read from gas sensors (modify addresses and calculation based on your sensors)
            co_raw = self.gas_bus.read_word_data(0x48, 0x00)  # Example I2C address
            co2_raw = self.gas_bus.read_word_data(0x49, 0x00)  # Example I2C address
            
            return {
                'co_level': co_raw * 0.1,  # Apply calibration factor
                'co2_level': co2_raw * 0.1,  # Apply calibration factor
                'temperature': temp,
                'humidity': 50.0,  # Add humidity sensor reading
                'pm25': 10.0,  # Add PM2.5 sensor reading
                'pm10': 20.0   # Add PM10 sensor reading
            }
        except Exception as e:
            print(f"Sensor reading failed: {e}")
            return self.simulate_sensor_reading()

    def simulate_sensor_reading(self):
        """Simulate sensor readings for demo purposes"""
        return {
            'co_level': random.uniform(0, 50),      # ppm
            'co2_level': random.uniform(300, 1000), # ppm
            'temperature': random.uniform(15, 35),   # Celsius
            'humidity': random.uniform(30, 90),      # %
            'pm25': random.uniform(0, 35),          # µg/m³
            'pm10': random.uniform(0, 150)          # µg/m³
        }

    def collect_data(self, location_id, duration_seconds=60, interval_seconds=5):
        """Collect data for specified duration at given interval"""
        conn = sqlite3.connect(self.db_path)
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            # Get sensor readings
            readings = self.get_sensor_reading()
            timestamp = datetime.now()
            
            # Store in database
            data = {
                'timestamp': timestamp,
                'location_id': location_id,
                **readings
            }
            
            df = pd.DataFrame([data])
            df.to_sql('sensor_data', conn, if_exists='append', index=False)
            print(f"Recorded data at {timestamp}: {readings}")
            
            # Check for threshold violations
            alerts = alarm_system.check_thresholds(readings)
            if alerts:
                print("ALERTS:", alerts)
                alarm_system.trigger_alarm()
            
            # Check for anomalies
            if alarm_system.check_anomalies(readings):
                print("ANOMALY DETECTED!")
                alarm_system.trigger_alarm(duration=1)
            
            time.sleep(interval_seconds)
        
        conn.close()

    def get_historical_data(self, start_date=None, end_date=None):
        """Retrieve historical data for analysis"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM sensor_data"
        if start_date and end_date:
            query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

def main():
    # Set use_sensors=True when running on Raspberry Pi with actual sensors
    collector = MineDataCollector(use_sensors=True)
    alarm_system = MineAlarmSystem()
    forecaster = MineDataForecaster()
    
    # Get historical data for training
    historical_data = collector.get_historical_data()
    
    if len(historical_data) > 24:  # Need at least 24 hours of data
        # Split data for training and testing
        train_size = int(len(historical_data) * 0.8)
        train_data = historical_data[:train_size]
        test_data = historical_data[train_size:]
        
        print("Training forecasting model...")
        forecaster.train(train_data)
        
        print("Evaluating model performance...")
        metrics = forecaster.evaluate(test_data)
        print("Model Metrics:")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2: {metrics['R2']:.4f}")
    
    print("Starting data collection...")
    
    # Initialize emergency response system
    emergency_system = EmergencyResponse()
    
    # Train anomaly detector with existing data
    historical_data = collector.get_historical_data()
    if len(historical_data) > 0:
        alarm_system.train_anomaly_detector(historical_data)
        print("Anomaly detector trained with historical data")
    
    # Collect data from three different locations
    locations = ['mine_shaft_1', 'mine_shaft_2', 'ventilation_area']
    
    for location in locations:
        print(f"\nCollecting data from {location}")
        
        # Get current readings
        readings = collector.get_sensor_reading()
        
        # Get predictions from forecaster
        if len(historical_data) > 24:
            sequence = historical_data[-24:][['co_level', 'co2_level', 'temperature', 
                                           'humidity', 'pm25', 'pm10']].values
            predictions = forecaster.predict(sequence)
        else:
            predictions = None
        
        # Check for anomalies
        is_anomaly = alarm_system.check_anomalies(readings)
        
        # Trigger emergency response if needed
        responses = emergency_system.assess_situation(readings, predictions, is_anomaly)
        if responses:
            print(f"Emergency responses triggered: {', '.join(responses)}")
        
        collector.collect_data(location, duration_seconds=30, interval_seconds=5)
    
    # Retrieve and display collected data
    historical_data = collector.get_historical_data()
    print("\nData collection summary:")
    print(f"Total records: {len(historical_data)}")
    print("\nSample of collected data:")
    print(historical_data.head())

if __name__ == "__main__":
    from system_manager import MineMonitoringSystem
    system = MineMonitoringSystem()
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.stop()
