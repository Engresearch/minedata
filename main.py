import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import random
import serial
from digi.xbee.devices import XBeeDevice

class MineDataCollector:
    def __init__(self, db_path='mine_data.db', use_sensors=False, port='/dev/ttyUSB0', baud_rate=9600):
        self.db_path = db_path
        self.use_sensors = use_sensors
        self.port = port
        self.baud_rate = baud_rate
        self.calibration_factors = {
            'co_level': 0.1,  # MQ7 calibration factor
            'co2_level': 0.2,  # MQ135 calibration factor
            'temperature': 1.0,  # DHT11 (no adjustment needed typically)
            'humidity': 1.0,   # DHT11
            'pm25': 1.0,       # SDS011
            'pm10': 1.0        # SDS011
        }
        self.setup_database()

        if use_sensors:
            try:
                self.xbee = XBeeDevice(self.port, self.baud_rate)
                self.xbee.open()
            except Exception as e:
                print(f"Warning: XBee initialization failed: {e}")
                self.use_sensors = False

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_log (
                timestamp DATETIME,
                sensor TEXT,
                raw_value FLOAT,
                calibrated_value FLOAT,
                calibration_factor FLOAT
            )
        ''')
        conn.commit()
        conn.close()

    def log_calibration(self, sensor, raw_value, calibrated_value, factor):
        conn = sqlite3.connect(self.db_path)
        data = {
            'timestamp': datetime.now(),
            'sensor': sensor,
            'raw_value': raw_value,
            'calibrated_value': calibrated_value,
            'calibration_factor': factor
        }
        df = pd.DataFrame([data])
        df.to_sql('calibration_log', conn, if_exists='append', index=False)
        conn.close()

    def get_sensor_reading(self):
        """Get readings from Arduino via Zigbee with calibration"""
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
                raise Exception("No data received")
        except Exception as e:
            print(f"Sensor reading failed: {e}")
            return self.simulate_sensor_reading()

    def simulate_sensor_reading(self):
        sim_reading = {
            'co_level': random.uniform(0, 50),
            'co2_level': random.uniform(300, 1000),
            'temperature': random.uniform(15, 35),
            'humidity': random.uniform(30, 90),
            'pm25': random.uniform(0, 35),
            'pm10': random.uniform(0, 150)
        }
        for key in sim_reading:
            self.log_calibration(key, sim_reading[key] / self.calibration_factors[key], 
                               sim_reading[key], self.calibration_factors[key])
        return sim_reading

    def collect_data(self, location_id, duration_seconds=60, interval_seconds=5):
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
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM sensor_data"
        if start_date and end_date:
            query += f" WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
