from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import sqlite3
import pandas as pd

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
    conn = sqlite3.connect('mine_data.db')
    df = pd.read_sql_query("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1000", conn)
    conn.close()
    return df

def get_recent_alerts():
    conn = sqlite3.connect('mine_data.db')
    # Assuming alerts are logged in sensor_data with a threshold check; adjust as needed
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
