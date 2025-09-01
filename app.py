import time
import pandas as pd
import joblib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
from threading import Lock, Event

# --- Backend Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
thread = None
thread_lock = Lock()
stop_event = Event() # Used to safely stop the simulation thread

# --- Modular Prediction Engine (Simplified) ---
class PredictionEngine:
    def __init__(self):
        self.rf_model = None
        self.encoder = None
        self.load_models()

    def load_models(self):
        print("Loading machine learning models and utilities...")
        try:
            self.rf_model = joblib.load('random_forest_model.joblib')
            self.encoder = joblib.load('label_encoder.joblib')
            print("Random Forest model loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model files: {e}")

    def predict(self, feature_vector, model_type='random_forest'):
        if feature_vector is None:
            return "Processing Error"
        if model_type == 'random_forest' and self.rf_model:
            pred_encoded = self.rf_model.predict(feature_vector)[0]
            prediction = self.encoder.inverse_transform([pred_encoded])[0]
            return prediction
        else:
            return "Model Not Loaded"

# --- Data Streaming from Shuffled Features ---
def stream_simulation_data(selected_model, selected_dataset):
    engine = PredictionEngine()
    
    if not os.path.exists(selected_dataset):
        print(f"Error: Dataset '{selected_dataset}' not found.")
        socketio.emit('simulation_status', {'status': 'error', 'message': 'Dataset not found'})
        return

    df = pd.read_csv(selected_dataset)
    print(f"Starting dynamic simulation from '{selected_dataset}' using '{selected_model}'...")
    
    feature_columns = engine.rf_model.feature_names_in_

    for index, row in df.head(100).iterrows():
        # Check if the stop event has been set
        if stop_event.is_set():
            break

        # --- THIS IS THE CORRECTED LOGIC FOR THE KEYERROR ---
        # Convert the row (a Series) to a one-row DataFrame, then select the columns
        feature_vector = row.to_frame().T[feature_columns]
        
        true_label = row['attack_type']
        prediction = engine.predict(feature_vector, model_type=selected_model)

        socketio.emit('live_update', {
            'prediction': prediction,
            'true_label': true_label,
            'is_correct': prediction == true_label,
            'window_id': row.get('time_window', index),
            'timestamp': time.strftime('%H:%M:%S')
        })
        
        socketio.sleep(1)
            
    print("Data stream finished or was stopped.")
    socketio.emit('simulation_status', {'status': 'finished'})

# --- Flask Routes and WebSocket Events ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_simulation')
def handle_start_simulation(data):
    global thread
    with thread_lock:
        if thread is None or not thread.is_alive():
            stop_event.clear() # Clear the stop flag before starting
            print(f"Received start signal. Model: {data['model']}, Dataset: {data['dataset']}")
            thread = socketio.start_background_task(
                stream_simulation_data, data['model'], data['dataset']
            )
        else:
            print("A simulation is already running.")

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """New event handler to stop the simulation."""
    print("Received stop signal from client.")
    stop_event.set()

if __name__ == '__main__':
    print("Starting Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

