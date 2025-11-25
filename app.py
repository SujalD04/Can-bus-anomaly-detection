import time
import pandas as pd
import joblib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
from threading import Lock, Event
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import psutil  # <-- for system metrics

# --- Backend Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
thread = None
thread_lock = Lock()
stop_event = Event()  # Used to safely stop the simulation thread

# --- Modular Prediction Engine ---
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
            print("✅ Random Forest model loaded successfully.")
        except FileNotFoundError as e:
            print(f"❌ Error loading model files: {e}")

    def predict(self, feature_vector, model_type='random_forest'):
        if feature_vector is None:
            return "Processing Error"
        if model_type == 'random_forest' and self.rf_model:
            pred_encoded = self.rf_model.predict(feature_vector)[0]
            prediction = self.encoder.inverse_transform([pred_encoded])[0]
            return prediction
        else:
            return "Model Not Loaded"

# --- Unified Stream: Predictions + System Metrics ---
def stream_simulation_data(selected_model, selected_dataset):
    engine = PredictionEngine()

    if not os.path.exists(selected_dataset):
        print(f"❌ Dataset '{selected_dataset}' not found.")
        socketio.emit('simulation_status', {'status': 'error', 'message': 'Dataset not found'})
        return

    df = pd.read_csv(selected_dataset)
    print(f"\n🚗 Streaming simulation from '{selected_dataset}' using '{selected_model}'...")
    feature_columns = engine.rf_model.feature_names_in_

    y_true_list, y_pred_list = [], []

    for index, row in df.head(100).iterrows():
        if stop_event.is_set():
            break

        feature_vector = row.to_frame().T[feature_columns]
        true_label = row['attack_type']
        prediction = engine.predict(feature_vector, model_type=selected_model)

        # --- System metrics ---
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # --- Store predictions ---
        y_true_list.append(true_label)
        y_pred_list.append(prediction)

        # --- Running accuracy ---
        running_acc = accuracy_score(y_true_list, y_pred_list)

        # --- Emit live data ---
        socketio.emit('live_update', {
            'prediction': prediction,
            'true_label': true_label,
            'is_correct': prediction == true_label,
            'window_id': row.get('time_window', index),
            'timestamp': time.strftime('%H:%M:%S'),
            'accuracy': round(running_acc, 4),
            # system metrics
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent
        })

        print(f"[{index+1}] True: {true_label:<15} | Pred: {prediction:<15} | "
              f"{'✅' if prediction == true_label else '❌'} | Acc: {running_acc:.3f} | "
              f"CPU: {cpu_percent}% | Mem: {memory_percent}% | Disk: {disk_percent}%")

        socketio.sleep(1)

    print("\n✅ Streaming finished or was stopped.")
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
            stop_event.clear()
            print(f"\n🟢 Starting simulation. Model: {data['model']}, Dataset: {data['dataset']}")
            thread = socketio.start_background_task(
                stream_simulation_data, data['model'], data['dataset']
            )
        else:
            print("⚠️ A simulation is already running.")

@socketio.on('stop_simulation')
def handle_stop_simulation():
    print("🛑 Stop signal received.")
    stop_event.set()

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
