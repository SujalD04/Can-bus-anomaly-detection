import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_random_forest(feature_file_path):
    """
    Loads feature data, splits it, trains a Random Forest classifier,
    and saves the model and label encoder.

    Args:
        feature_file_path (str): Path to the feature data CSV file.
    """
    print("\n--- Starting Random Forest Training ---")
    print(f"Loading features from {feature_file_path}...")
    features_df = pd.read_csv(feature_file_path)

    # --- 1. Prepare Features (X) and Labels (y) ---
    X = features_df.drop(columns=['time_window', 'car_model', 'attack_type'])
    y_text = features_df['attack_type']

    # --- 2. Encode Labels ---
    # Convert text labels (e.g., 'DoS-attacks') into integers (e.g., 0, 1, 2)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    
    # IMPORTANT: Save the encoder so we can decode the model's predictions later
    joblib.dump(encoder, 'label_encoder.joblib')
    print(f"Labels encoded. Found {len(encoder.classes_)} classes: {encoder.classes_}")

    # --- 3. Split Data into Training and Testing Sets ---
    # This is crucial for supervised learning to evaluate the model properly.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 4. Train the Model ---
    print("Training Random Forest model...")
    # n_jobs=-1 uses all available CPU cores to speed up training.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest training complete.")
    
    # --- 5. Save the Model ---
    model_filename = 'random_forest_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully to {model_filename}")

    # We will evaluate on the test set in the next step.
    return X_test, y_test, encoder

def train_isolation_forest(feature_file_path):
    """
    Loads feature data, trains an Isolation Forest model on normal traffic,
    and saves the trained model to a file.

    Args:
        feature_file_path (str): Path to the feature data CSV file.
    """
    print(f"Loading features from {feature_file_path}...")
    # Load the features we created in the last step
    features_df = pd.read_csv(feature_file_path)

    # --- Data Preparation ---
    # We will train the model ONLY on data labeled as 'attack-free'
    # NOTE: Your normal data might have a different name, e.g., 'normal' or 'benign'.
    # Update 'attack-free' if your label is different.
    normal_df = features_df[features_df['attack_type'] == 'attack-free'].copy()

    # Prepare the training data by dropping non-feature columns
    # 'time_window' is the index, and the labels are not features
    X_train = normal_df.drop(columns=['time_window', 'car_model', 'attack_type'])
    
    if X_train.empty:
        print("Error: No 'attack-free' data found for training. Please check your labels.")
        return

    print(f"Training Isolation Forest model on {len(X_train)} normal samples...")

    # --- Model Training ---
    # n_estimators is the number of trees; contamination is the expected % of anomalies
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    
    # Fit the model to our normal data
    model.fit(X_train)
    
    print("Model training complete.")

    # --- Save the Model ---
    # Saving the model allows us to use it for predictions later without retraining
    model_filename = 'isolation_forest_model.joblib'
    joblib.dump(model, model_filename)
    
    print(f"Model saved successfully to {model_filename}")

# --- Example Usage ---
if __name__ == '__main__':
    feature_file = 'features.csv'
    try:
        # First, train the Isolation Forest
        train_isolation_forest(feature_file)
        
        # Second, train the Random Forest
        train_random_forest(feature_file)

    except FileNotFoundError:
        print(f"Error: '{feature_file}' not found.")
        print("Please run the feature_engineering.py script first.")