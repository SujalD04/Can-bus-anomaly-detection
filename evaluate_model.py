import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_isolation_forest(model_path, feature_path):
    """
    Evaluates the trained Isolation Forest model.
    """
    print("--- Evaluating Isolation Forest (Unsupervised Anomaly Detection) ---")
    
    model = joblib.load(model_path)
    features_df = pd.read_csv(feature_path)

    X_test = features_df.drop(columns=['time_window', 'car_model', 'attack_type'])
    # Convert labels to binary format (1 for normal, -1 for anomaly)
    y_true = features_df['attack_type'].apply(lambda x: 1 if x == 'attack-free' else -1)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    print("\n--- Isolation Forest Report ---")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Anomaly (-1)', 'Normal (1)']))


# --- NEW FUNCTION FOR RANDOM FOREST ---
def evaluate_random_forest(model_path, feature_path, encoder_path):
    """
    Evaluates the trained Random Forest model.
    """
    print("\n\n--- Evaluating Random Forest (Supervised Classification) ---")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    features_df = pd.read_csv(feature_path)

    # Prepare data, keeping the multi-class labels
    X_test = features_df.drop(columns=['time_window', 'car_model', 'attack_type'])
    # Use the label encoder to get the true integer labels
    y_true = encoder.transform(features_df['attack_type'])
    
    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("\n--- Random Forest Report ---")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    # Use the encoder's classes for the target names in the report
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))


# --- Main execution block ---
if __name__ == '__main__':
    feature_file = 'features.csv'
    try:
        # Evaluate the first model
        evaluate_isolation_forest('isolation_forest_model.joblib', feature_file)
        
        # Evaluate the second model
        evaluate_random_forest('random_forest_model.joblib', feature_file, 'label_encoder.joblib')

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. {e}")
        print("Please ensure feature_engineering.py and train_model.py have been run successfully.")