import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data():
    """Load dataset and separate feature matrix and target labels."""
    data = pd.read_csv("cwru_processed_dataset.csv")
    feature_cols = [
        "RMS", 
        "Peak_to_Peak", 
        "Kurtosis", 
        "Skewness", 
        "Crest_Factor", 
        "Dominant_Freq_Hz", 
        "Spectral_Energy"
    ]
    
    X = data[feature_cols]
    y = data["Fault_Label"]
    return X, y

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    f1 = f1_score(y_test, predictions, average="weighted")

    return {
        "Name": model_name,
        "Accuracy": accuracy,
        "F1_Score": f1,
        "Train_Time_Sec": training_time,
        "Fitted_Model": model
    }

def main():
    print("=" * 60)
    print("BENCHMARK: Random Forest vs. XGBoost on Real CWRU Data")
    print("=" * 60)
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="mlogloss")

    print("Training Random Forest...")
    rf_results = train_and_evaluate(rf_model, "Random Forest", X_train, X_test, y_train, y_test)

    print("Training XGBoost...")
    xgb_results = train_and_evaluate(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print(f"{'Model Name':<18} {'Accuracy':<12} {'F1-Score':<12} {'Training Time':<15}")
    print("-" * 60)
    print(f"{rf_results['Name']:<18} {rf_results['Accuracy']:.2f}%{'':<5} {rf_results['F1_Score']:.4f}{'':<6} {rf_results['Train_Time_Sec']:.4f} sec")
    print(f"{xgb_results['Name']:<18} {xgb_results['Accuracy']:.2f}%{'':<5} {xgb_results['F1_Score']:.4f}{'':<6} {xgb_results['Train_Time_Sec']:.4f} sec")
    print("=" * 60)

    # Save best performing model for the Streamlit dashboard
    with open("best_bearing_model.pkl", "wb") as f:
        pickle.dump(xgb_results["Fitted_Model"], f)
    print("\n Saved top-performing model as 'best_bearing_model.pkl'")
if __name__ == "__main__":
    main()
