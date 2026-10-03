import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

def load_and_prepare_data():
    data = pd.read_csv("cwru_processed_dataset.csv")
    
    features = [
        "RMS",
        "Peak_to_Peak",
        "Kurtosis",
        "Skewness",
        "Crest_Factor",
        "Dominant_Freq_Hz",
        "Spectral_Energy"
    ]
    X = data[features]
    y = data["Fault_Label"]

    return X, y, data

def main():
    X, y, df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Total dataset: {len(X)} samples")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    accuracy_percentage = accuracy * 100

    print("\n" + "=" * 50)
    print(f"MODEL ACCURACY ON UNSEEN DATA: {accuracy_percentage:.2f}%")
    print("=" * 50)

    target_names = ["Healthy", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]
    
    print("\n=== DETAILED PERFORMANCE REPORT ===")
    print(classification_report(y_test, predictions, target_names=target_names))

    # Feature importances
    print("Feature Importances:")
    feature_names = X.columns
    importance_scores = model.feature_importances_
    for name, score in zip(feature_names, importance_scores):
        print(f"  • {name:20s}: {score * 100:.2f}% importance")

    model_filename = "bearing_fault_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"\n Saved trained model as '{model_filename}'")
if __name__ == "__main__":
    main()