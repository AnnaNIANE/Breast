# model.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def create_model():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Get feature names for later use in the web app
    feature_names = data.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model and scaler
    joblib.dump(model, 'models/breast_cancer_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("Model and scaler saved successfully!")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    create_model()