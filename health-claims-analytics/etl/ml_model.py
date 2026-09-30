import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(df: pd.DataFrame):
    # Required features
    features = ['age', 'bmi', 'smoker', 'income', 'chronic_count']

    # Drop rows with missing columns
    df = df.dropna(subset=features + ['annual_medical_cost'])

    # Map smoker strings to 0 or 1; default 0 if unknown
    smoker_map = {
        'yes': 1, 'Yes': 1, 'y': 1, 'Y': 1,
        'no': 0, 'No': 0, 'n': 0, 'N': 0,
        'never': 0, 'Never': 0,
        'occasionally': 1, 'Occasionally': 1,
    }
    df['smoker'] = df['smoker'].astype(str).map(smoker_map).fillna(0).astype(int)

    # Extract features and target
    X = df[features]
    y = df['annual_medical_cost']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(df: pd.DataFrame):
    X, y, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model MAE: {mae:.2f}")
    print(f"Model R^2: {r2:.3f}")

    # Save model and scaler
    joblib.dump(model, 'etl/medical_cost_model.joblib')
    joblib.dump(scaler, 'etl/scaler.joblib')
    print("Model and scaler saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ml_model.py <path_to_csv>")
        exit(1)
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    train_model(df)
