import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse_score
from sklearn.model_selection import train_test_split

from .model import build_model
from .preprocessing import TARGET

ROOT = Path(__file__).resolve().parents[2]    
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "data" / "train-house-prices-advanced-regression-techniques.csv"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns= TARGET)
    y = df[TARGET]

    return X, y

def evaluate (model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score( y, y_pred)
    rmse = rmse_score(y, y_pred)

    return {"r2": r2, "rmse":rmse}

def main():
    X,y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
    model = build_model().fit(X_train, y_train)
    #evaluate
    train_metrics = evaluate(model, X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)
    #Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print()
    print("========== Train Metrics ==========")
    print(f"R2   : {train_metrics['r2']:.4f}")
    print(f"RMSE : {train_metrics['rmse']:.4f}")

    print("\n========== Test Metrics ==========")
    print(f"R2   : {test_metrics['r2']:.4f}")
    print(f"RMSE : {test_metrics['rmse']:.4f}")





if __name__ == "__main__":
    main()