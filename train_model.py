from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/vedanthshenoy/Heart-Disease-UCI/master/heart-disease-data.csv"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def main():
    df = pd.read_csv(DATA_URL)
    df = df.replace("?", pd.NA)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["target"], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    y = (y > 0).astype(int)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ))
    ])
    pipeline.fit(X, y)

    output = Path("model/heart_model.joblib")
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    print(f"Model saved to {output}")

if __name__ == "__main__":
    main()
