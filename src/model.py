import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_PATH = "model_pipeline.joblib"
_pipeline = None

def _build_pipeline():
    numeric = ["budget_million","director_score","cast_popularity","runtime_minutes","marketing_spend","is_sequel"]
    categorical = ["genre","release_month"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = RandomForestClassifier(
    n_estimators=60,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)


    return Pipeline([("pre", pre), ("model", model)])

def train_model(df):
    global _pipeline

    X = df.drop("box_office_hit", axis=1)
    y = df["box_office_hit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _pipeline = _build_pipeline()
    _pipeline.fit(X_train, y_train)

    preds = _pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(_pipeline, MODEL_PATH)

    return {"accuracy": acc}

def load_model():
    global _pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")
    _pipeline = joblib.load(MODEL_PATH)

def predict_single(df):
    if _pipeline is None:
        load_model()
    preds = _pipeline.predict(df)
    score = _pipeline.predict_proba(df)[:,1]
    return {"prediction": preds, "score": score}

def is_model_available():
    return os.path.exists(MODEL_PATH)
