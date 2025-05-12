import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path: str):
    """Load raw CSV and return X_train, X_test, y_train, y_test (scaled)."""
    df = pd.read_csv(csv_path)

    # ---------- Feature engineering ----------
    df = df.drop(columns=["months_employed"])
    df["personal_account_months"] = (
        df["personal_account_m"] + 12 * df["personal_account_y"]
    )
    df = df.drop(columns=["personal_account_m", "personal_account_y"])

    # ---------- One‑hot encoding ----------
    df = pd.get_dummies(df, drop_first=True)
    y = df["e_signed"].copy()
    X = df.drop(columns=["e_signed", "entry_id"])

    # ---------- Train / test split (stratified) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=100
    )

    # ---------- Scaling ----------
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train, X_test, y_train, y_test, scaler
