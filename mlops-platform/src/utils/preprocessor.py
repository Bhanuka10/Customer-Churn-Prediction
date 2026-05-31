import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
CAT_COLS    = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaymentMethod']

def preprocess_input(data: dict) -> pd.DataFrame:
    """Transform a single prediction request into model-ready features."""
    df = pd.DataFrame([data])
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'Churn' in df.columns:
        df.drop('Churn', axis=1, inplace=True)

    if 'TotalCharges' in df.columns:
        total_charges = pd.to_numeric(df['TotalCharges'], errors='coerce')
    else:
        total_charges = pd.Series([0] * len(df))
    df['TotalCharges'] = total_charges.fillna(0)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (df[col].isin(['Male', 'Yes', 1, True])).astype(int)

    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def load_scaler(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)