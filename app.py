from fastapi import FastAPI
from datetime import datetime
from cleaned_data import cleaning_data
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

rf_model = joblib.load("rfmodel.pkl")
features = joblib.load("feature_columns.pkl")

app = FastAPI()

class InputData(BaseModel):
    donation_date: datetime
    payment_method: str
    city: str
    count_lifetime_donations: int
    count_donations_by_year: int
    campaign_indicator: int
    major_donor_flag: int

@app.post("/predict")
def predict(data: InputData):
    X = pd.DataFrame([data.model_dump()])
    X["donor_id"] = 0
    X["donation_id"] = 0
    X = cleaning_data(X, production = True)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    y_pred = rf_model.predict(X)[0]

    #building confidence intervals
    all_tree_preds = np.array([tree.predict(X)[0] for tree in rf_model.estimators_])
    lower = np.percentile(all_tree_preds, 2.5)
    upper = np.percentile(all_tree_preds, 97.5)

    return {
        "predicted_donation": int(round(y_pred, 2)),
        "95%_interval": [float(round(lower,2)), float(round(upper, 2))]
    }