import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/predict_flag_invoice.pkl"

def load_model(model_path):
    """
    Load trained freight cost prediction model
    """
    with open(model_path,"rb") as f:
        model = joblib.load(f)
    return model

def scaler(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def predict_invoice_flag(data):
    """
    Predict invoice flag for new vendor invoices
    
    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model(MODEL_PATH)
    input_df = pd.DataFrame(data,index=[0])
    scaled_array = scaler(input_df)
    predictions = model.predict(scaled_array).round()
    input_df["predicted_flag"] = predictions
    return input_df


if __name__ == "__main__":

    # Example inference run
    sample_data = {
       "invoice_quantity":48,
        "invoice_dollars":352.95,
        "Freight":1.73,
        "total_item_quantity":162,
        "total_item_dollars":2476.99
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)