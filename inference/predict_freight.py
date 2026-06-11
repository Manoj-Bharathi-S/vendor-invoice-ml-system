import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path):
    """
    Load trained freight cost prediction model
    """
    with open(model_path,"rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(data):
    """
    Predict freight cost for new vendor invoices
    
    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted freight costs
    """
    model = load_model(MODEL_PATH)
    input_df = pd.DataFrame(data)
    input_df["predicted_freight_cost"] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":

    # Example inference run
    sample_data = {
        "Dollars":[18500,9000,3000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
    
