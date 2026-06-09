import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path:str):
    """
    Load vendor invoice data from the SQLite database.
    """
    connection = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, connection)
    connection.close()
    return df

def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable
    """
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y

def split_data(X:int,y:int,test_size=0.2,random_state=42):
    """
    Split data into training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)