import joblib
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Instantiate the module-level logger
logger = logging.getLogger(__name__)


def load_invoice_data(db_path:str):
    """
    Load invoice data from the SQLite database.
    """
    logger.info("Connecting to database at: %s", db_path)
    connection = sqlite3.connect(db_path)
    query = """
            WITH purchase_agg AS(
            SELECT
                p.PONumber,
                COUNT(DISTINCT p.Brand) as total_brands,
                SUM(p.Quantity) as total_item_quantity,
                SUM(p.Dollars) as total_item_dollars,
                AVG(julianday(ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
            FROM purchases p
            GROUP BY p.PONumber
            )
                                
            SELECT
            vi.Quantity as invoice_quantity,
            vi.Dollars as invoice_dollars,
            vi.Freight,
            (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
            (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay

            FROM vendor_invoice as vi
            LEFT JOIN purchase_agg pa
                ON vi.PONumber = pa.PONumber
            """
    logger.info("Executing analytical SQL query...")
    df = pd.read_sql_query(query, connection)
    connection.close()
    logger.info("Successfully fetched %d records from the database.", len(df))
    return df

def create_invoice_risk_label(row):

    # Invoice total mismatch with item-level total
    if (abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5):
        return 1
    
    # Abnormally high receiving delay
    if row["avg_receiving_delay"] > 10:
        return 2
    return 0

def apply_labels(df):
    logger.info("Applying multi-class invoice risk labelling criteria...")
    df["flag_invoice"] = df.apply(create_invoice_risk_label,axis=1)
    
    # Log the distribution of generated classes for visibility
    class_counts = df["flag_invoice"].value_counts().to_dict()
    logger.info("Label distribution generated: %s", class_counts)
    return df

def split_data(df,features,target,test_size=0.2,random_state=42):
    """
    Split data into training and testing sets
    """
    logger.info("Extracting feature columns and target mapping...")
    X = df[features]
    y = df[target]
    
    logger.info("Performing train_test_split (test_size=%s)...", test_size)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train,X_test,model_path):
    """
    Scale features and target variable
    """
    logger.info("Initializing StandardScaler fitting on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # CRITICAL BUG FIX note: Changed fit_transform here to transform so you don't leak test data stats!
    logger.info("Transforming testing features using fitted training metrics...")
    X_test_scaled = scaler.transform(X_test)

    logger.info("Saving fitted scaler artifact to: %s", model_path)
    joblib.dump(scaler,model_path)

    return X_train_scaled, X_test_scaled