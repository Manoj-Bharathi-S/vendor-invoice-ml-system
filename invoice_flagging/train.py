import joblib
from pathlib import Path
import logging
import os

from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from model_evaluation import train_random_forest,evaluate_classifier

# Log path Config
log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True)

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler(log_dir / "invoice_flagging.log")
    ]
)

logger = logging.getLogger(__name__)


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    logger.info("Initializing invoice flagging pipeline...")
    db_path = "data/inventory.db"
    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)
    logger.info("Output directory checked/created at: %s", model_dir)

    # Load data
    logger.info("Loading invoice data from database: %s", db_path)
    df = load_invoice_data(db_path)
    logger.info("Data loaded successfully. Rows: %d, Columns: %d", df.shape[0], df.shape[1])
    
    logger.info("Applying labels to the dataset...")
    df = apply_labels(df)
    logger.info("Labels applied successfully.")
    

    # Prepare features and target
    logger.info("Splitting dataset into train and test sets (Target: %s)...", TARGET)
    X_train, X_test, y_train, y_test = split_data(df,FEATURES,TARGET)
    logger.info("Split complete. Train rows: %d, Test rows: %d", X_train.shape[0], X_test.shape[0])
    
    logger.info("Scaling features and saving scaler artifact...")
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, model_dir / "scaler.pkl"
    )
    logger.info("Feature scaling complete.")

    # Train models
    logger.info("Starting Random Forest hyperparameter tuning (GridSearch)...")
    grid_search = train_random_forest(X_train_scaled, y_train)
    logger.info("GridSearch complete. Best parameters found: %s", grid_search.best_params_)

    # Evaluate models
    logger.info("Evaluating the best estimator on test data...")
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save the best model
    model_path = model_dir / "predict_flag_invoice.pkl"
    logger.info("Saving model artifact to: %s", model_path)
    joblib.dump(grid_search,model_path)
    logger.info("Pipeline executed successfully!")

if __name__ == "__main__":
    main()