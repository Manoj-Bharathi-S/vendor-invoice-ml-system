import joblib
from pathlib import Path
import logging
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

# Log path Config
log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True)

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler(log_dir / "freight_cost_prediction.log")
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing vendor invoice training pipeline...")
    db_path = "data/inventory.db"
    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)
    logger.info("Output directory checked/created at: %s", model_dir)

    # Load data
    logger.info("Loading vendor invoice data from database: %s", db_path)
    df = load_vendor_invoice_data(db_path)
    logger.info("Data loaded successfully. Rows: %d, Columns: %d", df.shape[0], df.shape[1])

    # Prepare features and target
    logger.info("Extracting features and targets from dataset...")
    X, y = prepare_features(df)
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Split complete. Train rows: %d, Test rows: %d", X_train.shape[0], X_test.shape[0])

    # Train models
    logger.info("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    
    logger.info("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    
    logger.info("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    logger.info("Evaluating models on test data...")
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))

    # Select the best model (lowest MAE)
    logger.info("Comparing model evaluation metrics...")
    best_model = min(results, key=lambda x: x["mae"])
    best_model_name = best_model["model_name"]
    logger.info("Best model selected: %s (MAE: %s)", best_model_name, best_model["mae"])
    
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]
    # Save the best model
    model_path = model_dir / "predict_freight_model.pkl"
    logger.info("Saving best model artifact to: %s", model_path)
    joblib.dump(best_model,model_path)

    logger.info("Best model saved: %s", best_model_name)
    logger.info("Model path: %s", model_path)

if __name__ == "__main__":
    main()