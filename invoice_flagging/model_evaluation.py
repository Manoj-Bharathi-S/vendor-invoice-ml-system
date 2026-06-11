from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,classification_report,make_scorer,f1_score
import logging

logger = logging.getLogger(__name__)

def train_random_forest(X_train,y_train):
    rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1
    )
    
    param_grid = {
        "n_estimators":[100,200,300],
        "max_depth":[None,4,5,6],
        "min_samples_split":[2,3,5],
        "min_samples_leaf":[1,2,5],
        "criterion":['gini','entropy']
    }

    logger.info("Setting up Multi-class F1 Macro scorer for parameter evaluation...")
    scorer = make_scorer(f1_score, average='macro')
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    logger.info("Launching GridSearchCV over %d parameter combinations with 5-fold CV...", 
                len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"]) * len(param_grid["min_samples_leaf"]) * len(param_grid["criterion"]))
    
    grid_search.fit(X_train,y_train)

    logger.info("GridSearchCV optimization completed successfully.")
    logger.info("Best parameters identified: %s", grid_search.best_params_)
    logger.info("Best Cross-Validation F1-Macro score: %.4f", grid_search.best_score_)

    return grid_search


def evaluate_classifier(model,X_test,y_test,model_name):
    logger.info("Running out-of-sample inference using model: %s", model_name)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))
    
    logger.info("Model evaluation performance summary printed to stdout.")