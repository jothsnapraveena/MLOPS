import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import load_object

class ModelEvaluation:
    def __init__(self):
        logging.info("ModelEvaluation Started")
        # Set up basic MLflow tracking
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Evaluation metrics captured")
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            logging.info("Starting model evaluation")
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])
            
            # Load the model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            logging.info("Model loaded successfully")
            
            # Make predictions
            prediction = model.predict(X_test)
            rmse, mae, r2 = self.eval_metrics(y_test, prediction)
            
            # Start MLflow run
            with mlflow.start_run():
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log model type
                mlflow.log_param("model_type", type(model).__name__)
                
                logging.info(f"Successfully logged to MLflow")
            
            logging.info(f"Model evaluation completed - RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
            return rmse, mae, r2

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise customexception(e, sys)