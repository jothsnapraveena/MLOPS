import os
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

try:
    print("✅ Starting Training Pipeline...")

    # ✅ Step 1: Data Ingestion
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initialize_data_ingestion()
    print(f"✅ Data Ingestion Complete: Train → {train_data_path}, Test → {test_data_path}")

    # ✅ Step 2: Data Transformation
    data_transformation = DataTransformation()
   
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

    # ✅ Step 3: Model Training
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initate_model_training(train_arr, test_arr)
    print("✅ Model Training Complete!")

    # ✅ Step 4: Model Evaluation
    model_eval_obj = ModelEvaluation()
    model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
    print("✅ Model Evaluation Complete!")

except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    logging.error(f"Training Pipeline Failed: {str(e)}")
    raise customexception(e, sys)
