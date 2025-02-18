import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object,evaluate_model

from sklearn.linear_model import LinearRegression , Ridge 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor





@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependeny and Independent variables")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'XGBRegressor':XGBRegressor()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)

            logging.info(f'model_report:{model_report}')

            best_model_score=max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            print(f"✅ Model saved at: {self.model_trainer_config.trained_model_file_path}")

            logging.info("Model training completed successfully.") 


        
        except Exception as e:
            logging.info("Model training failed")
            raise customexception(e,sys)

