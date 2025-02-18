import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object


class PredictPipeline:

    def __init__(self):
        print("initialising the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            print("*** Loading Preprocessor and Model ***")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            print("*** Checking Input DataFrame Before preprocessing ***")
            print(features.head())

            scaled_features=preprocessor.transform(features)
            print("*** Features Transformed Successfully ***")
            pred=model.predict(scaled_features)
            print(f" Prediction Output :{pred}")
            return pred

        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,
                 Store: str,  # ✅ Change to lowercase
                 Holiday_Flag: int,
                 Temperature: float,
                 Fuel_Price: float,
                 CPI: float,
                 Unemployment: float,
                 month: int,
                 season: str):

        # ✅ Keep all instance variables lowercase
        self.Store = Store
        self.Holiday_Flag = Holiday_Flag
        self.Temperature = Temperature
        self.Fuel_Price = Fuel_Price
        self.CPI = CPI
        self.Unemployment = Unemployment
        self.month = month
        self.season = season

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Store': [self.Store],
                'Holiday_Flag': [self.Holiday_Flag],
                'Temperature': [self.Temperature],
                'Fuel_Price': [self.Fuel_Price],
                'CPI': [self.CPI],
                'Unemployment': [self.Unemployment],
                'month': [self.month],
                'season': [self.season]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise customexception(e,sys)