            
import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from src.utils.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def feature_engineering(self, df):
        """
        Extract month, year, and season from 'Date' column.
        """
        try:
            logging.info("Starting feature engineering...")

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'],format='mixed')  # Convert to datetime
                df['month'] = df['Date'].dt.month
                df['year'] = df['Date'].dt.year

                # Define function to extract season
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 'Winter'
                    elif month in [3, 4, 5]:
                        return 'Spring'
                    elif month in [6, 7, 8]:
                        return 'Summer'
                    else:
                        return 'Fall'

                df['season'] = df['month'].apply(get_season)

                # Drop the original Date column after extracting features
                df.drop(columns=['Date'], inplace=True)

            logging.info("Feature engineering completed successfully.")
            return df

        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise customexception(e, sys)

    def get_data_transformation(self):
        """
        Creates and returns a preprocessing pipeline.
        """
        try:
            logging.info('Starting data transformation pipeline setup...')

            # Define Features
            num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'month']
            cat_features = ['Store', 'season']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('binaryencoder', BinaryEncoder())
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, cat_features)
                ]
            )

            logging.info("Data transformation pipeline setup completed.")
            return preprocessor

        except Exception as e:
            logging.error("Exception occurred in data transformation setup")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_data_path, test_data_path):
        """
        Loads train and test data, applies feature engineering, then applies transformations.
        """
        try:
            logging.info("Reading train and test data files...")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head: \n{test_df.head().to_string()}")

            # Apply Feature Engineering
            logging.info("Applying feature engineering to train and test data...")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            print(train_df.columns)

            logging.info("Feature engineering applied successfully.")

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformation()

            # Define target column and drop unnecessary columns
            target_column_name = 'Weekly_Sales'
            drop_columns = [target_column_name,'year']

            # Prepare train and test features and target
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            print(input_feature_train_df.columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations to training and testing datasets...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Transformations applied successfully.")

            # Combine transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info("Saving preprocessor object...")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessing pickle file saved successfully at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Exception occurred in initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)
