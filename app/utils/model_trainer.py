from exception_logging.logger import logging
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from utils.feature_engineering import FeatureEngineer
from config import RANDOM_STATE, TEST_SIZE, CROSS_VAL_FOLDS, RANDOM_SEARCH_PARAMS

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
#from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

import numpy as np

from google.cloud import storage


class ModelTrainer:
    def __init__(self, X, y, feature_engineer: FeatureEngineer = None):
        self.X = X
        self.y = y

    def train_model(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        try:
            logging.info("Starting model training process.")
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            logging.info("Data split into training and testing sets.")

            #model = XGBRegressor()
            # Use hyperparameters if provided
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_STATE
            )

            pipeline = make_pipeline(model)
            logging.info("Pipeline created with DecisionTreeRegressor.")

            return X_train, X_test, y_train, y_test, pipeline

        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

    def evaluate_model(self, pipeline, X_test, y_test):
        """Evaluate model and return RMSE"""
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse
    
    
    def upload_model_to_gcs(self, bucket_name, source_file, destination_blob):
        try:
            logging.info(f"Uploading model to GCS bucket: {bucket_name}, from {source_file} to {destination_blob}")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob)
            blob.upload_from_filename(source_file)
            logging.info("Model uploaded to GCS successfully.")
        except Exception as e:
            logging.error(f"An error occurred while uploading the model to GCS: {e}")

