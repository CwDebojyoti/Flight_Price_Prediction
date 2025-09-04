from exception_logging.logger import logging
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from utils.feature_engineering import FeatureEngineer
from config import RANDOM_STATE, TEST_SIZE, CROSS_VAL_FOLDS, RANDOM_SEARCH_PARAMS

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
#from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from google.cloud import storage


class ModelTrainer:
    def __init__(self, X, y, feature_engineer: FeatureEngineer = None):
        self.X = X
        self.y = y

    def train_model(self):
        try:
            logging.info("Starting model training process.")
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            logging.info("Data split into training and testing sets.")

            #model = XGBRegressor()
            model = DecisionTreeRegressor()

            logging.info("Initialized DecisionTree model trainer with RandomSearch tuner.")
            randomsearch_model_tune = RandomizedSearchCV(estimator=model,
                                                          param_distributions=RANDOM_SEARCH_PARAMS['DecisionTree'],
                                                          n_iter=100,
                                                          cv=CROSS_VAL_FOLDS,
                                                          scoring='neg_mean_squared_error',
                                                          verbose=2,
                                                          random_state=RANDOM_STATE,
                                                          n_jobs=-1)
            
            logging.info("Configured RandomizedSearchCV for hyperparameter tuning.")

            logging.info("Model trainer setup complete. Ready to fit the model.")

            return X_train, X_test, y_train, y_test, randomsearch_model_tune


        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

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

