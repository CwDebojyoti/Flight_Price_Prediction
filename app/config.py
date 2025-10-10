import os

GCS_BUCKET_NAME = 'flight_price_data'
DATA_FILE_NAME = 'airlines_flights_data.csv'

TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL_FOLDS = 5

PROJECT_ID = "your-gcp-project-id"
MODEL_DIR = "models"  # Base folder in GCS where models are saved (trial subdirs will be created)
PROCESSOR_DIR = "preprocessor"  # Folder in GCS where preprocessor is saved
MODEL_FILENAME = "model.joblib"
PROCESSOR_FILENAME = "preprocessor.joblib"

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "flight-price-prediction-470515")
REGION = "us-central1"
SECRET_NAME = "flight-price-endpoint-id"

RANDOM_SEARCH_PARAMS = {
    'XGBRegressor': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
}