GCS_BUCKET_NAME = 'flight_price_data'
DATA_FILE_NAME = 'airlines_flights_data.csv'

TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL_FOLDS = 5

PROJECT_ID = "your-gcp-project-id"
MODEL_DIR = "models/flight_price_model"  # Folder in GCS where model is saved
PROCESSOR_DIR = "preprocessor"  # Folder in GCS where preprocessor is saved
MODEL_FILENAME = "model.pkl"

RANDOM_SEARCH_PARAMS = {
    'XGBRegressor': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}