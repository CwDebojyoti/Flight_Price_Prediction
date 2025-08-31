import joblib
import os
from config import GCS_BUCKET_NAME, DATA_FILE_NAME
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from exception_logging.logger import logging
from utils.feature_engineering import FeatureEngineer
from utils.feature_engineering import FeatureEngineer
from utils.model_trainer import ModelTrainer
from config import RANDOM_STATE, TEST_SIZE, CROSS_VAL_FOLDS, RANDOM_SEARCH_PARAMS

def main():

    data_loader = DataLoader(file_path=f"gs://{GCS_BUCKET_NAME}/data/{DATA_FILE_NAME}")

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    model_tuner = ModelTrainer(X=X, y=y)

    X_train, X_test, y_train, y_test, model = model_tuner.train_model()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    train_model = model.fit(X_train_transformed, y_train)

    tuned_model = train_model.best_estimator_

    try:
        os.makedirs('preprocessor', exist_ok=True)
        joblib.dump(preprocessor, "preprocessor/preprocessor.pkl")
    except:
        logging.error("Error saving the preprocessor locally.")

    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(tuned_model, "models/flight_price_model.pkl")
    except:
        logging.error("Error saving the model locally.")

    model_tuner.upload_model_to_gcs(bucket_name=GCS_BUCKET_NAME,
                                    source_file='models/flight_price_model.pkl',
                                    destination_blob='models/flight_price_model.pkl')








if __name__ == "__main__":
    main()
