import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from config import GCS_BUCKET_NAME, DATA_FILE_NAME, MODEL_DIR, MODEL_FILENAME, PROCESSOR_DIR, PROCESSOR_FILENAME
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from exception_logging.logger import logging
from utils.feature_engineering import FeatureEngineer
from utils.feature_engineering import FeatureEngineer
from utils.model_trainer import ModelTrainer
from sklearn.pipeline import Pipeline

def save_model_artifacts(pipeline, preprocessor):
    os.makedirs("artifacts", exist_ok=True)
    local_model_path = os.path.join("artifacts", f"{MODEL_FILENAME}")
    local_preprocessor_path = os.path.join("artifacts", f"{ PROCESSOR_FILENAME}")
    joblib.dump(pipeline, local_model_path)
    joblib.dump(preprocessor, local_preprocessor_path)
    print(f"✅ Model saved locally at {local_model_path}")
    print(f"✅ Preprocessor saved at {local_preprocessor_path}")
    return local_model_path, local_preprocessor_path


def main():

    data_loader = DataLoader(file_path=f"gs://{GCS_BUCKET_NAME}/data/{DATA_FILE_NAME}")

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    model_tuner = ModelTrainer(X=X, y=y)

    X_train, X_test, y_train, y_test, pipeline = model_tuner.train_model()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    pipeline = pipeline.fit(X_train_transformed, y_train)

    local_model_path, local_preprocessor_path = save_model_artifacts(pipeline, preprocessor)


    """
    # 4. Build full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", tuned_model)
    ])
    pipeline.fit(X, y)
    """


    # 6. Upload to GCS
    gcs_model_path = f"{MODEL_DIR}/{MODEL_FILENAME}"
    gcs_preprocessor_path = f"{PROCESSOR_DIR}/{PROCESSOR_FILENAME}"
    model_tuner.upload_model_to_gcs(GCS_BUCKET_NAME, local_model_path, gcs_model_path)
    model_tuner.upload_model_to_gcs(GCS_BUCKET_NAME, local_preprocessor_path, gcs_preprocessor_path)

    y_pred = pipeline.predict(X_test_transformed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)








if __name__ == "__main__":
    main()
