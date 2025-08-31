import joblib
import os
from config import GCS_BUCKET_NAME, DATA_FILE_NAME, MODEL_DIR, MODEL_FILENAME, PROCESSOR_DIR
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from exception_logging.logger import logging
from utils.feature_engineering import FeatureEngineer
from utils.feature_engineering import FeatureEngineer
from utils.model_trainer import ModelTrainer
from sklearn.pipeline import Pipeline


def main():

    data_loader = DataLoader(file_path=f"gs://{GCS_BUCKET_NAME}/data/{DATA_FILE_NAME}")

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    model_tuner = ModelTrainer(X=X, y=y)

    X_train, X_test, y_train, y_test, best_model = model_tuner.train_model()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # train_model = model.fit(X_train_transformed, y_train)

    tuned_model = best_model.fit(X_train_transformed, y_train).best_estimator_

    # 4. Build full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", tuned_model)
    ])
    pipeline.fit(X, y)

    # 5. Save model locally
    os.makedirs("artifacts", exist_ok=True)
    local_model_path = os.path.join("artifacts", MODEL_FILENAME)
    joblib.dump(pipeline, local_model_path)
    print(f"✅ Model saved locally at {local_model_path}")

    # 6. Upload to GCS
    gcs_path = f"{MODEL_DIR}/{MODEL_FILENAME}"
    model_tuner.upload_model_to_gcs(local_model_path, GCS_BUCKET_NAME, gcs_path)








if __name__ == "__main__":
    main()
