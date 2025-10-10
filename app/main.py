import joblib
import os
import sys
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
import argparse
import hypertune


print("Command-line arguments:", sys.argv)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', dest='max_depth', default=None, type=int, help='Max depth for DecisionTreeRegressor')
    parser.add_argument('--min_samples_split', dest='min_samples_split', default=2, type=int, help='Min samples split')
    parser.add_argument('--min_samples_leaf', dest='min_samples_leaf', default=1, type=int, help='Min samples leaf')
    return parser.parse_args()



def save_model_artifacts(pipeline, preprocessor, rmse=None):
    os.makedirs("artifacts", exist_ok=True)

    # Include RMSE in filename if provided
    if rmse is not None:
        model_filename = f"model_rmse_{rmse:.2f}.joblib"
        preprocessor_filename = f"preprocessor_rmse_{rmse:.2f}.joblib"
    else:
        model_filename = MODEL_FILENAME
        preprocessor_filename = PROCESSOR_FILENAME


    local_model_path = os.path.join("artifacts", f"{MODEL_FILENAME}")
    local_preprocessor_path = os.path.join("artifacts", f"{ PROCESSOR_FILENAME}")
    joblib.dump(pipeline, local_model_path)
    joblib.dump(preprocessor, local_preprocessor_path)
    print(f"✅ Model saved locally at {local_model_path}")
    print(f"✅ Preprocessor saved at {local_preprocessor_path}")
    return local_model_path, local_preprocessor_path, model_filename, preprocessor_filename


def main():

    args = parse_args()

    print(f"Training with hyperparameters: max_depth={args.max_depth}, "
          f"min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf}")

    data_loader = DataLoader(file_path=f"gs://{GCS_BUCKET_NAME}/data/{DATA_FILE_NAME}")

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    model_tuner = ModelTrainer(X=X, y=y)

    X_train, X_test, y_train, y_test, pipeline = model_tuner.train_model(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    pipeline = pipeline.fit(X_train_transformed, y_train)

    # Evaluate model
    rmse = model_tuner.evaluate_model(pipeline, X_test_transformed, y_test)
    print(f'RMSE: {rmse}')

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='rmse',
        metric_value=rmse,
        global_step=1
    )

    # Save model artifacts with RMSE in filename
    local_model_path, local_preprocessor_path, model_filename, preprocessor_filename = save_model_artifacts(
        pipeline, preprocessor, rmse
    )


    """
    # 4. Build full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", tuned_model)
    ])
    pipeline.fit(X, y)
    """

    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "0")  # default 0 if not tuning

    # 6. Upload to GCS
    gcs_model_path = f"{MODEL_DIR}/trial_{trial_id}/{MODEL_FILENAME}"
    gcs_preprocessor_path = f"{PROCESSOR_DIR}/{PROCESSOR_FILENAME}"
    model_tuner.upload_model_to_gcs(GCS_BUCKET_NAME, local_model_path, gcs_model_path)
    model_tuner.upload_model_to_gcs(GCS_BUCKET_NAME, local_preprocessor_path, gcs_preprocessor_path)

    print(f"✅ Training completed. RMSE: {rmse}")
    
    

    






if __name__ == "__main__":
    main()
