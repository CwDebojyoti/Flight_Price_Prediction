from config import GCS_BUCKET_NAME, DATA_FILE_NAME
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from utils.feature_engineering import FeatureEngineer


def main():

    data_loader = DataLoader(file_path=f"gs://{GCS_BUCKET_NAME}/data/{DATA_FILE_NAME}")

    data_cleaner = DataCleaner(data_loader=data_loader)
    X, y = data_cleaner.clean_data()

    preprocessor = FeatureEngineer(data_cleaner=data_cleaner).engineer_features()

    X_transformed = preprocessor.fit_transform(X)



    print(X_transformed.shape)




if __name__ == "__main__":
    main()