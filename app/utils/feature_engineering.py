from exception_logging.logger import logging
from utils.data_loader import DataLoader
from utils.data_cleaner import DataCleaner
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self, data_cleaner: DataCleaner):
        self.data_cleaner = data_cleaner

    def engineer_features(self):
        try:
            X, y = self.data_cleaner.clean_data()
            logging.info("Feature engineering process started.")

            numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # Define transformations for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define transformations for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Combine transformations into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ]
            )

            logging.info("Feature engineering process completed successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error(f"An error occurred during feature engineering: {e}")
            raise