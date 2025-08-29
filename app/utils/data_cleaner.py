import os
import sys
from exception_logging.logger import logging
from utils.data_loader import DataLoader


class DataCleaner:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def clean_data(self):
        try:
            X, y = self.data_loader.load_data(file_path=self.data_loader.file_path)
            logging.info("Data cleaning process started.")

            # Identify numeric and categorical features
            numeric_columns = X.select_dtypes(include=['number']).columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns

            # Fill missing numeric values with median
            for col in numeric_columns:
                if X[col].isnull().sum() > 0:
                    median_value = X[col].median()
                    X[col].fillna(median_value, inplace=True)
                    logging.info(f"Filled missing values in numeric column '{col}' with median value {median_value}.")

            # Fill missing categorical values with mode
            for col in categorical_columns:
                if X[col].isnull().sum() > 0:
                    mode = X[col].mode()[0]
                    X[col].fillna(mode, inplace=True)
                    logging.info(f"Filled missing values in categorical column '{col}' with mode: {mode}")

            logging.info("Data cleaning process completed successfully.")
            return X, y
        except Exception as e:
            logging.error(f"An error occurred during data cleaning: {e}")
            raise