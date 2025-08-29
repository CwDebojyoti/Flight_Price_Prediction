import pandas as pd
import numpy as np
from exception_logging.logger import logging

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.
        
        Args:
            file_path (str): The path to the CSV file.
        
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        try:
            logging.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully with shape {data.shape}")
            try:
                data = data.drop(columns=['index', 'flight'])
            except KeyError:
                logging.warning("'flight' column not found in data, skipping drop operation.")
            try:
                X = data.drop('price', axis=1)
                y = data['price']
            except KeyError as e:
                logging.error(f"'price' column not found in data: {e}")
                raise
            return X, y
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise