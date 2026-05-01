#Imports + Setup
import logging
import logging
import os
import sys
from venv import logger
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from src.utils.exception import CustomException

#Logger Initialization
logger = get_logger(__name__)

#Config Class
@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
            logger.info("DataIngestion initialized with config")
        except Exception as e:
            raise CustomException(e)

    #Create Directory Utility
    def _create_directories(self):
        try:
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)

            logger.info("Directories created successfully")
        except Exception as e:
            raise CustomException(e)
        
    #Load Data Method
    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError("Loaded dataframe is empty")

            logger.info(f"Data loaded successfully with shape {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e)
    
    #Save Data Method
    def save_data(self, df: pd.DataFrame, path: str):
        try:
            df.to_csv(path, index=False)
            logger.info(f"Data saved at {path}")
        except Exception as e:
            raise CustomException(e)
    
    #Train-Test Split
    def split_data(self, df: pd.DataFrame):
        try:
            logger.info("Splitting data into train and test")

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e)
        
    #Main Execution Method
    def initiate_data_ingestion(self, file_path: str):
        try:
            logger.info("Starting data ingestion process")

            # Step 1: Create directories
            self._create_directories()

            # Step 2: Load data
            df = self.load_data(file_path)
            le = LabelEncoder()
            df["Churn"] = le.fit_transform(df["Churn"])
            # Step 3: Save raw data
            self.save_data(df, self.config.raw_data_path)

            # Step 4: Split data
            train_df, test_df = self.split_data(df)

            # Step 5: Save splits
            self.save_data(train_df, self.config.train_data_path)
            self.save_data(test_df, self.config.test_data_path)

            logger.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error("Data ingestion failed")
            raise CustomException(e)