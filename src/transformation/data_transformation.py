import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

#Config Class
@dataclass
class DataTransformationConfig:
    preprocessor_path: str
    target_column: int

#Data Transformation Class
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        try:
            self.config = config
            logger.info("DataTransformation initialized")
        except Exception as e:
            raise CustomException(e)
    
    # Preprocessor Pipeline Creation
    def get_preprocessor(self, df: pd.DataFrame):
        try:
            logger.info("Creating preprocessing pipeline")

            target_col = self.config.target_column

            numerical_cols = df.select_dtypes(exclude="object").columns.tolist()
            categorical_cols = df.select_dtypes(include="object").columns.tolist()

            # Remove target column
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols)
            ])

            logger.info("Preprocessing pipeline created successfully")

            return preprocessor, numerical_cols, categorical_cols

        except Exception as e:
            raise CustomException(e)

    # Fit and Transform Training Data  
    def fit_transform(self, train_df: pd.DataFrame, preprocessor):
        try:
            logger.info("Fitting and transforming training data")

            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]

            X_train_transformed = preprocessor.fit_transform(X_train)

            logger.info("Training data transformed")

            return X_train_transformed, y_train, preprocessor

        except Exception as e:
            raise CustomException(e)
    
    # Transform Test Data
    def transform(self, test_df: pd.DataFrame, preprocessor):
        try:
            logger.info("Transforming test data")

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            X_test_transformed = preprocessor.transform(X_test)

            logger.info("Test data transformed")

            return X_test_transformed, y_test

        except Exception as e:
            raise CustomException(e)
    
    # Save Preprocessor Object
    def save_preprocessor(self, preprocessor):
        try:
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            pickle.dump(preprocessor, open(self.config.preprocessor_path, "wb"))

            logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")

        except Exception as e:
            raise CustomException(e)
    
    #Main Transformation Method
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logger.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Step 1: Build preprocessor
            preprocessor, num_cols, cat_cols = self.get_preprocessor(train_df)

            # Step 2: Fit on train
            X_train, y_train, preprocessor = self.fit_transform(train_df, preprocessor)

            # Step 3: Transform test
            X_test, y_test = self.transform(test_df, preprocessor)

            # Step 4: Save preprocessor
            self.save_preprocessor(preprocessor)

            logger.info("Data transformation completed")

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.config.preprocessor_path
            )

        except Exception as e:
            logger.error("Data transformation failed")
            raise CustomException(e)        