import os
import sys
import json
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.common import read_yaml

logger = get_logger(__name__)


#Config Class
@dataclass
class DataValidationConfig:
    schema_path: str
    report_path: str
    drift_threshold: float = 0.05


class DataValidation:
    #Class Initialization
    def __init__(self, config: DataValidationConfig):
        try:
            self.config = config
            self.schema = read_yaml(config.schema_path)
            logger.info("DataValidation initialized")
        except Exception as e:
            raise CustomException(e)

    #Directory Setup    
    def _create_directories(self):
        try:
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
        except Exception as e:
            raise CustomException(e)
    
    #Schema Validation
    def validate_schema(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema["columns"]

            actual_columns = list(df.columns)

            missing_cols = [col for col in expected_columns if col not in actual_columns]
            extra_cols = [col for col in actual_columns if col not in expected_columns]

            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                return False

            if extra_cols:
                logger.warning(f"Extra columns: {extra_cols}")

            logger.info("Schema validation passed")
            return True

        except Exception as e:
            raise CustomException(e)
    
    #Missing Value Check
    def check_missing_values(self, df: pd.DataFrame) -> dict:
        try:
            missing_report = df.isnull().sum().to_dict()

            logger.info("Missing value check completed")
            return missing_report

        except Exception as e:
            raise CustomException(e)
        

    #Data Drift Detection
    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        try:
            drift_report = {}

            for col in base_df.columns:
                if base_df[col].dtype != "object" and base_df[col].dtype != "str":
                    base_mean = base_df[col].mean()
                    current_mean = current_df[col].mean()

                    drift = abs(base_mean - current_mean)

                    drift_report[col] = {
                        "base_mean": base_mean,
                        "current_mean": current_mean,
                        "drift": drift,
                        "is_drifted": 1 if drift > self.config.drift_threshold else 0
                    }

            logger.info("Data drift detection completed")
            return drift_report

        except Exception as e:
            raise CustomException(e)
    
    #Save Validation Report
    def save_report(self, report: dict):
        try:
            with open(self.config.report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Validation report saved at {self.config.report_path}")

        except Exception as e:
            raise CustomException(e)
        
    #Main Validation Method
    def initiate_data_validation(self, train_path: str, test_path: str):
        try:
            logger.info("Starting data validation")

            self._create_directories()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Schema validation
            is_valid_schema = self.validate_schema(train_df)

            # Missing values
            missing_train = self.check_missing_values(train_df)
            missing_test = self.check_missing_values(test_df)

            # Drift detection
            drift_report = self.detect_data_drift(train_df, test_df)

            validation_report = {
                "schema_valid": is_valid_schema,
                "missing_values_train": missing_train,
                "missing_values_test": missing_test,
                "drift_report": drift_report
            }

            self.save_report(validation_report)

            if not is_valid_schema:
                raise Exception("Schema validation failed")

            logger.info("Data validation completed successfully")

            return validation_report

        except Exception as e:
            logger.error("Data validation failed")
            raise CustomException(e)