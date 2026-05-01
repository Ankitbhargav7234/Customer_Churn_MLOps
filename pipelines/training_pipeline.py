from src.ingestion.data_ingestion import DataIngestion, DataIngestionConfig
from src.validation.data_validation import DataValidation, DataValidationConfig
from src.eda.eda import EDA
from src.transformation.data_transformation import DataTransformation, DataTransformationConfig
from src.training.model_trainer import ModelTrainer, ModelTrainerConfig
from src.eda.eda_utils import EDAConfig, EDAArtifact
from tests.test_transformation import test_preprocessor_exists
from tests.test_ingestion import test_data_ingestion
from tests.test_validation import test_validation_report_exists
from src.evaluation.model_evaluation import ModelEvaluation, ModelEvaluationConfig

import os

def run_ingestion():
    config = DataIngestionConfig(
        raw_data_path="artifacts/raw.csv",
        train_data_path="artifacts/train.csv",
        test_data_path="artifacts/test.csv"
    )

    ingestion = DataIngestion(config)
    train_path, test_path = ingestion.initiate_data_ingestion("data/customer_churn.csv")

    return train_path, test_path



def run_validation(train_path, test_path):
    config = DataValidationConfig(
        schema_path="configs/schema.yaml",
        report_path="artifacts/validation_report.json"
    )

    validator = DataValidation(config)
    report = validator.initiate_data_validation(train_path, test_path)
    return report

def run_eda(train_path, test_path):
    config = EDAConfig(
        output_dir="artifacts/eda",
        target_column="Churn"
    )

    eda = EDA(config)
    eda_artifacts = eda.initiate_eda(train_path)
    return eda_artifacts

def run_transformation(train_path, test_path):
    config = DataTransformationConfig(
        preprocessor_path="artifacts/preprocessor.pkl",
        target_column="Churn"
    )

    transformer = DataTransformation(config)

    X_train, X_test, y_train, y_test, preprocessor_path = \
        transformer.initiate_data_transformation(train_path, test_path)

    return X_train, X_test, y_train, y_test

def run_training(X_train, y_train, X_test, y_test):
    config = ModelTrainerConfig(
        model_path="artifacts/model.pkl",
        mlflow_uri="http://localhost:5000"
    )

    trainer = ModelTrainer(config)

    model_path = trainer.initiate_model_training(
        X_train, y_train, X_test, y_test
    )

    return model_path

def run_evaluation(X_test, y_test):
    config = ModelEvaluationConfig(
        model_path="artifacts/model.pkl",
        report_path="artifacts/evaluation.json",
        mlflow_uri="http://localhost:5000",
        model_name="churn_model"
    )

    evaluator = ModelEvaluation(config)

    registered_model = evaluator.initiate_model_evaluation(X_test, y_test)

    return registered_model


if __name__ == "__main__":
    train_path, test_path = run_ingestion()
    test_data_ingestion()
    print("Data Ingestion and tests completed successfully.")

    report = run_validation(train_path, test_path)
    test_validation_report_exists()
    print("Data Validation and report generated successfully.")

    eda_artifacts = run_eda(train_path, test_path)
    print("EDA completed successfully. Plots saved in artifacts/eda.")

    X_train, X_test, y_train, y_test = run_transformation(train_path, test_path)
    test_preprocessor_exists()
    print("Data Transformation and preprocessor saved successfully.")

    model_path = run_training(X_train, y_train, X_test, y_test)
    print(f"Model training completed successfully. Model saved at {model_path}")

    registered_model = run_evaluation(X_test, y_test)
    print(f"Model evaluation completed successfully. Model registered in MLflow as {registered_model.name} version {registered_model.version}")