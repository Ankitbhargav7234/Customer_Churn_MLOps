import os
import sys
import pickle
import mlflow
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from src.utils.logger import get_logger
from src.utils.exception import CustomException


logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    model_path: str
    mlflow_uri: str

class ModelTrainer:
    #Class Initialization
    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config = config
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            logger.info("ModelTrainer initialized with MLflow")
        except Exception as e:
            raise CustomException(e)
    
    # Model Definer
    def get_models(self):
        try:
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier()
            }
            return models
        except Exception as e:
            raise CustomException(e)
    
    #Define Hyperparameters
    def get_params(self):
        return {
            "logistic_regression": {
                "C": [0.1, 1, 10]
            },
            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10]
            }
        }

    #Model Evaluation
    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            return {
                "accuracy": accuracy,
                "roc_auc": roc_auc
            }

        except Exception as e:
            raise CustomException(e)
    
    #Training + MLflow Logging
    def train_and_log(self, model_name, model, params, X_train, y_train, X_test, y_test):
        try:
            best_score = -np.inf
            best_model = None

            for param_name, values in params.items():
                for val in values:
                    with mlflow.start_run(run_name=f"{model_name}_{param_name}_{val}"):

                        # Set parameter
                        model.set_params(**{param_name: val})

                        # Train
                        model.fit(X_train, y_train)

                        # Evaluate
                        metrics = self.evaluate_model(model, X_test, y_test)

                        # Log to MLflow
                        mlflow.log_param(param_name, val)
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(model, model_name)

                        logger.info(f"{model_name} | {param_name}={val} | ROC-AUC={metrics['roc_auc']}")

                        # Track best model
                        if metrics["roc_auc"] > best_score:
                            best_score = metrics["roc_auc"]
                            best_model = model

            return best_model, best_score

        except Exception as e:
            raise CustomException(e)

    #Model Selection
    def select_best_model(self, X_train, y_train, X_test, y_test):
        try:
            models = self.get_models()
            params = self.get_params()

            best_model = None
            best_score = -np.inf
            best_model_name = None

            for model_name, model in models.items():
                logger.info(f"Training model: {model_name}")

                model_params = params.get(model_name, {})

                trained_model, score = self.train_and_log(
                    model_name, model, model_params,
                    X_train, y_train, X_test, y_test
                )

                if score > best_score:
                    best_score = score
                    best_model = trained_model
                    best_model_name = model_name

            logger.info(f"Best model: {best_model_name} with score {best_score}")

            return best_model

        except Exception as e:
            raise CustomException(e)
        
    #Save Best Model
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            pickle.dump(model, open(self.config.model_path, "wb"))

            logger.info(f"Model saved at {self.config.model_path}")

        except Exception as e:
            raise CustomException(e)

    #Initiate Model Training
    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model training")

            best_model = self.select_best_model(
                X_train, y_train, X_test, y_test
            )

            self.save_model(best_model)

            logger.info("Model training completed")

            return self.config.model_path

        except Exception as e:
            logger.error("Model training failed")
            raise CustomException(e)