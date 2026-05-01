import os
import sys
import json
import mlflow
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.common import load_object

logger = get_logger(__name__)

@dataclass
class ModelEvaluationConfig:
    model_path: str
    report_path: str
    mlflow_uri: str
    model_name: str
    threshold_roc_auc: float = 0.7

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        try:
            self.config = config
            mlflow.set_tracking_uri(config.mlflow_uri)
            logger.info("ModelEvaluation initialized")
        except Exception as e:
            raise CustomException(e)
    
    def load_model(self):
        try:
            model = load_object(self.config.model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            raise CustomException(e)
    
    def evaluate(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }

            logger.info(f"Evaluation metrics for model {self.config.model_name}: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e)

    def save_report(self, report: dict):
        try:
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)

            with open(self.config.report_path, "w") as f:
                json.dump(report, f, indent=4)

            logger.info(f"Evaluation report saved at {self.config.report_path}")

        except Exception as e:
            raise CustomException(e)
    
    def is_model_acceptable(self, metrics: dict) -> bool:
        try:
            roc_auc = metrics["roc_auc"]

            if roc_auc >= self.config.threshold_roc_auc:
                logger.info("Model passed acceptance criteria")
                return True
            else:
                logger.warning("Model failed acceptance criteria")
                return False

        except Exception as e:
            raise CustomException(e)


    # Register model to MLflow
    def register_model(self, model, metrics):
        try:
            logger.info("Registering model to MLflow")

            with mlflow.start_run():
                mlflow.sklearn.log_model(model, "model")

                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("f1_score", metrics["f1_score"])
                mlflow.log_metric("roc_auc", metrics["roc_auc"])
                #mlflow.log_metric("confusion_matrix", metrics["confusion_matrix"])
                mlflow.sklearn.log_model(model, "model")
                
                model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)

                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=self.config.model_name
                )

            logger.info(f"Model registered as {self.config.model_name}")

            return registered_model

        except Exception as e:
            raise CustomException(e)
    
    # Promote model to Production stage in MLflow
    def promote_to_production(self, model_version):
        try:
            client = mlflow.tracking.MlflowClient()

            client.transition_model_version_stage(
                name=self.config.model_name,
                version=model_version,
                stage="Production"
            )

            logger.info(f"Model version {model_version} promoted to Production")

        except Exception as e:
            raise CustomException(e)
    

    def initiate_model_evaluation(self, X_test, y_test):
        try:
            logger.info("Starting model evaluation pipeline")

            model = self.load_model()

            metrics = self.evaluate(model, X_test, y_test)

            self.save_report(metrics)

            if not self.is_model_acceptable(metrics):
                raise Exception("Model rejected due to poor performance")

            # Register model
            registered_model = self.register_model(model,metrics)

            # Promote to production
            self.promote_to_production(registered_model.version)

            logger.info("Model evaluation and registration completed")

            return registered_model

        except Exception as e:
            logger.error("Model evaluation failed")
            raise CustomException(e)