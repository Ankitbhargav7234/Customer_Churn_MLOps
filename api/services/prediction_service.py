import sys
import pandas as pd
#from src.transformation.data_transformation import transform
from src.utils.common import load_object
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)
class PredictionService:
    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            logger.info("Initializing PredictionService")
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
        except Exception as e:
            raise CustomException(e)
    
    def preprocess(self, input_data: dict):
        try:
            logger.info("Preprocessing input data")
            df = pd.DataFrame([input_data])
            transformed = self.preprocessor.transform(df)
            return transformed
        except Exception as e:
            raise CustomException(e)
    
    def predict(self, input_data: dict):
        try:
            logger.info("Making prediction")
            processed_data = self.preprocess(input_data)

            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0][1]

            return {
                "prediction": int(prediction),
                "probability": float(probability)
            }

        except Exception as e:
            raise CustomException(e)