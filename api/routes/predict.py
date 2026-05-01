from fastapi import APIRouter
from api.schemas import ChurnRequest
from api.services.prediction_service import PredictionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

logger.info("Initializing PredictionService in API route"),
# Initialize once (singleton)
service = PredictionService(
    model_path="artifacts/model.pkl",
    preprocessor_path="artifacts/preprocessor.pkl"
)

@router.post("/predict")
def predict(data: ChurnRequest):
    logger.info("Received prediction request")
    result = service.predict(data.dict())
    return {
        "status": "success",
        "data": result
    }