from fastapi import FastAPI
from api.routes import predict
from api.utils.error_handler import custom_exception_handler
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Initializing FastAPI app"),
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# Health check
@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "ok"}

# Register routes
app.include_router(predict.router)

app.add_exception_handler(Exception, custom_exception_handler)