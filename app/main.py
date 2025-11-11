from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .logging_config import get_logger
from .model_loader import predict_from_bytes, load_model_once

logger = get_logger("app")
app = FastAPI(title="DenseNet TinyImageNet API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting up and loading model...")
    load_model_once()

@app.get("/")
async def root():
    return {"message": "✅ DenseNet FastAPI is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"📁 Received file: {file.filename}")
    try:
        contents = await file.read()
        result = predict_from_bytes(contents)
        logger.info(f"📊 Prediction result: {result}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("❌ Error handling /predict")
        return JSONResponse(content={"error": str(e)}, status_code=500)
