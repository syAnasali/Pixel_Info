import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.predictor import CaptionPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load predictor once and cache in app state
    logger.info("Starting up FastAPI application...")
    try:
        predictor = CaptionPredictor()
        app.state.predictor = predictor
        logger.info("CaptionPredictor initialized successfully and cached in app state.")
    except Exception as e:
        logger.critical(f"Failed to initialize CaptionPredictor on startup: {e}", exc_info=True)
        app.state.predictor = None
        
    yield
    
    # Shutdown: clean up if necessary
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Pixel_Info Image Captioning API",
    description="FastAPI backend for generating image captions using deep learning (CNN-LSTM with ResNet50 and Beam Search).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Register routes router
app.include_router(router)
