from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from src.api.schemas import RootResponse, HealthResponse, CaptionResponse
import logging

logger = logging.getLogger("api_routes")
router = APIRouter()

@router.get("/", response_model=RootResponse)
async def root():
    return {"project": "Pixel_Info", "status": "running"}

@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "healthy"}

@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(request: Request, file: UploadFile = File(...)):
    # Verify file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        
    try:
        # Read file bytes
        image_bytes = await file.read()
        
        # Access predictor from app state
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Predictor service not initialized.")
            
        # Predict caption using beam search
        caption = predictor.predict(image_bytes, beam_width=3, alpha=0.75)
        
        return {"success": True, "caption": caption}
    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")
