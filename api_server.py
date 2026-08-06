"""
FastAPI Microservice REST API Server for Adaptive Super-Resolution Engine
Exposes HTTP POST endpoints for external projects (Land Cover, Water Change Detection, GIS applications).
"""

import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from PIL import Image
from sr_pipeline_service import RemoteSensingSREngine

app = FastAPI(
    title="Remote Sensing Adaptive Super-Resolution Microservice",
    description="REST API service providing 4x Region-Aware Adaptive Super-Resolution for downstream Earth Observation tasks.",
    version="1.0.0"
)

# Initialize engine globally on startup
engine = None

@app.on_event("startup")
def load_models():
    global engine
    ckpt_dir = os.getenv("CKPT_DIR", "all_checkpoints")
    engine = RemoteSensingSREngine(ckpt_dir=ckpt_dir)
    print("FastAPI Microservice Engine loaded successfully!")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Adaptive Super-Resolution Microservice",
        "scale_factor": "4x",
        "device": engine.device if engine else "unknown"
    }

@app.post("/v1/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded satellite/drone image (PNG, JPEG, TIFF) and returns the 4x super-resolved image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    enhanced_pil = engine.enhance_pil_image(pil_img)

    img_byte_arr = io.BytesIO()
    enhanced_pil.save(img_byte_arr, format="PNG")
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
