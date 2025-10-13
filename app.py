import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

# Env variables configuration
APP_MODEL = os.getenv("HF_MODEL", "Salesforce/blip-image-captioning-base")
PORT = int(os.getenv("PORT", 8000))

# Global variables for model and processor
processor = None
model = None
device = None

# Pydantic model for request validation
class CaptionRequest(BaseModel):
    image_url: Optional[str] = Field(None, alias="imageUrl")

    class Config:
        populate_by_name = True
        validate_by_name = True


# Lifespan context for FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/Shutdown context for FastAPI
    Loads the Hugging Face BLIP model once at startup.
    """
    global processor, model, device

    # Cache directory for Hugging Face models
    hf_cache = Path(__file__).resolve().parent / ".cache" / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {APP_MODEL} to device {device} ...")

    # Load model and processor
    processor = BlipProcessor.from_pretrained(APP_MODEL, use_fast=True)
    model_instance = BlipForConditionalGeneration.from_pretrained(APP_MODEL)
    model_instance.to(device).eval()
    model = model_instance
    print("Model loaded.")

    yield


# FastAPI app instance
app = FastAPI(
    title="BLIP caption API",
    version="0.3",
    root_path="/api",
    lifespan=lifespan,
)

# CORS middleware - Allow all origins for open source research project
# This is intentionally permissive to enable easy reproduction of thesis results
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/health")
def health():
    loaded = model is not None and processor is not None
    return JSONResponse(
        {
            "status": "ok" if loaded else "not_loaded",
            "model": APP_MODEL,
            "version": "0.2",
            "device": str(device),
        }
    )


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    return Image.open(resp.raw).convert("RGB")


@app.post("/caption")
async def caption_endpoint(payload: CaptionRequest):
    if not payload.image_url:
        raise HTTPException(
            status_code=400, detail="Field 'imageUrl' is required in JSON body."
        )

    try:
        img = load_image_from_url(payload.image_url)
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return {"caption": caption}

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Local development setup
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)
