import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# CRITICAL: Set thread limits BEFORE any ML library imports
# Strategy:
# - 1 worker: Use default threading (PyTorch uses all available vCPUs)
# - 2+ workers: Dynamically allocate threads to match hardware (vCPUs / workers)
#   Example: 16 vCPUs with 4 workers = 4 threads per worker
workers = int(os.getenv("WORKERS", "1"))

if workers > 1:
    # Get actual CPU count from the system
    cpu_count = os.cpu_count() or 1

    # Allocate threads per worker: divide vCPUs evenly across workers
    # Use max(1, ...) to ensure at least 1 thread per worker
    threads_per_worker = max(1, cpu_count // workers)

    # Set thread limits for all numerical libraries
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)

    total_threads = workers * threads_per_worker
    print(f"[Config] {workers} workers on {cpu_count} vCPUs: {threads_per_worker} thread(s) per worker (total: {total_threads} threads)")
else:
    # Single worker: use all available threads (default behavior)
    cpu_count = os.cpu_count() or 1
    print(f"[Config] 1 worker: Using default threading (will use all {cpu_count} vCPUs)")

from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import asyncio
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

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
    Uses a Semaphore at endpoint level to limit concurrency per worker.
    """
    global processor, model, device

    # Create semaphore within event loop context and store in app.state
    app.state.inference_semaphore = asyncio.Semaphore(1)
    print("[Debug] Inference semaphore created (limit: 1 concurrent request per worker)")

    # Debug: Print actual thread configuration
    print(f"[Debug] Worker PID: {os.getpid()}")    
    print(f"[Debug] PyTorch threads (for this worker): {torch.get_num_threads()}")

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
    """
    Health check endpoint with worker information for debugging multi-worker deployments
    """
    loaded = model is not None and processor is not None

    # Get worker info (useful for debugging multi-worker setups)
    worker_pid = os.getpid()

    return JSONResponse(
        {
            "status": "ok" if loaded else "not_loaded",
            "model": APP_MODEL,
            "version": "0.2",
            "device": str(device),
            "worker_pid": worker_pid,
            "workers_configured": os.getenv("WORKERS", "1"),
        }
    )


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    return Image.open(resp.raw).convert("RGB")


@app.post("/caption")
async def caption_endpoint(request: Request, payload: CaptionRequest):
    if not payload.image_url:
        raise HTTPException(
            status_code=400, detail="Field 'imageUrl' is required in JSON body."
        )

    worker_pid = os.getpid()
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"[{timestamp}] [worker_pid={worker_pid}] Received caption request for {payload.image_url}")

    # Use semaphore to ensure only 1 request processes at a time per worker
    async with request.app.state.inference_semaphore:
        semaphore_acquired_time = time.time()
        wait_time = semaphore_acquired_time - start_time
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(f"[{timestamp}] [worker_pid={worker_pid}] Processing request (acquired semaphore after {wait_time:.3f}s wait)")

        try:
            # Run blocking image download in thread pool
            download_start = time.time()
            img = await asyncio.to_thread(load_image_from_url, payload.image_url)
            download_time = time.time() - download_start

            # Run blocking ML inference in thread pool
            def run_inference():
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                return processor.decode(generated_ids[0], skip_special_tokens=True)

            inference_start = time.time()
            caption = await asyncio.to_thread(run_inference)
            inference_time = time.time() - inference_start

            total_time = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            print(f"[{timestamp}] [worker_pid={worker_pid}] Generated caption: {caption}")
            print(f"[{timestamp}] [worker_pid={worker_pid}] Timings - Wait: {wait_time:.3f}s, Download: {download_time:.3f}s, Inference: {inference_time:.3f}s, Total: {total_time:.3f}s")

            return {"caption": caption}

        except requests.RequestException as e:
            print(f"[worker_pid={worker_pid}] Image download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Could not download image: {str(e)}")
        except Exception as e:
            print(f"[worker_pid={worker_pid}] Inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Local development setup
if __name__ == "__main__":
    import sys

    # Get configuration from environment
    workers = int(os.getenv("WORKERS", "1"))
    host = os.getenv("HOST", "127.0.0.1")
    port = PORT

    print("=" * 42)
    print("BLIP Caption API - Starting Server")
    print("=" * 42)
    print("Configuration:")
    print(f"  Workers: {workers}")
    print(f"  Host:    {host}")
    print(f"  Port:    {port}")
    print("=" * 42)

    if workers == 1:
        print("Starting with Uvicorn (single worker mode)")
        print("Configuration: Baseline/Cloud-native (Config A/D)")
        print("=" * 42)

        import uvicorn
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            log_level="info"
        )
    else:       

        print(f"\nStarting {workers} workers. Each worker will load the model independently.")
        print("This may take a moment...\n")

        # Use Gunicorn with config file (same as Docker/start.sh)
        # All configuration is in gunicorn.conf.py
        import subprocess
        subprocess.run([
            "gunicorn",
            "app:app",
            "-c", "gunicorn.conf.py"
        ])
