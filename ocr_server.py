"""
DeepSeek OCR API Server
========================
HTTP API that accepts an image and returns markdown via DeepSeek-OCR.

Setup:
    conda create -n ollama python=3.11
    conda activate ollama
    pip install ollama fastapi uvicorn python-multipart httpx

Run:
    python ocr_server.py

Endpoints:
    POST /ocr          - Upload image file, returns markdown
    POST /ocr/base64   - Send base64 image, returns markdown
    GET  /health       - Health check + model status
"""

import os
import base64
import httpx
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ollama import Client

# ── Config ────────────────────────────────────────────────────────────────────

os.environ["OLLAMA_FLASH_ATTENTION"] = "1"  # Speeds up DeepSeek-OCR on Windows

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME  = "deepseek-ocr:3b"

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=180.0,
    write=10.0,
    pool=10.0,
)

# ── Ollama client ─────────────────────────────────────────────────────────────

client = Client(host=OLLAMA_HOST, timeout=TIMEOUT)

def ensure_model():
    """Pull model if not already downloaded."""
    try:
        installed = [m.model for m in client.list().models]
        if MODEL_NAME not in installed:
            print(f"📥 Pulling {MODEL_NAME} ...")
            client.pull(MODEL_NAME)
            print("✅ Model ready.")
        else:
            print(f"✅ Model '{MODEL_NAME}' already installed.")
    except Exception as e:
        print(f"⚠️  Could not verify model: {e}")

# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()
    yield

app = FastAPI(
    title="DeepSeek OCR API",
    description="Convert images to markdown using DeepSeek-OCR via Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_ocr(image_bytes: bytes, prompt: str) -> str:
    """Run DeepSeek-OCR and return the text response."""
    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images=[image_bytes],
            options={
                "temperature": 0,
                "keep_alive": 0,      # Release VRAM after each call
                "num_ctx": 8192,      # Large context for image tokens
            },
        )
        return response["response"].strip()
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="OCR timed out (>180 s). Try a smaller image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────

class Base64Request(BaseModel):
    image_base64: str
    prompt: Optional[str] = DEFAULT_PROMPT
    filename: Optional[str] = "image.png"

class OCRResponse(BaseModel):
    markdown: str
    model: str
    prompt: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check server and model status."""
    try:
        installed = [m.model for m in client.list().models]
        model_ready = MODEL_NAME in installed
        return {
            "status": "ok",
            "ollama": OLLAMA_HOST,
            "model": MODEL_NAME,
            "model_ready": model_ready,
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.post("/ocr", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(..., description="Image file (PNG, JPG, WEBP, etc.)"),
    prompt: str = Form(DEFAULT_PROMPT, description="OCR prompt sent to the model"),
):
    """
    Upload an image file and receive markdown back.

    Example (curl):
        curl -X POST http://localhost:8000/ocr \\
             -F "file=@invoice.png" \\
             -F "prompt=<image>\\n<|grounding|>Convert the document to markdown."
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    markdown = run_ocr(contents, prompt)
    return OCRResponse(markdown=markdown, model=MODEL_NAME, prompt=prompt)


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(body: Base64Request):
    """
    Send a base64-encoded image and receive markdown back.

    Example (Python):
        import base64, requests
        with open("invoice.png", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post("http://localhost:8000/ocr/base64",
                          json={"image_base64": b64})
        print(r.json()["markdown"])
    """
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data.")

    markdown = run_ocr(image_bytes, body.prompt)
    return OCRResponse(markdown=markdown, model=MODEL_NAME, prompt=body.prompt)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ocr_server:app", host="0.0.0.0", port=8000, reload=False)