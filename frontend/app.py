#!/usr/bin/env python3
"""Simple frontend for testing clinical assertion inference."""

import os
import logging
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend.api.svc.cluster.local:8080")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

app = FastAPI(title="Clinical Assertion UI", version="1.0.0")
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def api_predict(sentence: str = Form(...)):
    """Proxy endpoint to backend."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                f"{BACKEND_URL}/predict",
                json={"sentence": sentence},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPError as e:
            logger.error(f"Backend request failed: {e}")
            raise HTTPException(status_code=502, detail="Backend service unavailable")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "frontend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)