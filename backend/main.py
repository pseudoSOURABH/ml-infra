#!/usr/bin/env python3
"""FastAPI backend service for clinical assertion inference."""

import os
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import numpy as np

from models import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse
)
from triton_client import TritonClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
TRITON_URL = os.getenv("TRITON_URL", "triton-inference.triton.svc.cluster.local:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "clinical_assertion")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "32"))

# Global state
tokenizer = None
triton_client: Optional[TritonClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tokenizer, triton_client
    
    # Startup
    logger.info("Starting up...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
    logger.info("Tokenizer loaded")
    
    # Initialize Triton client
    triton_client = TritonClient(
        triton_url=TRITON_URL,
        model_name=MODEL_NAME,
        max_connections=4
    )
    await triton_client.initialize()
    
    # Wait for model readiness
    retry_count = 0
    while retry_count < 30:  # 5 min max wait
        if await triton_client.is_model_ready():
            logger.info("Model ready for inference")
            break
        logger.warning(f"Model not ready, retrying... ({retry_count + 1}/30)")
        await asyncio.sleep(10)
        retry_count += 1
    else:
        logger.error("Failed to connect to model after retries")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if triton_client:
        await triton_client.close()


# Create FastAPI app
app = FastAPI(
    title="Clinical Assertion API",
    description="CPU-optimized inference for clinical text classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_ready = False
    triton_connected = False
    
    if triton_client:
        try:
            triton_connected = True
            model_ready = await triton_client.is_model_ready()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    return HealthResponse(
        status="healthy" if (model_ready and triton_connected) else "degraded",
        model_ready=model_ready,
        triton_connected=triton_connected
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Predict assertion status for a single sentence.
    
    - **sentence**: Clinical text containing a medical condition
    - Returns: Classification label and confidence score
    """
    if not triton_client or not await triton_client.is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        tokens = tokenizer(
            request.sentence,
            return_tensors="np",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        
        # Run inference
        result = await triton_client.predict(
            tokens,
            request_id=f"req-{int(time.time() * 1000)}"
        )
        
        # Log latency for monitoring
        latency_ms = (time.time() - start_time) * 1000
        if latency_ms > 100:
            logger.warning(f"Slow inference: {latency_ms:.2f}ms for '{request.sentence[:50]}...'")
        else:
            logger.debug(f"Inference: {latency_ms:.2f}ms")
        
        return PredictResponse(
            label=result["label"],
            score=result["score"]
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction endpoint for multiple sentences.
    
    - **sentences**: List of clinical texts (max 32)
    - Returns: List of predictions with processing time
    """
    if not triton_client or not await triton_client.is_model_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if len(request.sentences) > BATCH_SIZE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit of {BATCH_SIZE_LIMIT}"
        )
    
    start_time = time.time()
    
    try:
        # Tokenize batch
        tokens = tokenizer(
            request.sentences,
            return_tensors="np",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        
        # Run batch inference
        results = await triton_client.predict_batch(tokens)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictResponse(
            results=[PredictResponse(**r) for r in results],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """API root with documentation link."""
    return {
        "message": "Clinical Assertion API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=os.getenv("ENV", "production") != "production",
        workers=int(os.getenv("WORKERS", "1"))
    )