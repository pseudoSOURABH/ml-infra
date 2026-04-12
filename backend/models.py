from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re


class PredictRequest(BaseModel):
    """Single prediction request."""
    sentence: str = Field(..., min_length=1, max_length=512, description="Clinical text to analyze")
    
    @validator('sentence')
    def sanitize(cls, v):
        """Basic sanitization."""
        return re.sub(r'\s+', ' ', v.strip())


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    sentences: List[str] = Field(..., min_items=1, max_items=32, description="List of sentences")
    
    @validator('sentences', each_item=True)
    def sanitize_each(cls, v):
        return re.sub(r'\s+', ' ', v.strip())


class PredictResponse(BaseModel):
    """Single prediction response."""
    label: str = Field(..., description="Classification label: PRESENT, ABSENT, or CONDITIONAL")
    score: float = Field(..., ge=0, le=1, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {"label": "ABSENT", "score": 0.9842}
        }


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictResponse]
    processing_time_ms: Optional[float] = Field(None, description="Server processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_ready: bool
    triton_connected: bool
    version: str = "1.0.0"