from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
import re


class PredictRequest(BaseModel):
    """Single prediction request."""
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Clinical text to analyze"
    )

    @field_validator('sentence')          # v2: @field_validator not @validator
    @classmethod
    def sanitize(cls, v: str) -> str:
        return re.sub(r'\s+', ' ', v.strip())


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    sentences: List[str] = Field(
        ...,
        min_length=1,                     # v2: min_length/max_length on List, not min_items/max_items
        max_length=32,
        description="List of sentences (max 32)"
    )

    @field_validator('sentences', mode='before')
    @classmethod
    def sanitize_each(cls, v: list) -> list:
        return [re.sub(r'\s+', ' ', s.strip()) for s in v]


class PredictResponse(BaseModel):
    """Single prediction response."""
    label: str = Field(
        ...,
        description="Classification label: PRESENT, ABSENT, or CONDITIONAL"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    model_config = {                      # v2: model_config dict replaces inner Config class
        "json_schema_extra": {
            "example": {"label": "ABSENT", "score": 0.9842}
        }
    }


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictResponse]
    processing_time_ms: Optional[float] = Field(
        None,
        description="Server processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_ready: bool
    triton_connected: bool
    version: str = "1.0.0"