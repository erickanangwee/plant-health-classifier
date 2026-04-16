from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    filename:          str
    prediction:        str   = Field(..., examples=["HEALTHY", "DISEASED"])
    confidence:        float = Field(..., ge=0.0, le=1.0,
                                    description="Model probability for predicted class")
    healthy_prob:      float = Field(..., ge=0.0, le=1.0)
    diseased_prob:     float = Field(..., ge=0.0, le=1.0)
    is_leaf:           bool
    leaf_similarity:   float = Field(...,
                                    description="Cosine similarity vs. training centroid")
    model_used:        str
    message:           Optional[str] = None


class RejectionDetail(BaseModel):
    filename:          str
    rejected:          bool  = True
    reason:            str
    leaf_similarity:   float
    threshold_used:    float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_type:   Optional[str] = None
    version:      str = "1.0.0"


class ClassesResponse(BaseModel):
    classes:   list[str]
    label_map: dict[str, int]
    description: str