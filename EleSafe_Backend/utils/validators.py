from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class PredictRequest(BaseModel):
    """Request model for risk prediction"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

    @validator('date')
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')


class ForecastRequest(BaseModel):
    """Request model for temporal forecast"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    forecast_days: int = Field(default=7, ge=1, le=30, description="Number of days to forecast")


class HotspotsRequest(BaseModel):
    """Request model for hotspot retrieval"""
    date: Optional[str] = None
    risk_threshold: float = Field(default=0.7, ge=0, le=1)

    @validator('date')
    def validate_date(cls, v):
        """Validate date format if provided"""
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
                return v
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v