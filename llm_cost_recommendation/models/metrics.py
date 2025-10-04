"""
Metrics and billing data models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class TimeSeriesPoint(BaseModel):
    """Single time-series data point"""

    timestamp: str
    value: float


class Metrics(BaseModel):
    """Resource metrics model"""

    resource_id: str
    timestamp: datetime
    metrics: Dict[str, float] = Field(default_factory=dict)
    period_days: int = Field(default=30)

    # Common metrics patterns
    cpu_utilization_p50: Optional[float] = None
    cpu_utilization_p90: Optional[float] = None
    cpu_utilization_p95: Optional[float] = None
    cpu_utilization_min: Optional[float] = None
    cpu_utilization_max: Optional[float] = None
    cpu_utilization_stddev: Optional[float] = None
    cpu_utilization_trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    cpu_utilization_volatility: Optional[str] = None  # "low", "moderate", "high"
    cpu_utilization_peak_hours: List[int] = Field(default_factory=list)
    cpu_utilization_patterns: Dict[str, Any] = Field(default_factory=dict)
    cpu_timeseries_data: List[Dict[str, Any]] = Field(default_factory=list)

    memory_utilization_p50: Optional[float] = None
    memory_utilization_p90: Optional[float] = None
    memory_utilization_p95: Optional[float] = None

    # Storage metrics
    iops_read: Optional[float] = None
    iops_write: Optional[float] = None
    throughput_read: Optional[float] = None
    throughput_write: Optional[float] = None

    # Network metrics
    network_in: Optional[float] = None
    network_out: Optional[float] = None

    # Usage patterns
    is_idle: bool = False
    peak_usage_hours: List[int] = Field(default_factory=list)


class BillingData(BaseModel):
    """Billing data model"""

    service: str
    resource_id: Optional[str] = None
    region: str
    usage_type: str
    usage_amount: float
    usage_unit: str
    unblended_cost: float = Field(ge=0, description="Cost must be non-negative")
    amortized_cost: float = Field(
        ge=0, description="Amortized cost must be non-negative"
    )
    credit: float = Field(default=0.0, description="Applied credits")
    bill_period_start: datetime
    bill_period_end: datetime
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("bill_period_end")
    @classmethod
    def end_after_start(cls, v, values):
        if hasattr(values, "data") and "bill_period_start" in values.data:
            if v <= values.data["bill_period_start"]:
                raise ValueError("bill_period_end must be after bill_period_start")
        return v
