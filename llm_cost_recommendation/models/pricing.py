"""
Pricing models for cloud cost optimization system.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from decimal import Decimal


class PricingModel(str, Enum):
    """Supported pricing models"""
    ON_DEMAND = "on_demand"
    RESERVED = "reserved"
    SPOT = "spot"
    SAVINGS_PLAN = "savings_plan"
    COMMITTED_USE = "committed_use"  # GCP
    PREEMPTIBLE = "preemptible"  # GCP


class ReservationTerm(str, Enum):
    """Reservation term lengths"""
    ONE_YEAR = "1_year"
    THREE_YEAR = "3_year"


class PaymentOption(str, Enum):
    """Payment options for reservations"""
    NO_UPFRONT = "no_upfront"
    PARTIAL_UPFRONT = "partial_upfront"
    ALL_UPFRONT = "all_upfront"


class Currency(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


class PricingUnit(BaseModel):
    """Base pricing unit"""
    amount: Decimal = Field(..., description="Price amount")
    currency: Currency = Field(default=Currency.USD)
    unit: str = Field(..., description="Pricing unit (e.g., 'hour', 'GB-month', 'request')")
    
    class Config:
        use_enum_values = True


class ReservationPricing(BaseModel):
    """Reservation pricing details"""
    term: ReservationTerm
    payment_option: PaymentOption
    upfront_cost: Decimal = Field(default=Decimal('0'))
    hourly_cost: Decimal = Field(default=Decimal('0'))
    effective_hourly_cost: Decimal = Field(..., description="Effective hourly cost including upfront")
    
    class Config:
        use_enum_values = True


class ServicePricing(BaseModel):
    """Pricing for a specific service configuration"""
    service_type: str = Field(..., description="Service type (e.g., 'AWS.EC2', 'AWS.S3')")
    region: str = Field(..., description="AWS region")
    instance_type: Optional[str] = Field(None, description="Instance type for compute services")
    storage_type: Optional[str] = Field(None, description="Storage type for storage services")
    
    # Pricing models
    on_demand: Optional[PricingUnit] = None
    reserved: Optional[List[ReservationPricing]] = None
    spot: Optional[PricingUnit] = None
    
    # Additional attributes for complex pricing
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="aws_pricing_api", description="Pricing data source")
    
    class Config:
        use_enum_values = True


class PricingRule(BaseModel):
    """Configuration for pricing calculations"""
    service_type: str
    cloud_provider: str
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Calculation rules
    calculation_method: str = Field(default="direct_lookup")
    price_dimensions: List[str] = Field(default_factory=list)
    
    # Caching
    cache_duration_hours: int = Field(default=24)
    enable_fallback: bool = Field(default=True)
    fallback_multiplier: float = Field(default=1.1, description="Fallback pricing multiplier")
    
    # Regional pricing
    region_mappings: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class CostCalculation(BaseModel):
    """Result of cost calculation"""
    resource_id: str
    service_type: str
    pricing_model: PricingModel
    
    # Current costs
    current_hourly_cost: Decimal
    current_monthly_cost: Decimal
    current_annual_cost: Decimal
    
    # Recommended costs (if different configuration)
    recommended_hourly_cost: Optional[Decimal] = None
    recommended_monthly_cost: Optional[Decimal] = None
    recommended_annual_cost: Optional[Decimal] = None
    
    # Savings potential
    monthly_savings: Decimal = Field(default=Decimal('0'))
    annual_savings: Decimal = Field(default=Decimal('0'))
    savings_percentage: float = Field(default=0.0)
    
    # Breakdown
    cost_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Metadata
    calculation_date: datetime = Field(default_factory=datetime.utcnow)
    assumptions: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class PricingCache(BaseModel):
    """Pricing cache entry"""
    cache_key: str
    service_pricing: ServicePricing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    hit_count: int = Field(default=0)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if cache entry is valid"""
        return not self.is_expired()


class PricingConfig(BaseModel):
    """Overall pricing configuration"""
    enabled: bool = Field(default=True)
    default_currency: Currency = Field(default=Currency.USD)
    default_region: str = Field(default="us-east-1")
    
    # API configuration
    aws_pricing_api_region: str = Field(default="us-east-1")
    enable_pricing_cache: bool = Field(default=True)
    cache_duration_hours: int = Field(default=24)
    
    # Rate limiting
    api_rate_limit_per_second: int = Field(default=10)
    max_concurrent_requests: int = Field(default=50)
    
    # Fallback pricing
    enable_fallback_pricing: bool = Field(default=True)
    fallback_pricing_file: Optional[str] = None
    
    class Config:
        use_enum_values = True
