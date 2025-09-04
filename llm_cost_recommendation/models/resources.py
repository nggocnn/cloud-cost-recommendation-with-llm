"""
Resource models for cloud infrastructure.
"""

from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

from .types import ServiceTypeUnion


class Resource(BaseModel):
    """Base resource model"""

    resource_id: str
    service: ServiceTypeUnion
    region: str
    availability_zone: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    # Service-specific extensions
    extensions: Dict[str, Any] = Field(default_factory=dict)
