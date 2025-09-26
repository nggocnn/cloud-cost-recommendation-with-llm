"""
Resource models for cloud infrastructure.
"""

from datetime import datetime
from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator

from .types import ServiceTypeUnion, ServiceType


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

    @field_validator('service', mode='before')
    @classmethod
    def validate_service(cls, v):
        """Convert string service names to ServiceType enum"""
        if isinstance(v, str):
            # Map common service name strings to ServiceType
            service_mapping = {
                "EC2": ServiceType.AWS.EC2,
                "EBS": ServiceType.AWS.EBS,
                "S3": ServiceType.AWS.S3,
                "EFS": ServiceType.AWS.EFS,
                "RDS": ServiceType.AWS.RDS,
                "LAMBDA": ServiceType.AWS.LAMBDA,
                "ALB": ServiceType.AWS.ALB,
                "NLB": ServiceType.AWS.NLB,
                "CLOUDFRONT": ServiceType.AWS.CLOUDFRONT,
                "DYNAMODB": ServiceType.AWS.DYNAMODB,
                "ELASTIC_IP": ServiceType.AWS.ELASTIC_IP,
                "NAT_GATEWAY": ServiceType.AWS.NAT_GATEWAY,
                "VPC_ENDPOINTS": ServiceType.AWS.VPC_ENDPOINTS,
                "SQS": ServiceType.AWS.SQS,
                "SNS": ServiceType.AWS.SNS,
                # Add more mappings as needed
            }
            
            mapped_service = service_mapping.get(v.upper())
            if mapped_service:
                return mapped_service
            else:
                # Try to find a match by checking if the string is in any service type value
                for service_type in [ServiceType.AWS, ServiceType.Azure, ServiceType.GCP]:
                    for service in service_type:
                        if v.upper() in service.value.upper() or service.value.upper() in v.upper():
                            return service
                
                # If no match found, raise an error
                raise ValueError(f"Unknown service type: {v}")
        
        return v
