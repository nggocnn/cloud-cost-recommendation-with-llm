"""
Core types and enums for the LLM cost recommendation system.
"""

from enum import Enum
from typing import Union


class CloudProvider(str, Enum):
    """Supported cloud providers"""
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"


class ServiceType:
    """Hierarchical cloud service types organized by provider"""
    
    class AWS(str, Enum):
        """AWS Services"""
        EC2 = "AWS.EC2"
        EBS = "AWS.EBS"
        S3 = "AWS.S3"
        EFS = "AWS.EFS"
        RDS = "AWS.RDS"
        RDS_SNAPSHOTS = "AWS.RDS_SNAPSHOTS"
        DYNAMODB = "AWS.DynamoDB"
        LAMBDA = "AWS.Lambda"
        CLOUDFRONT = "AWS.CloudFront"
        ALB = "AWS.ALB"
        NLB = "AWS.NLB"
        GWLB = "AWS.GWLB"
        ELASTIC_IP = "AWS.ElasticIP"
        NAT_GATEWAY = "AWS.NATGateway"
        VPC_ENDPOINTS = "AWS.VPCEndpoints"
        SQS = "AWS.SQS"
        SNS = "AWS.SNS"
    
    class Azure(str, Enum):
        """Azure Services"""
        VM = "Azure.VM"
        DISK = "Azure.Disk"
        STORAGE = "Azure.Storage"
        SQL = "Azure.SQL"
        COSMOS = "Azure.Cosmos"
        FUNCTIONS = "Azure.Functions"
        CDN = "Azure.CDN"
        LOAD_BALANCER = "Azure.LoadBalancer"
        PUBLIC_IP = "Azure.PublicIP"
        NAT_GATEWAY = "Azure.NATGateway"
    
    class GCP(str, Enum):
        """GCP Services"""
        COMPUTE = "GCP.Compute"
        DISK = "GCP.Disk"
        STORAGE = "GCP.Storage"
        SQL = "GCP.SQL"
        FIRESTORE = "GCP.Firestore"
        FUNCTIONS = "GCP.Functions"
        CDN = "GCP.CDN"
        LOAD_BALANCER = "GCP.LoadBalancer"
    
    @classmethod
    def get_all_services(cls):
        """Get all service types from all providers"""
        services = []
        services.extend(list(cls.AWS))
        services.extend(list(cls.Azure))
        services.extend(list(cls.GCP))
        return services
    
    @classmethod
    def get_aws_services(cls):
        """Get all AWS services"""
        return list(cls.AWS)
    
    @classmethod
    def get_azure_services(cls):
        """Get all Azure services"""
        return list(cls.Azure)
    
    @classmethod
    def get_gcp_services(cls):
        """Get all GCP services"""
        return list(cls.GCP)
    
    @classmethod
    def get_provider(cls, service_type):
        """Get the cloud provider for a service type"""
        return service_type.value.split('.')[0]
    
    @classmethod
    def get_service_name(cls, service_type):
        """Get the service name without provider prefix"""
        return service_type.value.split('.')[1]
    
    @classmethod
    def from_string(cls, service_string: str):
        """Get service type from string value"""
        for provider_class in [cls.AWS, cls.Azure, cls.GCP]:
            for service in provider_class:
                if service.value == service_string:
                    return service
        return None


# Union type for Pydantic compatibility
ServiceTypeUnion = Union[ServiceType.AWS, ServiceType.Azure, ServiceType.GCP]


class RecommendationType(str, Enum):
    """Types of recommendations"""
    RIGHTSIZING = "rightsizing"
    PURCHASING_OPTION = "purchasing_option"
    LIFECYCLE = "lifecycle"
    TOPOLOGY = "topology"
    STORAGE_CLASS = "storage_class"
    IDLE_RESOURCE = "idle_resource"


class RiskLevel(str, Enum):
    """Risk levels for recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConditionOperator(str, Enum):
    """Operators for custom conditions"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class ConditionField(str, Enum):
    """Available fields for conditions"""
    # Resource fields
    RESOURCE_ID = "resource_id"
    SERVICE = "service"
    REGION = "region"
    AZ = "availability_zone"
    ACCOUNT_ID = "account_id"

    # Tag fields (use tag.key_name)
    TAG = "tag"

    # Metrics fields
    CPU_P50 = "cpu_utilization_p50"
    CPU_P90 = "cpu_utilization_p90"
    CPU_P95 = "cpu_utilization_p95"
    MEMORY_P50 = "memory_utilization_p50"
    MEMORY_P90 = "memory_utilization_p90"
    MEMORY_P95 = "memory_utilization_p95"

    # Network metrics
    NETWORK_IN = "network_in"
    NETWORK_OUT = "network_out"

    # Storage metrics
    IOPS_READ = "iops_read"
    IOPS_WRITE = "iops_write"
    THROUGHPUT_READ = "throughput_read"
    THROUGHPUT_WRITE = "throughput_write"

    # Cost fields
    MONTHLY_COST = "monthly_cost"
    DAILY_COST = "daily_cost"

    # Time-based fields
    CREATED_AT = "created_at"
    AGE_DAYS = "age_days"

    # Boolean fields
    IS_IDLE = "is_idle"

    # Custom property fields (use property.key_name)
    PROPERTY = "property"
