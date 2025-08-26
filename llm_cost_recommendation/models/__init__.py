"""
Core data models for the LLM cost recommendation system.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


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


class CustomCondition(BaseModel):
    """Custom condition for agent decision making"""

    field: str  # Field to evaluate (can be ConditionField or custom like "tag.Environment")
    operator: ConditionOperator
    value: Union[str, int, float, bool, List[Union[str, int, float]]]
    description: Optional[str] = None


class ConditionalRule(BaseModel):
    """A rule with conditions and actions"""

    name: str
    description: str
    conditions: List[CustomCondition]
    logic: Literal["AND", "OR"] = "AND"  # How to combine conditions

    # Actions when conditions are met
    actions: Dict[str, Any] = Field(default_factory=dict)

    # Override thresholds when conditions are met
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)

    # Skip certain recommendation types
    skip_recommendation_types: List[RecommendationType] = Field(default_factory=list)

    # Force certain recommendation types
    force_recommendation_types: List[RecommendationType] = Field(default_factory=list)

    # Custom prompt additions
    custom_prompt: Optional[str] = None

    # Risk level adjustment
    risk_adjustment: Optional[Literal["increase", "decrease"]] = None

    # Priority for this rule (higher = more important)
    priority: int = 0

    # Enable/disable the rule
    enabled: bool = True


class Resource(BaseModel):
    """Base resource model"""

    resource_id: str
    service: ServiceTypeUnion
    region: str
    availability_zone: Optional[str] = None
    account_id: str
    tags: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    # Service-specific extensions
    extensions: Dict[str, Any] = Field(default_factory=dict)


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

    account_id: str
    service: str
    resource_id: Optional[str] = None
    region: str
    usage_type: str
    usage_amount: float
    usage_unit: str
    unblended_cost: float
    amortized_cost: float
    credit: float = 0.0
    bill_period_start: datetime
    bill_period_end: datetime
    tags: Dict[str, str] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """Recommendation model"""

    id: str = Field(..., description="Unique recommendation identifier")
    resource_id: str
    service: ServiceTypeUnion
    recommendation_type: RecommendationType

    # Current configuration
    current_config: Dict[str, Any]

    # Recommended configuration
    recommended_config: Dict[str, Any]

    # Cost impact
    current_monthly_cost: float
    estimated_monthly_cost: float
    estimated_monthly_savings: float
    annual_savings: float

    # Risk and impact
    risk_level: RiskLevel
    impact_description: str
    rollback_plan: str

    # Detailed reasoning
    rationale: str
    evidence: Dict[str, Any] = Field(default_factory=dict)

    # Implementation details
    implementation_steps: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)

    # Metadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str

    # Business context
    business_hours_impact: bool = False
    downtime_required: bool = False
    sla_impact: Optional[str] = None


class RecommendationReport(BaseModel):
    """Complete recommendation report"""

    id: str
    account_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Summary metrics
    total_monthly_savings: float
    total_annual_savings: float
    total_recommendations: int

    # Recommendations by category
    recommendations: List[Recommendation]

    # Risk distribution
    low_risk_count: int = 0
    medium_risk_count: int = 0
    high_risk_count: int = 0

    # Service breakdown
    savings_by_service: Dict[str, float] = Field(default_factory=dict)

    # Implementation timeline
    quick_wins: List[str] = Field(
        default_factory=list
    )  # Can be implemented immediately
    medium_term: List[str] = Field(default_factory=list)  # Require planning
    long_term: List[str] = Field(default_factory=list)  # Require significant changes

    # Quality metrics
    coverage: Dict[str, Any] = Field(default_factory=dict)  # What was analyzed
    data_quality_issues: List[str] = Field(default_factory=list)


class AgentCapability(BaseModel):
    """Service agent capability definition"""

    service: ServiceTypeUnion
    supported_recommendation_types: List[RecommendationType]
    required_metrics: List[str]
    optional_metrics: List[str] = Field(default_factory=list)

    # Configuration
    thresholds: Dict[str, float] = Field(default_factory=dict)
    analysis_window_days: int = 30


class ServiceAgentConfig(BaseModel):
    """Configuration for service agents"""

    agent_id: str
    service: ServiceTypeUnion
    enabled: bool = True

    # Capability definition
    capability: AgentCapability

    # Prompt configuration
    base_prompt: str
    service_specific_prompt: str

    # LLM settings
    temperature: float = 0.1
    max_tokens: int = 2000

    # Analysis settings
    min_cost_threshold: float = 1.0  # Don't recommend for resources < $1/month
    confidence_threshold: float = 0.7  # Minimum confidence to include recommendation

    # Custom conditional rules
    custom_rules: List[ConditionalRule] = Field(default_factory=list)


class CoordinatorConfig(BaseModel):
    """Configuration for the coordinator agent"""

    enabled_services: List[ServiceTypeUnion]

    # Deduplication settings
    similarity_threshold: float = 0.8

    # Ranking weights
    savings_weight: float = 0.4
    risk_weight: float = 0.3
    confidence_weight: float = 0.2
    implementation_ease_weight: float = 0.1

    # Report settings
    max_recommendations_per_service: int = 50
    include_low_impact: bool = False  # Include recommendations < $10/month savings
