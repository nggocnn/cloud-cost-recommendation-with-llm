# LLM Cost Recommendation System - Architecture & Usage Guide

## 🏗️ Architecture Overview

This is a **multi-agent LLM-powered AWS cost optimization system** that analyzes cloud resources and provides actionable cost reduction recommendations.

### Core Components

```
llm_cost_recommendation/
├── agents/           # Multi-agent system
│   ├── coordinator.py    # Orchestrates analysis
│   └── base.py          # Service-specific agents (EC2, S3, RDS, etc.)
├── models/           # Data models & schemas
├── services/         # Core services
│   ├── config.py        # Configuration management
│   ├── llm.py          # LangChain/OpenAI integration
│   ├── ingestion.py    # Data ingestion & processing
│   └── logging.py      # Enhanced logging system
├── cli.py           # Command-line interface
└── __main__.py      # Module entry point
```

## 🎯 System Flow

### 1. Data Ingestion
```
AWS Data Sources → DataIngestionService → Structured Models
```
- **Billing Data**: CSV files with cost information
- **Inventory Data**: JSON files with resource configurations
- **Metrics Data**: CSV files with performance metrics

### 2. Multi-Agent Analysis
```
Coordinator Agent → Service Agents → LLM Analysis → Recommendations
```
- **Coordinator**: Orchestrates the analysis workflow
- **Service Agents**: Specialized agents for each AWS service (EC2, S3, RDS, etc.)
- **LLM Integration**: Uses GPT-4 for intelligent analysis

### 3. Report Generation
```
Recommendations → Aggregation → Risk Assessment → Final Report
```

## ⚙️ Configuration System

### Environment Configuration (`.env`)
```bash
# LLM Configuration
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# Optional: Model provider settings
LLM_PROVIDER=openai  # openai, anthropic, etc.
```

### Service Configuration (`config/*.yaml`)

#### Coordinator Settings (`config/coordinator.yaml`)
```yaml
enabled_services:
  - EC2
  - EBS
  - S3
  - RDS
  # ... more services

# Recommendation weighting
savings_weight: 0.4          # 40% weight on cost savings
risk_weight: 0.3             # 30% weight on implementation risk
confidence_weight: 0.2       # 20% weight on confidence level
implementation_ease_weight: 0.1  # 10% weight on ease of implementation

# Analysis parameters
max_recommendations_per_service: 50
similarity_threshold: 0.8
include_low_impact: false
```

#### Service Agent Settings (`config/ec2_agent.yaml`)
```yaml
agent_id: ec2_agent
service: EC2
enabled: true

capability:
  supported_recommendation_types:
    - rightsizing
    - purchasing_option
    - idle_resource
  
  required_metrics:
    - cpu_utilization_p50
    - cpu_utilization_p95
    - memory_utilization_p50
  
  optional_metrics:
    - network_in
    - network_out
  
  thresholds:
    cpu_idle_threshold: 5.0
    cpu_low_threshold: 20.0
    memory_low_threshold: 30.0
    uptime_threshold: 0.95

# LLM settings
base_prompt: "You are an expert AWS cost optimization specialist..."
service_specific_prompt: "Analyze EC2 instances for cost optimization..."
max_tokens: 2000
confidence_threshold: 0.7
min_cost_threshold: 1.0
```

## 🚀 Usage Examples

### Basic Analysis
```bash
# Install and activate
pip install -e .
source .venv/bin/activate

# Run with sample data
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --sample-data \
  --log-format human

# Run with real data
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --billing-file data/billing/costs.csv \
  --inventory-file data/inventory/resources.json \
  --metrics-file data/metrics/performance.csv \
  --output-file report.json
```

### Advanced Options
```bash
# Verbose debugging
python -m llm_cost_recommendation --account-id test --sample-data --verbose

# Quiet mode (warnings only)
python -m llm_cost_recommendation --account-id test --sample-data --quiet

# JSON logging for automation
python -m llm_cost_recommendation --status --log-format json
```

## 🔄 System Workflow

### 1. Initialization Phase
```python
# Load configuration
config_manager = ConfigManager("config")
coordinator = CoordinatorAgent(config_manager, llm_service)

# Initialize service agents
agents = {
    "EC2": EC2Agent(config_manager.get_agent_config("EC2")),
    "S3": S3Agent(config_manager.get_agent_config("S3")),
    # ... more agents
}
```

### 2. Data Processing Phase
```python
# Ingest data from various sources
resources = data_service.ingest_inventory_data(inventory_file)
metrics = data_service.ingest_metrics_data(metrics_file)
billing = data_service.ingest_billing_data(billing_file)
```

### 3. Analysis Phase
```python
# Coordinator orchestrates analysis
for service in enabled_services:
    agent = agents[service]
    service_resources = filter_resources_by_service(resources, service)
    
    # Agent analyzes resources using LLM
    recommendations = await agent.analyze_resources(
        resources=service_resources,
        metrics=metrics,
        billing=billing
    )
```

### 4. Aggregation Phase
```python
# Combine recommendations from all agents
all_recommendations = []
for agent_recs in agent_recommendations.values():
    all_recommendations.extend(agent_recs)

# Remove duplicates and rank by priority
final_recommendations = deduplicate_and_rank(all_recommendations)
```

## 🛠️ Extending the System

### Adding New AWS Services

#### 1. Create Service Agent Configuration
Create `config/newservice_agent.yaml`:
```yaml
agent_id: newservice_agent
service: NEWSERVICE
enabled: true

capability:
  supported_recommendation_types:
    - rightsizing
    - custom_optimization
  
  required_metrics:
    - utilization_metric
    - performance_metric
  
  thresholds:
    utilization_threshold: 80.0

base_prompt: "You are an expert in NEWSERVICE optimization..."
```

#### 2. Add Service to Models
Update `models/__init__.py`:
```python
class ServiceType(str, Enum):
    # ... existing services
    NEWSERVICE = "NEWSERVICE"
```

#### 3. Create Specialized Agent Class
Create agent in `agents/base.py`:
```python
class NewServiceAgent(BaseAgent):
    """Agent for NEWSERVICE cost optimization"""
    
    def __init__(self, config: ServiceAgentConfig, llm_service: LLMService):
        super().__init__(config, llm_service)
    
    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze a single resource and generate recommendations"""
        recommendations = []
        
        if not self._validate_resource_data(resource):
            return recommendations
            
        # Apply custom rules
        rule_results = self._apply_custom_rules(resource, metrics, billing_data)
        
        # Merge thresholds with custom overrides
        merged_thresholds = self._merge_thresholds(
            self.config.capability.thresholds,
            rule_results.get("threshold_overrides", {})
        )
        
        # Apply service-specific analysis logic
        context_data = self._prepare_context_data(resource, metrics, billing_data)
        context_data["thresholds"] = merged_thresholds
        
        # Generate LLM recommendations with custom rule context
        llm_recommendations = await self._generate_recommendations_from_llm(
            context_data, rule_results
        )
        
        # Convert to recommendation models
        for llm_rec in llm_recommendations:
            rec = self._convert_llm_recommendation_to_model(llm_rec, resource)
            if rec:
                recommendations.append(rec)
        
        return recommendations
        billing_data: Dict[str, List[BillingData]],
        **kwargs
    ) -> List[Recommendation]:
        """Custom analysis logic for NEWSERVICE"""
        recommendations = []
        
        for resource in resources:
            # Custom analysis logic
            if self._should_optimize(resource, metrics_data.get(resource.resource_id)):
                rec = await self._generate_recommendation(resource, metrics_data, billing_data)
                recommendations.append(rec)
        
        return recommendations
    
    def _should_optimize(self, resource: Resource, metrics: Metrics) -> bool:
        """Service-specific optimization criteria"""
        # Implement your logic
        return True
    
    async def _generate_recommendation(self, resource, metrics, billing) -> Recommendation:
        """Generate LLM-powered recommendation"""
        # Use LLM service to generate recommendation
        pass
```

#### 4. Register Agent in Coordinator
Update `agents/coordinator.py`:
```python
def _create_agents(self) -> Dict[ServiceType, BaseAgent]:
    agents = {}
    
    for service in self.enabled_services:
        if service == ServiceType.NEWSERVICE:
            agents[service] = NewServiceAgent(
                self.config_manager.get_agent_config(service)
            )
        # ... existing services
    
    return agents
```

#### 5. Update Configuration
Add to `config/coordinator.yaml`:
```yaml
enabled_services:
  - EC2
  - S3
  # ... existing services
  - NEWSERVICE
```

### Adding New Cloud Providers

#### 1. Create Provider-Specific Models
```python
# models/azure.py
class AzureServiceType(str, Enum):
    VIRTUAL_MACHINES = "VIRTUAL_MACHINES"
    STORAGE_ACCOUNTS = "STORAGE_ACCOUNTS"
    SQL_DATABASE = "SQL_DATABASE"

class AzureResource(Resource):
    """Azure-specific resource model"""
    subscription_id: str
    resource_group: str
    location: str
```

#### 2. Create Provider Interface
```python
# services/providers.py
from abc import ABC, abstractmethod

class CloudProvider(ABC):
    """Abstract base class for cloud providers"""
    
    @abstractmethod
    async def get_resources(self, subscription_id: str) -> List[Resource]:
        pass
    
    @abstractmethod
    async def get_metrics(self, resource_id: str) -> Metrics:
        pass

class AzureProvider(CloudProvider):
    """Azure cloud provider implementation"""
    
    async def get_resources(self, subscription_id: str) -> List[Resource]:
        # Implement Azure API calls
        pass
```

#### 3. Create Provider-Specific Agents
```python
# agents/azure.py
class AzureVMAgent(BaseAgent):
    """Azure Virtual Machine optimization agent"""
    
    def __init__(self, config: ServiceAgentConfig):
        super().__init__(config, AzureServiceType.VIRTUAL_MACHINES)
```

#### 4. Update Configuration System
```python
# services/config.py
class ProviderConfig(BaseModel):
    name: str  # "aws", "azure", "gcp"
    enabled: bool
    credentials: Dict[str, str]
    services: List[str]

class ConfigManager:
    def __init__(self, config_dir: str):
        self.providers = self._load_provider_configs()
    
    def _load_provider_configs(self) -> Dict[str, ProviderConfig]:
        # Load provider-specific configurations
        pass
```

## 📊 Data Flow & Integration

### Input Data Formats

#### Billing Data (CSV)
```csv
resource_id,service,cost,date,region
i-1234567890abcdef0,EC2,45.50,2025-01-01,us-east-1
vol-0123456789abcdef0,EBS,12.30,2025-01-01,us-east-1
```

#### Inventory Data (JSON)
```json
[
  {
    "resource_id": "i-1234567890abcdef0",
    "service": "EC2",
    "region": "us-east-1",
    "instance_type": "t3.medium",
    "configuration": {
      "vcpus": 2,
      "memory_gb": 4,
      "storage": [{"type": "gp3", "size": 20}]
    },
    "tags": {
      "Environment": "production",
      "Owner": "team-alpha"
    }
  }
]
```

#### Metrics Data (CSV)
```csv
resource_id,metric_name,value,timestamp
i-1234567890abcdef0,cpu_utilization_p50,15.2,2025-01-01T00:00:00Z
i-1234567890abcdef0,memory_utilization_p50,45.8,2025-01-01T00:00:00Z
```

## 🎛️ Advanced Configuration

### LLM Provider Switching
```python
# services/llm.py
class LLMService:
    def __init__(self, config: LLMConfig):
        if config.provider == "openai":
            self.client = ChatOpenAI(model=config.model)
        elif config.provider == "anthropic":
            self.client = ChatAnthropic(model=config.model)
        # Add more providers as needed
```

### Custom Recommendation Types
```python
class RecommendationType(str, Enum):
    # Standard types
    RIGHTSIZING = "rightsizing"
    PURCHASING_OPTION = "purchasing_option"
    
    # Custom types
    SECURITY_OPTIMIZATION = "security_optimization"
    PERFORMANCE_TUNING = "performance_tuning"
    COST_ALLOCATION = "cost_allocation"
```

### Agent Customization
```python
class CustomEC2Agent(EC2Agent):
    """Custom EC2 agent with organization-specific logic"""
    
    def _should_recommend_spot(self, resource: Resource) -> bool:
        """Custom logic for Spot instance recommendations"""
        # Organization-specific rules
        if resource.tags.get("Environment") == "production":
            return False  # Never recommend Spot for production
        return super()._should_recommend_spot(resource)
```

## 🔧 Development & Debugging

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=llm_cost_recommendation tests/
```

### Debugging Tips
```bash
# Verbose logging with human-readable output
python -m llm_cost_recommendation --verbose --log-format human --sample-data --account-id debug

# JSON logging for parsing
python -m llm_cost_recommendation --log-format json --sample-data --account-id debug | jq

# Test specific service
python -c "
from llm_cost_recommendation.agents.base import EC2Agent
from llm_cost_recommendation.services.config import ConfigManager
config = ConfigManager('config')
agent = EC2Agent(config.get_agent_config('EC2'))
print(agent.get_agent_status())
"
```

This architecture provides a robust, extensible foundation for multi-cloud cost optimization using LLM intelligence!
