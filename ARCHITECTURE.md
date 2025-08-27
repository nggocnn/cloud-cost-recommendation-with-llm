# LLM Cost Recommendation System - Architecture & Usage Guide

## Architecture Overview

This is a **multi-agent LLM-powered multi-cloud cost optimization system** that analyzes cloud resources across AWS, Azure, and Google Cloud Platform to provide actionable cost reduction recommendations.

### Core Components

```text
llm_cost_recommendation/
├── agents/                  # Multi-agent system
│   ├── coordinator.py       # Orchestrates analysis across all cloud providers
│   └── base.py             # Service-specific agents (EC2, VMs, Compute Engine, etc.)
├── models/                 # Data models & schemas
│   ├── __init__.py         # Core models and enums
│   └── *.py               # Provider-specific models
├── services/              # Core services
│   ├── config.py          # Configuration management
│   ├── llm.py             # LangChain/OpenAI integration
│   ├── ingestion.py       # Data ingestion & processing
│   └── logging.py         # Enhanced logging system
├── cli.py                 # Command-line interface with export options
├── console.py             # Console output formatting
└── __main__.py           # Module entry point
```

## System Flow

### 1. Data Ingestion

```text
Multi-Cloud Data Sources → DataIngestionService → Structured Models
```

- **Billing Data**: CSV files with cost information from any cloud provider
- **Inventory Data**: JSON files with resource configurations
- **Metrics Data**: CSV files with performance metrics
- **Export Formats**: JSON (detailed), CSV (summary), Excel (multi-sheet)

### 2. Multi-Agent Analysis

```text
Coordinator Agent → Cloud Provider Agents → Service Agents → LLM Analysis → Recommendations
```

- **Coordinator**: Orchestrates the analysis workflow across all providers
- **Provider Agents**: AWS, Azure, GCP-specific logic
- **Service Agents**: Specialized agents for each service (EC2/VMs/Compute, Storage, Databases, etc.)
- **LLM Integration**: Uses OpenAI GPT models for intelligent analysis

### 3. Report Generation & Export

```text
Recommendations → Aggregation → Risk Assessment → Multi-Format Export
```

- **JSON**: Complete detailed report with all metadata
- **CSV**: Summary table for spreadsheet analysis
- **Excel**: Multi-sheet workbook with comprehensive data

## Configuration System

### Environment Configuration (`.env`)

```bash
# LLM Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

### Multi-Cloud Service Configuration (`config/*.yaml`)

#### Coordinator Settings (`config/coordinator.yaml`)

```yaml
enabled_services:
  # AWS Services
  - EC2
  - EBS
  - S3
  - RDS
  - Lambda
  - ALB
  - CloudFront
  - DynamoDB
  
  # Azure Services  (not work yet)
  - VirtualMachines
  - ManagedDisks
  - StorageAccounts
  - SQLDatabase
  - AzureFunctions
  
  # GCP Services (not work yet)
  - ComputeEngine
  - PersistentDisks
  - CloudStorage
  - CloudSQL
  - CloudFunctions

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

#### AWS Service Agent Settings (`config/aws.ec2_agent.yaml`)

```yaml
agent_id: ec2_agent
base_prompt: You are an expert AWS cost optimization specialist focusing on EC2 instances.
capability:
  analysis_window_days: 30
  optional_metrics:
  - network_in
  - network_out
  required_metrics:
  - cpu_utilization_p50
  - cpu_utilization_p95
  - memory_utilization_p50
  service: AWS.EC2
  supported_recommendation_types:
  - rightsizing
  - purchasing_option
  - idle_resource
  thresholds:
    cpu_idle_threshold: 5.0
    cpu_low_threshold: 20.0
    memory_low_threshold: 30.0
    uptime_threshold: 0.95
confidence_threshold: 0.7
enabled: true
max_tokens: 2000
min_cost_threshold: 1.0
service: AWS.EC2
service_specific_prompt: '

  Analyze EC2 instances for cost optimization opportunities. Consider:

  1. CPU and memory utilization patterns

  2. Instance family and generation efficiency

  3. Purchase options (On-Demand vs Reserved vs Spot)

  4. Idle or underutilized instances

  5. Right-sizing opportunities based on actual usage


  Provide specific recommendations with exact instance types and cost calculations.

  '
temperature: 0.1
```

## Usage Examples

### Basic Analysis

```bash
# Install and activate
pip install -e .
source .venv/bin/activate

# Run with sample data  
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --sample-data

# Run with real data
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --billing-file data/billing/sample_billing.csv \
  --inventory-file data/inventory/sample_inventory.json \
  --metrics-file data/metrics/sample_metrics.csv \
  --output-file report.json

# Account ID is provided as a placeholder for AWS Account ID in case of multiple account data is provided (will be remove to support multiple cloud)
```

### Export Format Options

```bash
# Generate detailed JSON report (default)
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --sample-data \
  --output-format json \
  --output-file detailed_report.json

# Generate CSV summary table
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --sample-data \
  --output-format csv \
  --output-file summary.csv

# Generate Excel workbook with multiple sheets
python -m llm_cost_recommendation \
  --account-id "123456789012" \
  --sample-data \
  --output-format excel \
  --output-file comprehensive_report.xlsx
```

### Advanced Options

```bash
# Verbose debugging
python -m llm_cost_recommendation --account-id test --sample-data --verbose

# Check system status
python -m llm_cost_recommendation --status
```

## System Workflow

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

## Extending the System

### Adding New Cloud Services

#### 1. Create Service Agent Configuration

Create `config/{provider}.{service}_agent.yaml`:

```yaml
# Example: config/aws.newservice_agent.yaml
agent_id: aws.newservice_agent
service: NEWSERVICE
cloud_provider: AWS
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

base_prompt: "You are an expert AWS NEWSERVICE optimization specialist..."
service_specific_prompt: "Analyze NEWSERVICE resources for cost optimization..."
```

#### 2. Add Service to Models

Update `llm_cost_recommendation/models/__init__.py`:

```python
class ServiceType(str, Enum):
    # AWS Services
    EC2 = "EC2"
    EBS = "EBS"
    S3 = "S3"
    RDS = "RDS"
    LAMBDA = "Lambda"
    # ... existing services
    NEWSERVICE = "NEWSERVICE"  # Add new service
    
    # Azure Services
    VIRTUAL_MACHINES = "VirtualMachines"
    # ... existing Azure services
    
    # GCP Services  
    COMPUTE_ENGINE = "ComputeEngine"
    # ... existing GCP services

class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"
```

#### 3. Update Coordinator Configuration

Add to `config/coordinator.yaml`:

```yaml
enabled_services:
  # AWS Services
  - EC2
  - S3
  # ... existing services
  - NEWSERVICE  # Add new service
  
  # Azure Services
  - VirtualMachines
  # ... existing Azure services
  
  # GCP Services
  - ComputeEngine
  # ... existing GCP services
```

### Adding New Cloud Providers

#### 1. Create Provider-Specific Configuration Files

Create configuration files with provider prefix:

```bash
# Create new provider configurations
touch config/newprovider.compute_agent.yaml
touch config/newprovider.storage_agent.yaml
touch config/newprovider.database_agent.yaml
```

#### 2. Create Provider-Specific Models

```python
# llm_cost_recommendation/models/newprovider.py
from enum import Enum
from .base import Resource

class NewProviderServiceType(str, Enum):
    COMPUTE = "Compute"
    STORAGE = "Storage"
    DATABASE = "Database"

class NewProviderResource(Resource):
    """New provider-specific resource model"""
    subscription_id: str
    resource_group: str
    location: str
    provider_specific_field: str
```

#### 3. Update Core Models

```python
# llm_cost_recommendation/models/__init__.py
class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"
    NEWPROVIDER = "NewProvider"  # Add new provider
```

#### 4. Create Provider-Specific Agent Classes

```python
# llm_cost_recommendation/agents/newprovider.py
from .base import BaseAgent
from ..models.newprovider import NewProviderServiceType

class NewProviderComputeAgent(BaseAgent):
    """New provider compute optimization agent"""
    
    def __init__(self, config: ServiceAgentConfig, llm_service: LLMService):
        super().__init__(config, llm_service)
        self.provider = CloudProvider.NEWPROVIDER
        self.service_type = NewProviderServiceType.COMPUTE
## Data Flow & Integration

### Input Data Formats

#### Billing Data (CSV)

```csv
resource_id,service,cost,date,region,cloud_provider
i-1234567890abcdef0,EC2,45.50,2025-01-01,us-east-1,AWS
vm-abcd1234,VirtualMachines,38.20,2025-01-01,eastus,Azure
instance-xyz789,ComputeEngine,42.10,2025-01-01,us-central1-a,GCP
```

#### Inventory Data (JSON)

```json
[
  {
    "resource_id": "i-1234567890abcdef0",
    "service": "EC2", 
    "cloud_provider": "AWS",
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
  },
  {
    "resource_id": "vm-abcd1234",
    "service": "VirtualMachines",
    "cloud_provider": "Azure", 
    "region": "eastus",
    "vm_size": "Standard_B2s",
    "configuration": {
      "vcpus": 2,
      "memory_gb": 4,
      "os_disk": {"type": "Premium_LRS", "size": 30}
    },
    "tags": {
      "Environment": "development",
      "Team": "backend"
    }
  }
]
```

#### Metrics Data (CSV)

```csv
resource_id,metric_name,value,timestamp,cloud_provider
i-1234567890abcdef0,cpu_utilization_p50,15.2,2025-01-01T00:00:00Z,AWS
i-1234567890abcdef0,memory_utilization_p50,45.8,2025-01-01T00:00:00Z,AWS
vm-abcd1234,cpu_utilization_p50,12.5,2025-01-01T00:00:00Z,Azure
vm-abcd1234,memory_utilization_p50,38.2,2025-01-01T00:00:00Z,Azure
```

## Advanced Configuration

### Multi-Cloud Agent Configuration

The system automatically loads agent configurations based on naming conventions:

```text
config/
├── aws.ec2_agent.yaml          # AWS EC2 agent
├── aws.s3_agent.yaml           # AWS S3 agent
├── azure.vm_agent.yaml         # Azure VM agent
├── azure.storage_agent.yaml    # Azure Storage agent
├── gcp.compute_agent.yaml      # GCP Compute agent
├── gcp.storage_agent.yaml      # GCP Storage agent
└── coordinator.yaml            # Global coordination settings
```

### Export Format Configuration

```python
# llm_cost_recommendation/cli.py
class ExportFormat(str, Enum):
    JSON = "json"      # Detailed report with all metadata
    CSV = "csv"        # Summary table for spreadsheet analysis  
    EXCEL = "excel"    # Multi-sheet workbook with comprehensive data

# Usage examples:
# --output-format json    # Default: detailed JSON report
# --output-format csv     # Summary CSV table  
# --output-format excel   # Excel workbook with multiple sheets
```

### Custom Recommendation Types

```python
class RecommendationType(str, Enum):
    # Standard optimization types
    RIGHTSIZING = "rightsizing"
    PURCHASING_OPTION = "purchasing_option" 
    IDLE_RESOURCE = "idle_resource"
    STORAGE_CLASS = "storage_class"
    
    # Cloud-specific types
    RESERVED_INSTANCES = "reserved_instances"      # AWS/Azure
    SPOT_INSTANCES = "spot_instances"              # AWS
    PREEMPTIBLE_INSTANCES = "preemptible_instances" # GCP
    COMMITTED_USE_DISCOUNTS = "committed_use_discounts" # GCP
```

## Development & Debugging

### Running Tests

```bash
# Install development dependencies
pip install -e .

# Run with sample data for testing
python -m llm_cost_recommendation --account-id test --sample-data

# Test specific export formats
python -m llm_cost_recommendation --account-id test --sample-data --output-format csv
python -m llm_cost_recommendation --account-id test --sample-data --output-format excel
```

### Debugging Multi-Cloud Analysis

```bash
# Verbose logging for debugging
python -m llm_cost_recommendation --verbose --sample-data --account-id debug

# Check system status and configuration
python -m llm_cost_recommendation --status

# Test with real data
python -m llm_cost_recommendation \
  --account-id "production-123" \
  --billing-file data/billing/sample_billing.csv \
  --inventory-file data/inventory/sample_inventory.json \
  --metrics-file data/metrics/sample_metrics.csv \
  --output-format excel \
  --output-file analysis_report.xlsx
```

### Configuration Validation

```python
# Test agent configuration loading
from llm_cost_recommendation.services.config import ConfigManager

config = ConfigManager('config')

# List all available agents
print("Available agents:")
for agent_config in config.agent_configs.values():
    print(f"- {agent_config.agent_id} ({agent_config.cloud_provider}.{agent_config.service})")

# Test specific agent
ec2_config = config.get_agent_config('aws.ec2_agent')
print(f"EC2 Agent: {ec2_config.service_specific_prompt[:100]}...")
```

This multi-cloud architecture provides a robust, extensible foundation for cost optimization across AWS, Azure, and Google Cloud Platform using LLM intelligence!
