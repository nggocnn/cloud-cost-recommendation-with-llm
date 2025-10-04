# LLM Cost Recommendation System - Architecture & Usage Guide

## Architecture Overview

This is a **multi-agent LLM-powered multi-cloud cost optimization system** that analyzes cloud resources across AWS, Azure, and Google Cloud Platform to provide actionable cost reduction recommendations through both CLI and REST API interfaces.

### Recent Enhancements

- **Comprehensive Test Suite**: 68+ tests covering API security, performance, validation, and edge cases
- **FastAPI REST API**: Full web API with health monitoring, async processing, and OpenAPI documentation
- **Enhanced Security**: Rate limiting, input validation, and secure error handling
- **Performance Optimizations**: Async processing, concurrent request handling, and memory management
- **Improved Agent System**: Enhanced processors, managers, and report generators

### Core Components

```text
llm_cost_recommendation/
├── agents/                  # Enhanced multi-agent system
│   ├── coordinator.py       # Orchestrates analysis across all cloud providers
│   ├── base.py             # Service-specific agents (EC2, VMs, Compute Engine, etc.)
│   ├── agent_manager.py    # Agent lifecycle management
│   ├── recommendation_processor.py  # Recommendation processing engine
│   ├── report_generator.py # Advanced report generation
│   └── resource_processor.py       # Resource analysis processor
├── models/                 # Comprehensive data models & schemas
│   ├── types.py            # Core enums and types
│   ├── resources.py        # Resource models
│   ├── recommendations.py  # Recommendation models
│   ├── agents.py           # Agent configuration models
│   └── api_models.py       # API request/response models
├── services/              # Enhanced core services
│   ├── config.py          # Configuration management
│   ├── llm.py             # OpenAI integration with timeout handling
│   ├── ingestion.py       # Data ingestion & processing
│   ├── conditions.py      # Custom rules processing
│   ├── data_validation.py # Comprehensive validation service
│   ├── prompts.py         # LLM prompt management
│   └── logging.py         # Structured logging system
├── api.py                 # FastAPI REST API application
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

### 1.1. Testing Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Comprehensive Test Suite (68+ Tests)         │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  API Security   │   Performance   │   Validation    │ Edge Cases│
│     (16)        │      (9)        │      (14)       │    (15)   │
│                 │                 │                 │           │
│ • Rate limiting │ • Memory leaks  │ • Schema valid. │ • Network │
│ • Input valid.  │ • Bottlenecks   │ • Data formats  │ • Timeouts│
│ • Auth checks   │ • Concurrency   │ • Error msgs    │ • Recovery│
│ • Size limits   │ • Async ops     │ • Type safety   │ • Cleanup │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
            │                │               │              │
            └────────────────┼───────────────┼──────────────┘
                             │               │
        ┌───────────────────┼───────────────┼────────────────┐
        │                   │               │                │
        ▼                   ▼               ▼                ▼
   ┌─────────┐        ┌─────────┐   ┌──────────┐      ┌──────────┐
   │Security │        │   E2E   │   │Integration│      │   Unit   │
   │  (11)   │        │  Tests  │   │   Tests   │      │  Tests   │
   └─────────┘        └─────────┘   └──────────┘      └──────────┘
```

### 2. Multi-Interface Processing

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │    │   CLI Interface  │    │  Test Suite     │
│                 │    │                  │    │                 │
│ • FastAPI       │    │ • Argparse      │    │ • 68+ tests     │
│ • Async         │◄──▶│ • Rich output   │◄──▶│ • Full coverage │
│ • Health checks │    │ • Export formats │    │ • Security      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │    Multi-Agent Analysis  │
                    └──────────────────────────┘
```

### 3. Enhanced Multi-Agent Analysis

```text
Coordinator Agent → Cloud Provider Agents → Service Agents → LLM Analysis → Recommendations
```

- **Coordinator**: Orchestrates the analysis workflow across all providers with enhanced error handling
- **Agent Manager**: Manages agent lifecycle, configuration loading, and health monitoring
- **Resource Processor**: Advanced resource analysis with performance optimization
- **Recommendation Processor**: Intelligent recommendation consolidation and ranking
- **Report Generator**: Multi-format report generation (JSON, CSV, Excel)
- **Provider Agents**: AWS, Azure, GCP-specific logic with enhanced validation
- **Service Agents**: 36 specialized agents for each service with custom rules support
- **LLM Integration**: OpenAI GPT models with timeout handling and error recovery

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

#### Global Coordinator Settings (`config/global/coordinator.yaml`)

```yaml
enabled_services:
  # AWS Services (17 implemented)
  - AWS.EC2
  - AWS.EBS
  - AWS.S3
  - AWS.RDS
  - AWS.Lambda
  - AWS.ALB
  - AWS.NLB
  - AWS.GWLB
  - AWS.CloudFront
  - AWS.DynamoDB
  - AWS.SNS
  - AWS.SQS
  - AWS.NAT_Gateway
  - AWS.Elastic_IP
  - AWS.VPC_Endpoints
  - AWS.EFS
  - AWS.RDS_Snapshots
  
  # Azure Services (10 implemented)
  - Azure.VirtualMachines
  - Azure.Storage
  - Azure.Disk
  - Azure.SQL
  - Azure.Functions
  - Azure.LoadBalancer
  - Azure.NAT_Gateway
  - Azure.PublicIP
  - Azure.CDN
  - Azure.Cosmos
  
  # GCP Services (8 implemented)
  - GCP.Compute
  - GCP.Storage
  - GCP.Disk
  - GCP.SQL
  - GCP.Functions
  - GCP.LoadBalancer
  - GCP.CDN
  - GCP.Firestore

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

#### AWS Service Agent Configuration (`config/agents/aws/ec2.yaml`)

```yaml
agent_id: aws.ec2_agent
service: AWS.EC2
enabled: true
base_prompt: You are an expert AWS cost optimization specialist focusing on EC2 instances.

capability:
  analysis_window_days: 30
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

confidence_threshold: 0.7
min_cost_threshold: 1.0
max_tokens: 2000
temperature: 0.1

service_specific_prompt: |
  Analyze EC2 instances for cost optimization opportunities. Consider:
  1. CPU and memory utilization patterns
  2. Instance family and generation efficiency
  3. Purchase options (On-Demand vs Reserved vs Spot)
  4. Idle or underutilized instances
  5. Right-sizing opportunities based on actual usage
  
  Provide specific recommendations with exact instance types and cost calculations.

# Custom conditional rules for dynamic behavior
custom_rules:
  - name: "production_cpu_buffer"
    description: "Production instances need higher CPU buffer for reliability"
    conditions:
      - field: "tag.Environment"
        operator: "equals"
        value: "production"
    threshold_overrides:
      cpu_low_threshold: 30.0
    enabled: true
```

## Usage Examples

### Basic Analysis

```bash
# Install and activate
pip install -e .
source .venv/bin/activate

# Run with sample data  
python -m llm_cost_recommendation \
  --sample-data

# Run with real data
python -m llm_cost_recommendation \
  --billing-file data/billing/sample_billing.csv \
  --inventory-file data/inventory/sample_inventory.json \
  --metrics-file data/metrics/sample_metrics.csv \
  --output-file report.json
```

### Export Format Options

```bash
# Generate detailed JSON report (default)
python -m llm_cost_recommendation \
  --sample-data \
  --output-format json \
  --output-file detailed_report.json

# Generate CSV summary table
python -m llm_cost_recommendation \
  --sample-data \
  --output-format csv \
  --output-file summary.csv

# Generate Excel workbook with multiple sheets
python -m llm_cost_recommendation \
  --sample-data \
  --output-format excel \
  --output-file comprehensive_report.xlsx
```

### Advanced Options

```bash
# Verbose debugging
python -m llm_cost_recommendation --sample-data --verbose

# Check system status
python -m llm_cost_recommendation --status
```

## System Workflow

### 1. Initialization Phase

```python
# Load configuration and initialize components
config_manager = ConfigManager("config")
llm_service = LLMService(config_manager.llm_config)
coordinator = CoordinatorAgent(config_manager, llm_service)

# Coordinator automatically initializes service agents based on configuration
# Each agent is configured via YAML files in config/agents/{provider}/{service}.yaml
```

### 2. Data Processing Phase

```python
# Ingest data from various sources
data_service = DataIngestionService("data")

resources = data_service.ingest_inventory_data(inventory_file)  # JSON format
metrics = data_service.ingest_metrics_data(metrics_file)        # CSV format
billing = data_service.ingest_billing_data(billing_file)        # CSV format

# Data is automatically validated and converted to internal models
```

### 3. Analysis Phase (Batch Mode)

```python
# Coordinator orchestrates analysis across all service agents
report = await coordinator.analyze_resources_and_generate_report(
    resources=resources,
    metrics_data=metrics_by_resource_id,
    billing_data=billing_by_resource_id,
    batch_mode=True  # Efficient parallel processing
)

# Process:
# 1. Group resources by service type
# 2. Route to appropriate service agents or default agent
# 3. Apply custom rules and threshold overrides
# 4. Generate LLM-powered recommendations
# 5. Post-process: deduplicate, filter, rank
```

### 4. Aggregation & Report Generation

```python
# Coordinator automatically handles:
# - Deduplication of similar recommendations
# - Risk assessment (Low/Medium/High)
# - Savings calculations and prioritization
# - Coverage analysis (which services used specific vs default agents)
# - Implementation timeline categorization (quick wins, medium-term, long-term)

final_report = RecommendationReport(
    total_monthly_savings=report.total_monthly_savings,
    total_annual_savings=report.total_annual_savings,
    recommendations=processed_recommendations,
    coverage=coverage_metrics,
    risk_distribution=risk_counts,
    savings_by_service=savings_breakdown
)
```

## Extending the System

### Adding New Cloud Services

The system uses a single `ServiceAgent` class that adapts behavior based on YAML configuration. No code changes needed.

#### 1. Create Service Agent Configuration

Create `config/agents/{provider}/{service}.yaml`:

```yaml
# Example: config/agents/aws/newservice.yaml
agent_id: aws.newservice_agent
service: AWS.NEWSERVICE
enabled: true

base_prompt: "You are an expert AWS NEWSERVICE optimization specialist..."
service_specific_prompt: |
  Analyze NEWSERVICE resources for cost optimization opportunities:
  1. Service-specific optimization patterns
  2. Cost reduction strategies
  3. Performance vs cost trade-offs

capability:
  supported_recommendation_types:
    - rightsizing
    - purchasing_option
    - custom_optimization
  required_metrics:
    - utilization_metric
    - performance_metric
  thresholds:
    utilization_threshold: 80.0

confidence_threshold: 0.7
min_cost_threshold: 1.0
max_tokens: 2000
temperature: 0.1
```

#### 2. Add Service to Models

Update `llm_cost_recommendation/models/types.py`:

```python
class ServiceType(str, Enum):
    # AWS Services
    EC2 = "AWS.EC2"
    EBS = "AWS.EBS"
    S3 = "AWS.S3"
    # ... existing services
    NEWSERVICE = "AWS.NEWSERVICE"  # Add new service
```

#### 3. Update Coordinator Configuration

Add to `config/global/coordinator.yaml`:

```yaml
enabled_services:
  # AWS Services
  - AWS.EC2
  - AWS.S3
  # ... existing services
  - AWS.NEWSERVICE  # Add new service
```

### Adding New Cloud Providers

#### 1. Create Provider-Specific Service Types

Update `llm_cost_recommendation/models/types.py`:

```python
class ServiceType(str, Enum):
    # AWS Services
    EC2 = "AWS.EC2"
    # ... existing AWS services
    
    # Azure Services  
    VIRTUAL_MACHINES = "Azure.VirtualMachines"
    # ... existing Azure services
    
    # GCP Services
    COMPUTE_ENGINE = "GCP.Compute"
    # ... existing GCP services
    
    # New Provider Services
    NEWPROVIDER_COMPUTE = "NewProvider.Compute"
    NEWPROVIDER_STORAGE = "NewProvider.Storage"

class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"
    NEWPROVIDER = "NewProvider"  # Add new provider
```

#### 2. Create Provider-Specific Configuration Directory

```bash
# Create new provider configuration structure
mkdir -p config/agents/newprovider
touch config/agents/newprovider/compute.yaml
touch config/agents/newprovider/storage.yaml
touch config/agents/newprovider/database.yaml
```

#### 3. Configure Provider Services

```yaml
# config/agents/newprovider/compute.yaml
agent_id: newprovider.compute_agent
service: NewProvider.Compute
enabled: true

base_prompt: "You are an expert NewProvider cost optimization specialist..."
service_specific_prompt: |
  Analyze NewProvider compute resources for optimization:
  1. Instance rightsizing opportunities
  2. Reserved capacity options
  3. Regional pricing differences

capability:
  supported_recommendation_types:
    - rightsizing
    - purchasing_option
  required_metrics:
    - cpu_utilization_p50
    - memory_utilization_p50
  thresholds:
    cpu_low_threshold: 20.0
    memory_low_threshold: 30.0

confidence_threshold: 0.7
min_cost_threshold: 1.0
max_tokens: 2000
temperature: 0.1
```

### Advanced Configuration Features

#### Custom Conditional Rules

The system supports dynamic threshold adjustment based on resource characteristics:

```yaml
# config/agents/aws/ec2.yaml
custom_rules:
  - name: "production_safety_buffer"
    description: "Production resources need higher thresholds"
    conditions:
      - field: "tag.Environment"
        operator: "equals"
        value: "production"
    threshold_overrides:
      cpu_low_threshold: 30.0
      memory_low_threshold: 40.0
    enabled: true
    
  - name: "high_cost_focus"
    description: "Focus on high-cost resources"
    conditions:
      - field: "monthly_cost"
        operator: "greater_than"
        value: 100.0
    actions:
      force_recommendation_types:
        - rightsizing
        - purchasing_option
    enabled: true
```

#### Multi-Condition Rules

```yaml
custom_rules:
  - name: "critical_production_workload"
    description: "Critical production workloads need special handling"
    conditions:
      - field: "tag.Environment"
        operator: "equals"
        value: "production"
      - field: "tag.Criticality"
        operator: "equals"
        value: "high"
      - field: "monthly_cost"
        operator: "greater_than"
        value: 500.0
    threshold_overrides:
      cpu_low_threshold: 40.0
      memory_low_threshold: 50.0
    risk_level_override: "MEDIUM"  # Never mark as low risk
    enabled: true
```

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
python -m llm_cost_recommendation --sample-data

# Test specific export formats
python -m llm_cost_recommendation --sample-data --output-format csv
python -m llm_cost_recommendation --sample-data --output-format excel
```

### Debugging Multi-Cloud Analysis

```bash
# Verbose logging for debugging
python -m llm_cost_recommendation --verbose --sample-data

# Check system status and configuration
python -m llm_cost_recommendation --status

# Test with real data
python -m llm_cost_recommendation \
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
