# LLM Cost Recommendation System

A multi-agent system for cloud cost optimization using Large Language Models (LLM). This system analyzes cloud resources, billing data, and performance metrics to provide intelligent cost optimization recommendations.

## Features

- **Multi-Agent Architecture**: Coordinator agent orchestrates 36 service-specific agents (AWS: 17, Azure: 10, GCP: 8, Default: 1)
- **LLM-Powered Analysis**: Uses OpenAI GPT-4 for intelligent cost optimization recommendations
- **Configuration-Driven**: Single ServiceAgent class adapts behavior via YAML configs - add new services without code changes
- **Custom Rules Engine**: Dynamic threshold adjustment based on resource tags, costs, and metrics
- **Multi-Cloud Support**: AWS, Azure, and GCP with extensible architecture for additional providers
- **Multiple Export Formats**: JSON (detailed), CSV (summary), and Excel (multi-sheet) output formats
- **Comprehensive Analysis**: Analyzes rightsizing, purchasing options, storage classes, lifecycle policies, and idle resources
- **Risk Assessment**: Provides Low/Medium/High risk classifications with implementation guidance
- **Cost Calculations**: Exact cost calculations with monthly and annual savings estimates
- **Data Ingestion**: Supports CSV billing data, JSON inventory, and CSV metrics with validation
- **Sample Data Generation**: Built-in sample data for testing and demonstration
- **Intelligent Fallback**: Default agent handles unsupported services to ensure coverage
- **Batch Processing**: Parallel processing for efficiency with individual analysis fallback

## Architecture

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Coordinator    │    │  LLM Service    │
│                 │    │     Agent        │    │   (OpenAI)      │
│ • Billing CSV   │───▶│                  │◄──▶│                 │
│ • Inventory JSON│    │ • Routes to      │    │ • GPT-4         │
│ • Metrics CSV   │    │   service agents │    │ • Structured    │
└─────────────────┘    │ • Consolidates   │    │   prompts       │
                       │   recommendations│    └─────────────────┘
                       │ • Deduplicates   │
                       │ • Ranks by value │
                       └──────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼─────┐ ┌───────▼─────┐ ┌───────▼─────┐
        │ AWS Agents  │ │Azure Agents │ │ GCP Agents  │
        │    (17)     │ │    (10)     │ │     (8)     │
        │             │ │             │ │             │
        │ • EC2       │ │ • VMs       │ │ • Compute   │
        │ • EBS       │ │ • Disks     │ │ • Storage   │
        │ • S3        │ │ • Storage   │ │ • Functions │
        │ • RDS       │ │ • SQL       │ │ • SQL       │
        │ • Lambda    │ │ • Functions │ │ • CDN       │
        │ • ALB/NLB   │ │ • LB        │ │ • LB        │
        │ • DynamoDB  │ │ • Cosmos DB │ │ • Firestore │
        │ + 10 more   │ │ + 3 more    │ │ + 1 more    │
        └─────────────┘ └─────────────┘ └─────────────┘
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd llm-cost-recommendation
   ```

2. **Set up Python virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   Copy `.env.example` to `.env` and configure:

   ```bash
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL=gpt-4
   ```

## Quick Start

### Test with Sample Data

```bash
python -m llm_cost_recommendation --sample-data
```

This will:

- Generate sample cloud billing, inventory, and metrics data
- Run the complete analysis pipeline  
- Display a comprehensive cost optimization report

### Analyze Real Data

```bash
python -m llm_cost_recommendation \
  --billing-file data/billing/sample_billing.csv \
  --inventory-file data/inventory/sample_inventory.json \
  --metrics-file data/metrics/sample_metrics.csv \
  --output-file report.json \
  --output-format excel
```

### Export Options

The system supports multiple output formats:

- **JSON** (default): Full detailed report with all recommendation metadata
- **CSV**: Summary table with key metrics for spreadsheet analysis
- **Excel**: Multi-sheet workbook with recommendations, summary, and raw data

```bash
# Generate Excel report with multiple sheets
python -m llm_cost_recommendation --sample-data --output-format excel --output-file report.xlsx

# Generate CSV summary table
python -m llm_cost_recommendation --sample-data --output-format csv --output-file summary.csv
```

### Check System Status

```bash
python -m llm_cost_recommendation --status
```

## Data Formats

### Billing Data (CSV)

Required columns based on AWS Cost and Usage Report:

- `bill/BillingPeriodStartDate`, `bill/BillingPeriodEndDate`
- `lineItem/UsageAccountId`, `product/ProductName`
- `lineItem/ResourceId`, `product/region`
- `lineItem/UsageType`, `lineItem/UsageAmount`, `lineItem/UsageUnit`
- `lineItem/UnblendedCost`, `lineItem/NetAmortizedCost`
- `resourceTags/*` (optional)

### Inventory Data (JSON)

Array of resource objects:

```json
[
  {
    "resource_id": "i-1234567890abcdef0",
    "service": "EC2",
    "region": "us-west-2",
    "account_id": "123456789012",
    "tags": {"Environment": "production"},
    "properties": {
      "instance_type": "m5.large",
      "state": "running",
      "cpu_count": 2,
      "memory_gb": 8
    }
  }
]
```

### Metrics Data (CSV)

Performance metrics by resource:

- `resource_id`, `timestamp`, `period_days`
- `cpu_utilization_p50`, `cpu_utilization_p90`, `cpu_utilization_p95`
- `memory_utilization_p50`, `memory_utilization_p90`, `memory_utilization_p95`
- `iops_read`, `iops_write`, `throughput_read`, `throughput_write`
- `network_in`, `network_out`, `is_idle`

## Configuration

### Service Agents

Each cloud service has its own configuration file in `config/` with cloud provider prefixes:

**AWS Services:**

- `aws.ec2_agent.yaml` - EC2 instances
- `aws.ebs_agent.yaml` - EBS volumes  
- `aws.s3_agent.yaml` - S3 buckets
- `aws.rds_agent.yaml` - RDS databases
- `aws.lambda_agent.yaml` - Lambda functions

**Azure Services:** (not test yet)

- `azure.vm_agent.yaml` - Virtual Machines
- `azure.disk_agent.yaml` - Managed Disks
- `azure.storage_agent.yaml` - Storage Accounts
- `azure.sql_agent.yaml` - SQL Databases
- `azure.functions_agent.yaml` - Azure Functions

**GCP Services:** (not test yet)

### Agent Configurations

The system includes 36 pre-configured agents:

**AWS Agents (17)**:

- `aws/ec2.yaml` - EC2 instances
- `aws/ebs.yaml` - EBS volumes  
- `aws/s3.yaml` - S3 storage
- `aws/rds.yaml` - RDS databases
- `aws/lambda.yaml` - Lambda functions
- `aws/alb.yaml` - Application Load Balancers
- `aws/nlb.yaml` - Network Load Balancers
- `aws/gwlb.yaml` - Gateway Load Balancers
- `aws/cloudfront.yaml` - CloudFront distributions
- `aws/dynamodb.yaml` - DynamoDB tables
- `aws/sns.yaml` - SNS topics
- `aws/sqs.yaml` - SQS queues
- `aws/natgateway.yaml` - NAT Gateways
- `aws/elasticip.yaml` - Elastic IPs
- `aws/vpcendpoints.yaml` - VPC Endpoints
- `aws/efs.yaml` - EFS file systems
- `aws/rds_snapshots.yaml` - RDS snapshots

**Azure Agents (10)**:

- `azure/vm.yaml` - Virtual Machines
- `azure/disk.yaml` - Managed Disks
- `azure/storage.yaml` - Storage Accounts
- `azure/sql.yaml` - SQL Database
- `azure/functions.yaml` - Azure Functions
- `azure/loadbalancer.yaml` - Load Balancer
- `azure/natgateway.yaml` - NAT Gateway
- `azure/publicip.yaml` - Public IP
- `azure/cdn.yaml` - CDN
- `azure/cosmos.yaml` - Cosmos DB

**GCP Agents (8)**:

- `gcp/compute.yaml` - Compute Engine
- `gcp/disk.yaml` - Persistent Disks
- `gcp/storage.yaml` - Cloud Storage
- `gcp/sql.yaml` - Cloud SQL
- `gcp/functions.yaml` - Cloud Functions
- `gcp/loadbalancer.yaml` - Load Balancer
- `gcp/cdn.yaml` - CDN
- `gcp/firestore.yaml` - Firestore

**Default Agent (1)**:

- `default/default.yaml` - Fallback for unsupported services

Example agent configuration:

```yaml
agent_id: aws.ec2_agent
service: AWS.EC2
enabled: true

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

base_prompt: "You are an expert AWS cost optimization specialist focusing on EC2 instances."
service_specific_prompt: |
  Analyze EC2 instances for cost optimization opportunities. Consider:
  1. CPU and memory utilization patterns
  2. Instance family and generation efficiency
  3. Purchase options (On-Demand vs Reserved vs Spot)
  4. Idle or underutilized instances
  5. Right-sizing opportunities based on actual usage

confidence_threshold: 0.7
min_cost_threshold: 1.0
max_tokens: 2000
temperature: 0.1

# Custom conditional rules for dynamic behavior
custom_rules:
  - name: "production_cpu_buffer"
    description: "Production instances need higher CPU buffer"
    conditions:
      - field: "tag.Environment"
        operator: "equals"
        value: "production"
    threshold_overrides:
      cpu_low_threshold: 30.0
    enabled: true
```

### Global Coordinator Configuration

Global settings in `config/global/coordinator.yaml`:

```yaml
enabled_services:
  # AWS Services (17)
  - AWS.EC2
  - AWS.EBS
  - AWS.S3
  - AWS.RDS
  - AWS.Lambda
  # ... all 17 AWS services
  
  # Azure Services (10)  
  - Azure.VirtualMachines
  - Azure.Storage
  # ... all 10 Azure services
  
  # GCP Services (8)
  - GCP.Compute
  - GCP.Storage
  # ... all 8 GCP services

# Recommendation weighting
savings_weight: 0.4
risk_weight: 0.3
confidence_weight: 0.2
implementation_ease_weight: 0.1

# Analysis parameters
max_recommendations_per_service: 50
similarity_threshold: 0.8
include_low_impact: false
```

## Adding New Services

The system uses a configuration-driven architecture - new services require only YAML configuration:

1. **Create agent configuration**:

   ```bash
   # Create new service configuration
   cp config/agents/aws/ec2.yaml config/agents/aws/newservice.yaml
   ```

2. **Update the configuration** with service-specific settings:
   - Service identifier and cloud provider
   - Supported recommendation types
   - Required/optional metrics  
   - Service-specific thresholds
   - LLM prompts for analysis

3. **Add service to models** in `llm_cost_recommendation/models/types.py`:

   ```python
   class ServiceType(str, Enum):
       # AWS Services
       EC2 = "AWS.EC2"
       # ... existing services
       NEWSERVICE = "AWS.NEWSERVICE"  # Add new service
   ```

4. **Enable in coordinator** configuration `config/global/coordinator.yaml`:

   ```yaml
   enabled_services:
     # AWS Services
     - AWS.EC2
     # ... existing services  
     - AWS.NEWSERVICE  # Add new service
   ```

The system automatically discovers and loads the new agent - no additional code changes required!

### Custom Rules

Add conditional logic to agents for dynamic behavior:

```yaml
custom_rules:
  - name: "production_safety_buffer"
    description: "Production resources need higher safety margins"
    conditions:
      - field: "tag.Environment"
        operator: "equals"
        value: "production"
    threshold_overrides:
      cpu_low_threshold: 30.0
      memory_low_threshold: 40.0
    enabled: true
```

## Output Formats

### JSON Report (Detailed)

Comprehensive machine-readable report with:

- Complete recommendation details with implementation steps
- Risk assessments (Low/Medium/High) with justification
- Exact cost calculations (monthly/annual savings)
- Resource metadata and service coverage
- Analysis timing and agent utilization statistics

### CSV Summary (Spreadsheet-Friendly)

Tabular format for analysis with:

- One recommendation per row
- Key fields: service, savings, risk, priority
- Suitable for pivot tables and financial analysis

### Excel Workbook (Multi-Sheet)

Professional report format with:

- Summary sheet with key metrics
- Detailed recommendations by service
- Risk analysis and prioritization
- Implementation roadmap

## System Capabilities

### Current Service Coverage

- **AWS**: 17 services (EC2, S3, RDS, Lambda, EBS, ALB/NLB, DynamoDB, etc.)
- **Azure**: 10 services (VMs, Storage, SQL, Functions, Load Balancer, etc.)
- **GCP**: 8 services (Compute, Storage, SQL, Functions, Load Balancer, etc.)
- **Default Agent**: Handles any unsupported services automatically

### Analysis Features

- **Intelligent Batching**: Parallel processing for large resource sets
- **Dynamic Thresholds**: Custom rules adjust behavior based on resource characteristics
- **Risk Classification**: Low/Medium/High risk with implementation guidance
- **Cost Focus**: Prioritizes high-impact recommendations
- **Coverage Tracking**: Reports which services used specific vs default agents

### Quality Assurance

- **LLM Validation**: Structured output parsing with error handling
- **Data Validation**: Pydantic models ensure data integrity
- **Fallback Mechanisms**: Default agent ensures 100% resource coverage
- **Comprehensive Logging**: Structured logs for debugging and monitoring

## Development

### Project Structure

```text
llm-cost-recommendation/
├── llm_cost_recommendation/      # Main package
│   ├── agents/                  # Agent implementations
│   │   ├── base.py             # ServiceAgent class (single implementation)
│   │   └── coordinator.py      # Coordinator agent
│   ├── models/                  # Data models (Pydantic)
│   │   ├── types.py            # Enums and core types
│   │   ├── resources.py        # Resource models
│   │   ├── recommendations.py  # Recommendation models
│   │   └── agents.py           # Agent configuration models
│   ├── services/                # Core services
│   │   ├── llm.py              # LLM integration (OpenAI)
│   │   ├── config.py           # Configuration management
│   │   ├── ingestion.py        # Data ingestion and parsing
│   │   └── conditions.py       # Custom rules processing
│   ├── utils/                   # Utilities
│   │   └── logging.py          # Structured logging
│   ├── cli.py                  # Command line interface
│   ├── console.py              # Console output formatting
│   └── __main__.py             # Entry point
├── config/                      # Agent configurations (YAML)
│   ├── agents/                 # Service agent configurations
│   │   ├── aws/               # AWS service agents (17)
│   │   ├── azure/             # Azure service agents (10)
│   │   ├── gcp/               # GCP service agents (8)
│   │   └── default/           # Default fallback agent
│   └── global/                # Global configuration
│       └── coordinator.yaml   # Coordinator settings
├── data/                       # Sample and input data
│   ├── billing/               # Billing data files (CSV)
│   ├── inventory/             # Resource inventory files (JSON)
│   └── metrics/               # Performance metrics files (CSV)
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   ├── REQUIREMENT.md         # Technical requirements
│   ├── BASIC_DESIGN.md        # Implementation details
│   └── CUSTOM_CONDITIONS.md   # Custom rules documentation
└── scripts/                    # Utility scripts
```

### Key Implementation Features

### Core Implementation Features

- **Single Agent Class**: `ServiceAgent` adapts behavior via YAML configuration
- **Configuration-Driven**: Add new services without code changes
- **Async Processing**: Efficient parallel analysis with asyncio
- **Error Resilience**: Comprehensive error handling and logging
- **Extensible Design**: Easy to add new providers and services

### Testing

Run with sample data:

```bash
python -m llm_cost_recommendation --sample-data
```

### Installing as Package

```bash
# Install in development mode
pip install -e .

# Run from anywhere
llm-cost-recommendation --sample-data
```

### Logging

Structured JSON logging with rich context:

```json
{
  "event": "LLM recommendations generated",
  "agent_id": "aws.ec2_agent", 
  "recommendations_count": 3,
  "response_time_ms": 1250.5,
  "timestamp": "2025-09-08T10:30:00.000Z",
  "level": "info"
}
```

## Supported Services Summary

### AWS Services (17)

EC2, EBS, S3, RDS, Lambda, ALB, NLB, GWLB, CloudFront, DynamoDB, EFS, Elastic IPs, NAT Gateway, VPC Endpoints, SQS, SNS, RDS Snapshots

### Azure Services (10)

Virtual Machines, Managed Disks, Storage Accounts, SQL Databases, Azure Functions, Load Balancer, Public IPs, NAT Gateway, CDN, Cosmos DB

### Google Cloud Platform (8)

Compute Engine, Persistent Disks, Cloud Storage, Cloud SQL, Cloud Functions, Load Balancer, CDN, Firestore

### Default Agent (1)

Handles any unsupported services automatically with intelligent fallback analysis

---

**Total Coverage**: 36 service agents across 3 cloud providers with intelligent fallback ensuring 100% resource coverage.
