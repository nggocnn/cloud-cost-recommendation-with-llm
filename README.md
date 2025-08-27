# LLM Cost Recommendation System

A multi-agent system for cloud cost optimization using Large Language Models (LLM). This system analyzes cloud resources, billing data, and performance metrics to provide intelligent cost optimization recommendations.

## Features

- **Multi-Agent Architecture**: Coordinator agent orchestrates service-specific agents for different cloud services
- **LLM-Powered Analysis**: Uses OpenAI GPT models for intelligent cost optimization recommendations
- **Multi-Cloud Support**: Designed for AWS, Azure, and GCP with extensible architecture
- **Config-Driven**: Service agents are configured through YAML files - add new services without code changes
- **Multiple Export Formats**: JSON (detailed), CSV (summary), and Excel (multi-sheet) output formats
- **Comprehensive Analysis**: Analyzes rightsizing, purchasing options, storage classes, lifecycle policies, and idle resources
- **Risk Assessment**: Provides low/medium/high risk classifications with implementation guidance
- **Cost Calculations**: Exact cost calculations with monthly and annual savings estimates
- **Data Ingestion**: Supports CSV billing data, JSON inventory, and CSV metrics
- **Sample Data Generation**: Built-in sample data for testing and demonstration

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
        │             │ │             │ │             │
        │ • EC2       │ │ • VMs       │ │ • Compute   │
        │ • EBS       │ │ • Disks     │ │ • Storage   │
        │ • S3        │ │ • Storage   │ │ • Functions │
        │ • RDS       │ │ • SQL       │ │ • SQL       │
        │ • Lambda    │ │ • Functions │ │ • CDN       │
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

- `gcp.compute_agent.yaml` - Compute Engine
- `gcp.disk_agent.yaml` - Persistent Disks
- `gcp.storage_agent.yaml` - Cloud Storage
- `gcp.sql_agent.yaml` - Cloud SQL
- `gcp.functions_agent.yaml` - Cloud Functions

Example agent configuration:

```yaml
agent_id: aws.ec2_agent
service: EC2
enabled: true
capability:
  service: EC2
  supported_recommendation_types:
    - rightsizing
    - purchasing_option
    - idle_resource
  required_metrics:
    - cpu_utilization_p50
    - cpu_utilization_p95
    - memory_utilization_p50
  thresholds:
    cpu_idle_threshold: 5.0
    cpu_low_threshold: 20.0
    memory_low_threshold: 30.0
base_prompt: "You are an expert AWS cost optimization specialist focusing on EC2 instances."
service_specific_prompt: |
  Analyze EC2 instances for cost optimization opportunities. Consider:
  1. CPU and memory utilization patterns
  2. Instance family and generation efficiency
  3. Purchase options (On-Demand vs Reserved vs Spot)
  4. Idle or underutilized instances
  5. Right-sizing opportunities based on actual usage
min_cost_threshold: 1.0
confidence_threshold: 0.7
```

### Coordinator Configuration

Global settings in `config/coordinator.yaml`:

```yaml
enabled_services:
  - EC2
  - EBS
  - S3
  - RDS
  - Lambda
similarity_threshold: 0.8
savings_weight: 0.4
risk_weight: 0.3
confidence_weight: 0.2
implementation_ease_weight: 0.1
max_recommendations_per_service: 50
include_low_impact: false
```

## Adding New Services

1. **Create agent configuration**:

   ```bash
   cp config/aws.ec2_agent.yaml config/aws.newservice_agent.yaml
   ```

2. **Update the configuration** with service-specific:
   - Supported recommendation types
   - Required/optional metrics  
   - Service-specific thresholds
   - LLM prompts

3. **Add service to enum** in `llm_cost_recommendation/models/__init__.py`:

   ```python
   class ServiceType(str, Enum):
       # ... existing services
       NEW_SERVICE = "NewService"
   ```

4. **Enable in coordinator** configuration:

   ```yaml
   enabled_services:
     - NEW_SERVICE
   ```

The system automatically discovers and loads the new agent - no code changes required!

## Output

### JSON Report

Detailed machine-readable report with:

- Complete recommendation details
- Implementation steps
- Risk assessments
- Cost calculations
- Resource metadata

## Development

### Project Structure

```text
llm-cost-recommendation/
├── llm_cost_recommendation/  # Main package
│   ├── agents/              # Agent implementations
│   ├── models/              # Data models (Pydantic)
│   ├── services/            # Core services (LLM, config, ingestion)
│   ├── cli.py              # Command line interface
│   └── console.py          # Console output formatting
├── config/                  # Agent configurations (YAML)
│   ├── aws.*.yaml          # AWS service agents
│   ├── azure.*.yaml        # Azure service agents
│   └── gcp.*.yaml          # GCP service agents
├── data/                    # Sample and input data
│   ├── billing/            # Billing data files
│   ├── inventory/          # Resource inventory files
│   └── metrics/            # Performance metrics files
├── pyproject.toml          # Package configuration
├── setup.py               # Package setup
└── requirements.txt       # Python dependencies
```

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

Structured JSON logging to stdout:

```json
{
  "event": "LLM recommendations generated",
  "agent_id": "aws.ec2_agent", 
  "recommendations_count": 3,
  "response_time_ms": 1250.5,
  "timestamp": "2025-08-26T04:09:35.815868Z",
  "level": "info"
}
```

## API Integration (TODO)

The system is designed to integrate with:

- **AWS Cost and Usage Reports** (billing data)
- **AWS Config** (resource inventory)  
- **CloudWatch** (performance metrics)
- **AWS Organizations** (multi-account analysis)

## Supported Services

### AWS Services

- EC2 (Elastic Compute Cloud)
- EBS (Elastic Block Store)  
- S3 (Simple Storage Service)
- RDS (Relational Database Service)
- Lambda (Serverless Functions)
- ALB (Application Load Balancer)
- NLB (Network Load Balancer)
- CloudFront (Content Delivery Network)
- DynamoDB (NoSQL Database)
- EFS (Elastic File System)
- Elastic IPs
- NAT Gateway
- VPC Endpoints
- SQS (Simple Queue Service)
- SNS (Simple Notification Service)

### Azure Services

**Configured (Ready for Testing)**:

- Virtual Machines
- Managed Disks
- Storage Accounts
- SQL Databases
- Azure Functions
- Load Balancer
- Public IPs
- NAT Gateway
- CDN (Content Delivery Network)
- Cosmos DB

### Google Cloud Platform

**Configured (Ready for Testing)**:

- Compute Engine
- Persistent Disks
- Cloud Storage
- Cloud SQL
- Cloud Functions
- Load Balancer
- CDN (Cloud CDN)
- Firestore

### Multi-Cloud Architecture

The system uses a unified approach across all cloud providers:

- **Config-driven**: Each service has its own YAML configuration
- **Provider-specific prompts**: Tailored LLM prompts for each cloud platform
- **Consistent data models**: Unified recommendation format across providers
- **Extensible**: Add new services without code changes

All services are configured and ready to analyze sample data. For production use with real cloud data, additional tuning may be required for specific environments.
