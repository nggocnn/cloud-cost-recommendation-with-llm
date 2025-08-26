# LLM Cost Recommendation System

A multi-agent system for AWS cost optimization using Large Language Models (LLM). This system analyzes AWS resources, billing data, and performance metrics to provide intelligent cost optimization recommendations.

## Features

- **Multi-Agent Architecture**: Coordinator agent orchestrates service-specific agents (EC2, EBS, S3, RDS, Lambda, etc.)
- **LLM-Powered Analysis**: Uses OpenAI GPT models for intelligent cost optimization recommendations
- **Config-Driven**: Service agents are configured through YAML files - add new services without code changes
- **Comprehensive Analysis**: Analyzes rightsizing, purchasing options, storage classes, lifecycle policies, and idle resources
- **Risk Assessment**: Provides low/medium/high risk classifications with implementation guidance
- **Cost Calculations**: Exact cost calculations with monthly and annual savings estimates
- **Data Ingestion**: Supports CSV billing data, JSON inventory, and CSV metrics
- **Structured Output**: Both human-readable reports and machine-readable JSON

## Architecture

```
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
        │ EC2 Agent   │ │ EBS Agent   │ │ S3 Agent    │
        │             │ │             │ │             │
        │ • Rightsizing│ │ • Volume    │ │ • Storage   │
        │ • Reserved  │ │   optimization│ │   classes   │
        │   Instances │ │ • Unattached │ │ • Lifecycle │
        │ • Idle      │ │   volumes   │ │   rules     │
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
python main.py --account-id 123456789012 --sample-data
```

This will:
- Generate sample AWS billing, inventory, and metrics data
- Run the complete analysis pipeline
- Display a comprehensive cost optimization report

### Analyze Real Data

```bash
python main.py \
  --account-id 123456789012 \
  --billing-file data/aws_billing_export.csv \
  --inventory-file data/aws_inventory.json \
  --metrics-file data/cloudwatch_metrics.csv \
  --output-file report.json
```

### Check System Status

```bash
python main.py --status
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

Each AWS service has its own configuration file in `config/`:
- `ec2_agent.yaml` - EC2 instances
- `ebs_agent.yaml` - EBS volumes  
- `s3_agent.yaml` - S3 buckets
- `rds_agent.yaml` - RDS databases
- `lambda_agent.yaml` - Lambda functions

Example agent configuration:
```yaml
agent_id: ec2_agent
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
   cp config/ec2_agent.yaml config/newservice_agent.yaml
   ```

2. **Update the configuration** with service-specific:
   - Supported recommendation types
   - Required/optional metrics
   - Service-specific thresholds
   - LLM prompts

3. **Add service to enum** in `src/models/__init__.py`:
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

### Console Report
```
================================================================================
AWS COST OPTIMIZATION REPORT
================================================================================
Account ID: 123456789012
Generated: 2025-08-26 04:09:44
Total Recommendations: 15
Monthly Savings: $2,450.00
Annual Savings: $29,400.00

RISK DISTRIBUTION:
  Low Risk:    8 recommendations
  Medium Risk: 5 recommendations  
  High Risk:   2 recommendations

SAVINGS BY SERVICE:
  EC2: $1,200.00/month
  EBS: $450.00/month
  S3: $600.00/month
  RDS: $200.00/month

IMPLEMENTATION TIMELINE:
  Quick Wins:   8 recommendations
  Medium Term:  5 recommendations
  Long Term:    2 recommendations

TOP RECOMMENDATIONS:
1. Rightsizing - EC2
   Resource: i-1234567890abcdef0
   Monthly Savings: $400.00
   Risk Level: low
   Rationale: Instance shows consistent low CPU utilization...
```

### JSON Report
Detailed machine-readable report with:
- Complete recommendation details
- Implementation steps
- Risk assessments
- Cost calculations
- Resource metadata

## Development

### Project Structure
```
llm-cost-recommendation/
├── src/
│   ├── models/           # Data models (Pydantic)
│   ├── services/         # Core services (LLM, config, ingestion)
│   └── agents/           # Agent implementations
├── config/               # Agent configurations (YAML)
├── data/                 # Sample and input data
├── main.py              # CLI application
└── requirements.txt     # Python dependencies
```

### Testing

Run with sample data:
```bash
python main.py --account-id test --sample-data
```

### Logging

Structured JSON logging to stdout:
```json
{
  "event": "LLM recommendations generated",
  "agent_id": "ec2_agent", 
  "recommendations_count": 3,
  "response_time_ms": 1250.5,
  "timestamp": "2025-08-26T04:09:35.815868Z",
  "level": "info"
}
```

## API Integration

The system is designed to integrate with:
- **AWS Cost and Usage Reports** (billing data)
- **AWS Config** (resource inventory)  
- **CloudWatch** (performance metrics)
- **AWS Organizations** (multi-account analysis)

## Supported Services

✅ **Implemented**:
- EC2 (Elastic Compute Cloud)
- EBS (Elastic Block Store)  
- S3 (Simple Storage Service)
- RDS (Relational Database Service)
- Lambda (Serverless Functions)

🚧 **Configured but needs tuning**:
- EFS (Elastic File System)
- DynamoDB
- CloudFront
- Application Load Balancer (ALB)
- Network Load Balancer (NLB)
- Elastic IPs
- NAT Gateway
- VPC Endpoints
- SQS (Simple Queue Service)
- SNS (Simple Notification Service)

## Roadmap

- [ ] Azure and GCP support
- [ ] Web UI dashboard
- [ ] API server mode
- [ ] Integration with AWS Cost Explorer
- [ ] ML-based usage pattern prediction
- [ ] Architecture diagram analysis (vision module)
- [ ] Automated implementation workflows
- [ ] Savings tracking and validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add service configurations in `config/`
4. Update documentation
5. Test with sample data
6. Submit pull request

## License

MIT License - see LICENSE file for details.
