# LLM Cost Recommendation System - Implementation Requirements

## Project Overview

This document describes the complete implementation requirements for the multi-cloud LLM-powered cost optimization system. This serves as a blueprint for reimplementing the project from scratch.

## Information Sources and Research Foundation

This project was built using comprehensive research from multiple authoritative sources to ensure best practices in cloud cost optimization and LLM implementation:

### Cloud Cost Optimization Resources

1; **AWS Cost Optimization Documentation**

- AWS Well-Architected Framework - Cost Optimization Pillar
- AWS Cost and Usage Reports (CUR) data structure and billing analysis
- AWS Trusted Advisor recommendations patterns
- EC2 right-sizing methodologies and instance family optimization
- S3 storage class analysis and lifecycle policies
- Reserved Instance and Savings Plan optimization strategies

2; **Azure Cost Management Best Practices**

- Azure Cost Management + Billing documentation
- Azure Advisor cost recommendations methodology
- Virtual Machine sizing recommendations and performance metrics
- Storage account optimization and access tier strategies
- Azure Reserved VM Instances and Spot pricing models

3; **Google Cloud Cost Optimization**

- Google Cloud cost management and billing best practices
- Compute Engine machine type recommendations
- Cloud Storage class and lifecycle management
- Committed Use Discounts and Preemptible instances
- Cloud Billing API data structures and export formats

### LLM and AI Framework Research

4; **LangChain Framework Documentation**

- Multi-agent system architecture patterns
- Chain composition and prompt engineering best practices
- Structured output parsing and validation techniques
- LangGraph workflow orchestration for complex analysis pipelines

5; **OpenAI API Integration**

- GPT-4 prompt engineering for technical analysis tasks
- Structured response formatting and JSON schema enforcement
- Token optimization and cost-effective API usage patterns
- Error handling and retry mechanisms for production systems

### Industry Standards and Frameworks

6; **FinOps Foundation Guidelines**

- Cloud financial management best practices
- Cost optimization recommendation categorization
- Risk assessment frameworks for cloud changes
- Multi-cloud cost visibility and reporting standards

7; **Cloud Security and Compliance**

- AWS, Azure, GCP security best practices for cost tools
- Data privacy considerations for billing and usage data
- Least privilege access patterns for cost APIs
- Audit logging and compliance requirements

### Technical Implementation Patterns

8; **Python Best Practices**

- Pydantic data modeling and validation patterns
- Async/await patterns for concurrent cloud API calls
- Configuration-driven architecture to prevent code explosion
- Pandas data processing for financial data analysis

9; **Multi-Cloud Architecture Patterns**

- Provider abstraction layers and common data models
- Service mapping between AWS, Azure, and GCP equivalents
- Normalization strategies for cross-cloud cost data
- Plugin architectures for extensible service support

### Data Sources and Formats

10; **Billing Data Standards**
    - AWS Cost and Usage Reports (CUR) CSV format specification
    - Azure consumption API and export formats
    - Google Cloud Billing export BigQuery schema
    - Common fields mapping for multi-cloud normalization

11; **Resource Inventory APIs**
    - AWS Resource Groups Tagging API and Config service
    - Azure Resource Graph queries and inventory management
    - Google Cloud Asset Inventory API and resource discovery
    - Performance metrics collection from CloudWatch, Azure Monitor, Cloud Monitoring

### Real-World Implementation Studies

12; **Case Studies and Patterns**
    - Enterprise cost optimization success stories and methodologies
    - Common anti-patterns and pitfalls in cloud cost management
    - Multi-agent AI system design patterns from industry implementations
    - Cost optimization tool architecture reviews from major platforms

This comprehensive research foundation ensures the system follows industry best practices, implements proven patterns, and provides actionable recommendations based on established cost optimization methodologies across all major cloud providers.

## System Architecture

### Core Components Implemented

1. **Multi-Agent LLM System** using LangChain framework
2. **Multi-Cloud Support** (AWS, Azure, GCP)
3. **Config-Driven Agent Architecture** preventing code explosion
4. **Multiple Export Formats** (JSON, CSV, Excel)
5. **Sample Data Generation** for testing and demonstration
6. **Structured Output** with risk assessment and cost calculations

### Technology Stack

```python
# Core Dependencies (requirements.txt)
langchain>=0.2.0           # LLM framework and chains
langchain-openai>=0.1.8    # OpenAI integration
langgraph>=0.0.69          # Advanced workflow graphs
pydantic>=2.7.1            # Data models and validation
python-dotenv>=1.0.1       # Environment configuration
pandas>=2.2.2              # Data processing and export
numpy>=1.26.4              # Numerical computations
pyyaml>=6.0.1              # Configuration file parsing
structlog>=24.1.0          # Structured logging
boto3>=1.34.0              # AWS SDK (for future pricing API)
fastapi>=0.111.0           # Web API framework (optional)
uvicorn>=0.30.1            # ASGI server (optional)
```

## Project Structure Implementation

```text
llm-cost-recommendation/
├── pyproject.toml              # Package configuration
├── setup.py                   # Package setup
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
├── README.md                  # User documentation
├── ARCHITECTURE.md            # Technical architecture
├── REQUIREMENTS.md            # This implementation guide
├── CUSTOM_CONDITIONS.md       # Custom rules documentation
│
├── llm_cost_recommendation/   # Main package
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   ├── cli.py                # Command-line interface
│   ├── console.py            # Output formatting
│   ├── agents/               # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base.py           # Base agent classes
│   │   └── coordinator.py    # Orchestrator agent
│   ├── models/               # Data models (Pydantic)
│   │   ├── __init__.py       # Core models and enums
│   │   └── *.py             # Additional models
│   └── services/             # Core services
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── llm.py           # LLM service integration
│       ├── ingestion.py     # Data ingestion
│       └── logging.py       # Logging configuration
│
├── config/                    # Agent configurations
│   ├── coordinator.yaml      # Global settings
│   ├── aws.*.yaml           # AWS service agents (17 services)
│   ├── azure.*.yaml         # Azure service agents (10 services)
│   └── gcp.*.yaml           # GCP service agents (8 services)
│
└── data/                     # Sample and input data
    ├── billing/              # Cost data
    ├── inventory/            # Resource configurations
    └── metrics/              # Performance data
```

## Data Models and Configuration Implementation

### Core Data Models (Pydantic)

```python
# llm_cost_recommendation/models/__init__.py

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure" 
    GCP = "GCP"

class ServiceType(str, Enum):
    # AWS Services
    EC2 = "EC2"
    EBS = "EBS"
    S3 = "S3"
    RDS = "RDS"
    LAMBDA = "Lambda"
    ALB = "ALB"
    NLB = "NLB"
    CLOUDFRONT = "CloudFront"
    DYNAMODB = "DynamoDB"
    EFS = "EFS"
    ELASTICIP = "ElasticIP"
    NATGATEWAY = "NATGateway"
    VPCENDPOINTS = "VPCEndpoints"
    SQS = "SQS"
    SNS = "SNS"
    
    # Azure Services
    VIRTUAL_MACHINES = "VirtualMachines"
    MANAGED_DISKS = "ManagedDisks"
    STORAGE_ACCOUNTS = "StorageAccounts"
    SQL_DATABASE = "SQLDatabase"
    AZURE_FUNCTIONS = "AzureFunctions"
    
    # GCP Services
    COMPUTE_ENGINE = "ComputeEngine"
    PERSISTENT_DISKS = "PersistentDisks"
    CLOUD_STORAGE = "CloudStorage"
    CLOUD_SQL = "CloudSQL"
    CLOUD_FUNCTIONS = "CloudFunctions"

class RecommendationType(str, Enum):
    RIGHTSIZING = "rightsizing"
    PURCHASING_OPTION = "purchasing_option"
    IDLE_RESOURCE = "idle_resource"
    STORAGE_CLASS = "storage_class"
    LIFECYCLE_POLICY = "lifecycle_policy"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Resource(BaseModel):
    resource_id: str
    service: ServiceType
    cloud_provider: CloudProvider
    region: str
    account_id: str
    tags: Dict[str, str] = {}
    properties: Dict[str, Union[str, int, float]] = {}

class Metrics(BaseModel):
    resource_id: str
    timestamp: str
    period_days: int
    cpu_utilization_p50: Optional[float]
    cpu_utilization_p90: Optional[float]
    cpu_utilization_p95: Optional[float]
    memory_utilization_p50: Optional[float]
    memory_utilization_p90: Optional[float]
    memory_utilization_p95: Optional[float]
    is_idle: bool = False

class BillingData(BaseModel):
    resource_id: str
    service: ServiceType
    cloud_provider: CloudProvider
    cost: float
    date: str
    region: str
    usage_type: str
    usage_amount: float
    usage_unit: str

class Recommendation(BaseModel):
    id: str
    resource_id: str
    service: ServiceType
    cloud_provider: CloudProvider
    recommendation_type: RecommendationType
    title: str
    description: str
    rationale: str
    current_configuration: Dict[str, Union[str, int, float]]
    recommended_configuration: Dict[str, Union[str, int, float]]
    monthly_savings: float
    annual_savings: float
    implementation_effort: str
    risk_level: RiskLevel
    confidence_score: float
```

### Agent Configuration Schema

```yaml
# config/coordinator.yaml
enabled_services:
  # AWS Services
  - EC2
  - EBS
  - S3
  - RDS
  - Lambda
  - ALB
  - NLB
  - CloudFront
  - DynamoDB
  - EFS
  - ElasticIP
  - NATGateway
  - VPCEndpoints
  - SQS
  - SNS
  
  # Azure Services
  - VirtualMachines
  - ManagedDisks
  - StorageAccounts
  - SQLDatabase
  - AzureFunctions
  
  # GCP Services
  - ComputeEngine
  - PersistentDisks
  - CloudStorage
  - CloudSQL
  - CloudFunctions

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

```yaml
# config/aws.ec2_agent.yaml (Example Service Agent)
agent_id: aws.ec2_agent
service: EC2
cloud_provider: AWS
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

# LLM prompts
base_prompt: "You are an expert AWS cost optimization specialist focusing on EC2 instances."
service_specific_prompt: |
  Analyze EC2 instances for cost optimization opportunities. Consider:
  1. CPU and memory utilization patterns
  2. Instance family and generation efficiency  
  3. Purchase options (On-Demand vs Reserved vs Spot)
  4. Idle or underutilized instances
  5. Right-sizing opportunities based on actual usage

# Agent settings
max_tokens: 2000
confidence_threshold: 0.7
min_cost_threshold: 1.0
```

### 1 Overall architecture

- **Ingestion layer**

  - **Billing:** Start with CSV exports (e.g., AWS cost & usage). Required fields: account, service, resource ID, region/AZ, usage type, usage amount + unit, blended/unblended cost, discounts/credits, and tags.
  - **Inventory:** Periodic snapshots of live resources (EC2, EBS, S3, etc.) with their properties. (currently we can use mock data from json file)
  - **Metrics:** Rolling windows (e.g., 7/30/90 days) of utilization per service (CPU/mem/IOPS/latency/requests). (csv file)
- **Normalization**

  - Map all providers to a **common schema** (Resource, Metrics, Pricing, Tags). Keep provider-specific extras in an “extensions” field.

- **Recommendation engine** (base on LangChain framework, may use LangGraph as well, use OpenAI API configure with configuration from .env file)

  - **Coordinator agent** (orchestrator) that:

    - Routes each resource to the right **service agent** (EC2 agent, S3 agent, etc.).
    - Consolidates recommendations, deduplicates conflicts, and ranks by savings vs. risk.
  - **Service agents** (one per service/provider) that:

    - Read normalized inputs + service-specific metrics.
    - Propose rightsizing, tier changes, lifecycle rules, purchasing options, and architectural tweaks.
    - Service agents must be implemented in config-based entity so that when adding new service, new config file will be add, no code explosion happens
  - **Rules & guardrails layer** (deterministic checks) for “obvious” wins (e.g., idle EIP, unattached EBS, NAT with huge GB).
- **Outputs**

  - Human-readable report: itemized recommendations, estimated monthly saving, risk/impact, and exact “why”.
  - Machine-readable JSON for pipelines/dashboards.

### 2 Prevent “code explosion”

- **Plugin model**

  - Each service agent is a **plugin** discovered via a registry (name, provider, version, capabilities).
  - Shared **base agent contract**: inputs (schema), expected outputs (Recommendation list with cost delta), evaluation rubric.
- **Config-first**

  - Load agents, thresholds, and analysis windows from configuration, not hard-coded.
  - Service agents must be implemented in config-based entity so that when adding new service, new config file will be add, no code explosion happens
- **Shared prompt scaffolding**

  - **Base prompt**: global goals (minimize cost, preserve SLOs), constraints (no downtime unless flagged), and common data dictionary.
  - **Service prompt add-ons**: only the deltas (e.g., S3 storage classes, EBS IOPS semantics).
  - **Response schema**: require structured fields (action, rationale, before/after config, cost delta, risk, rollback).

### 3 Vision module (optional, additive)

- **Input:** Architecture diagrams (PNG/SVG/PDF) or exported graphs.
- **Processing:** A vision LLM extracts components (icons, labels), links (data flows), and annotations (regions/AZs).
- **Fusion:** Cross-check diagram entities with inventory; flag mismatches (e.g., undocumented NAT, overlooked peering).
- **Value:** Improves **data-transfer** and **topology-driven** recommendations (e.g., move workloads to reduce cross-AZ, add Gateway Endpoint to cut NAT egress).

### 4 Exact pricing with recommended configurations

- **For each recommendation**, compute:

  - Current monthly cost (from bill/metrics).
  - Proposed monthly cost (from pricing API using the recommended size/class/commitment).
  - Savings, break-even (if RI/SP), and sensitivity (load growth).
- **Include**: storage request tiers, retrieval charges, per-hour LCU/NLCU/GLCU, NAT GB processed, data transfer matrices, Lambda duration × memory × invokes.

### 5. Let the agents propose rightsizing

- **Inputs to give the agents:**

  - Utilization percentiles (P50/P90/P95), peak windows, sustained vs. spiky patterns.
  - Performance headroom target (e.g., keep P95 CPU < 60%).
  - Business constraints (prod vs. non-prod, maintenance windows).
- **Expected outputs from agents:**

  - Concrete new size/class/tier (e.g., `m7g.large` from `m5.xlarge`, or S3 IA from Standard).
  - Price comparison table and performance implications.
  - Rollout plan (test first, canary, revert path).

### 6 Starting with CSV billing data

- **Minimum columns to include:**

  - `bill_period_start`, `bill_period_end`, `account_id`, `service`, `region`, `availability_zone` (if any), `resource_id` (or line-item resource), `usage_type`, `usage_amount`, `usage_unit`, `unblended_cost`, `amortized_cost`, `credit`, `savings_plan_eligible`, `reservation_applied`, `tags_*`.
- **Transformations:**

  - Aggregate per resource per day.
  - Pivot key usage types (e.g., `DataTransfer-Out-Bytes`, `TimedStorage-ByteHrs`, `ReadRequests`, `WriteRequests`).
  - Join with inventory + metrics by `resource_id` and time.
- **Quality checks:**

  - Missing tags, missing metrics, anomalies (sudden spikes), orphaned spend (no matching resource).

### 7 Extending to new services and providers

- **Service plugin contract** (applies to AWS today; Azure/GCP later):

  - **Metadata:** `provider`, `service`, `version`.
  - **Inputs:** normalized resource doc + metrics + pricing accessor.
  - **Outputs:** list of Recommendations with structured fields.
  - **Capabilities:** which recommendation types it can emit (rightsizing, lifecycle, purchasing option, topology).
- **Provider adapter**

  - Implements: inventory fetch, metrics fetch, pricing lookup, and id mapping.
  - Keeps provider-specific quirks inside the adapter; the rest of the system stays the same.
- **Placeholders**

  - Create empty entries for Azure and GCP in the registry (disabled until configured), and a shared mapping table for “equivalent” services (e.g., EC2 ↔️ VM ↔️ Compute Engine; S3 ↔️ Blob ↔️ GCS).

## Multi-Agent System Implementation

### 4. Base Agent Architecture (agents/base.py)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from ..models import Resource, Metrics, BillingData, Recommendation, ServiceAgentConfig
from ..services.llm import LLMService
import structlog

logger = structlog.get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all service-specific agents"""
    
    def __init__(self, config: ServiceAgentConfig, llm_service: LLMService):
        self.config = config
        self.llm_service = llm_service
        self.service_type = config.service
        self.cloud_provider = config.cloud_provider
    
    async def analyze_resources(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics],
        billing_data: Dict[str, List[BillingData]]
    ) -> List[Recommendation]:
        """Main entry point for resource analysis"""
        recommendations = []
        
        for resource in resources:
            if self._should_analyze_resource(resource):
                resource_recommendations = await self.analyze_resource(
                    resource, 
                    metrics_data.get(resource.resource_id),
                    billing_data.get(resource.resource_id, [])
                )
                recommendations.extend(resource_recommendations)
        
        return self._filter_recommendations(recommendations)
    
    @abstractmethod
    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> List[Recommendation]:
        """Analyze a single resource and generate recommendations"""
        pass
    
    def _should_analyze_resource(self, resource: Resource) -> bool:
        """Determine if this agent should analyze the given resource"""
        return (resource.service == self.service_type and 
                resource.cloud_provider == self.cloud_provider)
    
    def _filter_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Filter recommendations based on confidence and cost thresholds"""
        filtered = []
        for rec in recommendations:
            if (rec.confidence_score >= self.config.confidence_threshold and
                rec.monthly_savings >= self.config.min_cost_threshold):
                filtered.append(rec)
        
        return filtered[:self.config.max_recommendations_per_service]
```

### 5. Coordinator Agent (agents/coordinator.py)

```python
from typing import Dict, List
from ..models import Resource, Metrics, BillingData, Recommendation, ServiceType
from ..services.config import ConfigManager
from ..services.llm import LLMService
from .base import BaseAgent
import structlog

logger = structlog.get_logger(__name__)

class CoordinatorAgent:
    """Orchestrates analysis across all service agents"""
    
    def __init__(self, config_manager: ConfigManager, llm_service: LLMService):
        self.config_manager = config_manager
        self.llm_service = llm_service
        self.agents = self._create_agents()
        self.enabled_services = config_manager.coordinator_config.enabled_services
    
    def _create_agents(self) -> Dict[str, BaseAgent]:
        """Dynamically create agents based on configuration"""
        agents = {}
        
        for agent_id, config in self.config_manager.agent_configs.items():
            if config.enabled and config.service in self.enabled_services:
                # Create specific agent based on service type
                agent_class = self._get_agent_class(config.service, config.cloud_provider)
                agents[agent_id] = agent_class(config, self.llm_service)
        
        return agents
    
    def _get_agent_class(self, service: str, provider: str) -> type:
        """Return the appropriate agent class for service/provider combination"""
        # In a full implementation, this would import and return specific agent classes
        # For now, return a generic agent implementation
        return GenericServiceAgent
    
    async def analyze_all_resources(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics],
        billing_data: Dict[str, List[BillingData]]
    ) -> List[Recommendation]:
        """Coordinate analysis across all agents"""
        all_recommendations = []
        
        # Group resources by service/provider
        resource_groups = self._group_resources_by_service(resources)
        
        for service_key, service_resources in resource_groups.items():
            agent = self._find_agent_for_service(service_key)
            if agent:
                logger.info(f"Analyzing {len(service_resources)} resources with {agent.__class__.__name__}")
                
                recommendations = await agent.analyze_resources(
                    service_resources, metrics_data, billing_data
                )
                all_recommendations.extend(recommendations)
        
        return self._deduplicate_and_rank_recommendations(all_recommendations)
    
    def _group_resources_by_service(self, resources: List[Resource]) -> Dict[str, List[Resource]]:
        """Group resources by service type and cloud provider"""
        groups = {}
        for resource in resources:
            key = f"{resource.cloud_provider}.{resource.service}"
            if key not in groups:
                groups[key] = []
            groups[key].append(resource)
        return groups
    
    def _find_agent_for_service(self, service_key: str) -> Optional[BaseAgent]:
        """Find the appropriate agent for a service"""
        for agent_id, agent in self.agents.items():
            if service_key in agent_id:
                return agent
        return None
    
    def _deduplicate_and_rank_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicates and rank by priority"""
        # Implementation for deduplication and ranking logic
        # based on coordinator configuration weights
        pass

class GenericServiceAgent(BaseAgent):
    """Generic agent implementation for config-driven analysis"""
    
    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> List[Recommendation]:
        """Analyze resource using LLM with configuration-driven prompts"""
        
        # Prepare context data
        context_data = {
            "resource": resource.dict(),
            "metrics": metrics.dict() if metrics else {},
            "billing": [b.dict() for b in billing_data] if billing_data else [],
            "thresholds": self.config.capability.get("thresholds", {}),
            "supported_recommendations": self.config.capability.get("supported_recommendation_types", [])
        }
        
        # Generate recommendations using LLM
        llm_recommendations = await self.llm_service.generate_recommendations(
            self.config.base_prompt,
            self.config.service_specific_prompt,
            context_data,
            self.config.max_tokens
        )
        
        # Convert to recommendation models
        recommendations = []
        for i, llm_rec in enumerate(llm_recommendations):
            rec = Recommendation(
                id=f"{resource.resource_id}_{i}",
                resource_id=resource.resource_id,
                service=resource.service,
                cloud_provider=resource.cloud_provider,
                **llm_rec
            )
            recommendations.append(rec)
        
        return recommendations
```

## CLI and Export Implementation

### 6. Command Line Interface (cli.py)

```python
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import structlog

from .services.config import ConfigManager
from .services.llm import LLMService
from .services.ingestion import DataIngestionService
from .agents.coordinator import CoordinatorAgent
from .console import ConsoleReportGenerator

logger = structlog.get_logger(__name__)

class CostRecommendationApp:
    """Main application class"""
    
    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        self.config_manager = ConfigManager(config_dir)
        self.data_service = DataIngestionService(data_dir)
        self.llm_service = LLMService()
        self.coordinator = CoordinatorAgent(self.config_manager, self.llm_service)
        self.console = ConsoleReportGenerator()
    
    async def run_analysis(
        self,
        billing_file: Optional[str] = None,
        inventory_file: Optional[str] = None,
        metrics_file: Optional[str] = None,
        sample_data: bool = False
    ):
        """Run the complete analysis pipeline"""
        
        if sample_data:
            logger.info("Generating sample data for analysis")
            self.data_service.generate_sample_data("multi-account")  # Default account ID
            billing_file = "data/billing/sample_billing.csv"
            inventory_file = "data/inventory/sample_inventory.json"
            metrics_file = "data/metrics/sample_metrics.csv"
        
        # Ingest data
        resources = self.data_service.ingest_inventory_data(inventory_file)
        metrics_data = {m.resource_id: m for m in self.data_service.ingest_metrics_data(metrics_file)}
        billing_data = {}
        for b in self.data_service.ingest_billing_data(billing_file):
            if b.resource_id not in billing_data:
                billing_data[b.resource_id] = []
            billing_data[b.resource_id].append(b)
        
        # Analyze resources
        recommendations = await self.coordinator.analyze_all_resources(
            resources, metrics_data, billing_data
        )
        
        return {
            "account_id": "multi-account",  # Default for multi-account analysis
            "resources": resources,
            "recommendations": recommendations,
            "summary": self._generate_summary(recommendations)
        }
    
    def export_report(self, report, output_file: str, format_type: str = "json"):
        """Export report in specified format"""
        
        if format_type == "json":
            self._export_json(report, output_file)
        elif format_type == "csv":
            self._export_csv(report, output_file)
        elif format_type == "excel":
            self._export_excel(report, output_file)
    
    def _export_json(self, report, output_file: str):
        """Export detailed JSON report"""
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _export_csv(self, report, output_file: str):
        """Export summary CSV table"""
        recommendations = report["recommendations"]
        df = pd.DataFrame([{
            "Resource ID": r.resource_id,
            "Service": r.service,
            "Cloud Provider": r.cloud_provider,
            "Recommendation Type": r.recommendation_type,
            "Monthly Savings": r.monthly_savings,
            "Annual Savings": r.annual_savings,
            "Risk Level": r.risk_level,
            "Confidence": r.confidence_score
        } for r in recommendations])
        df.to_csv(output_file, index=False)
    
    def _export_excel(self, report, output_file: str):
        """Export multi-sheet Excel workbook"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([report["summary"]])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Recommendations sheet
            rec_df = pd.DataFrame([r.dict() for r in report["recommendations"]])
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Resources sheet
            res_df = pd.DataFrame([r.dict() for r in report["resources"]])
            res_df.to_excel(writer, sheet_name='Resources', index=False)

def main():
    parser = argparse.ArgumentParser(description="LLM Cost Recommendation System")

    parser.add_argument("--billing-file", help="Path to billing data CSV")
    parser.add_argument("--inventory-file", help="Path to inventory data JSON")
    parser.add_argument("--metrics-file", help="Path to metrics data CSV")
    parser.add_argument("--sample-data", action="store_true", help="Use sample data")
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--output-format", choices=["json", "csv", "excel"], 
                       default="json", help="Output format")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    app = CostRecommendationApp()
    
    if args.status:
        print("System Status: OK")
        print(f"Enabled Services: {len(app.config_manager.coordinator_config.enabled_services)}")
        print(f"Configured Agents: {len(app.config_manager.agent_configs)}")
        return
    
    # Run analysis
    report = asyncio.run(app.run_analysis(
        args.billing_file,
        args.inventory_file, 
        args.metrics_file,
        args.sample_data
    ))
    
    # Display console report
    app.console.display_report(report)
    
    # Export if requested
    if args.output_file:
        app.export_report(report, args.output_file, args.output_format)

if __name__ == "__main__":
    main()
```

## Environment Configuration

### 7. Environment Setup (.env.example)

```bash
# LLM Configuration - Required
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Optional: AWS Pricing API (for future enhancements)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key  
# AWS_DEFAULT_REGION=us-east-1

# Optional: Logging Configuration
# LOG_LEVEL=INFO
# LOG_FORMAT=json
```

### 8. Package Configuration (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-cost-recommendation"
version = "1.0.0"
description = "A multi-agent system for multi-cloud cost optimization using Large Language Models"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Cloud Cost Optimization Team", email = "team@example.com"}
]
keywords = ["cloud", "cost-optimization", "llm", "ai", "aws", "azure", "gcp", "finops"]

dependencies = [
    "langchain>=0.2.0",
    "langchain-openai>=0.1.8", 
    "langgraph>=0.0.69",
    "pydantic>=2.7.1",
    "python-dotenv>=1.0.1",
    "pandas>=2.2.2",
    "numpy>=1.26.4",
    "pyyaml>=6.0.1",
    "structlog>=24.1.0",
    "openpyxl>=3.1.0",
    "boto3>=1.34.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0"
]

[project.scripts]
llm-cost-recommendation = "llm_cost_recommendation.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["llm_cost_recommendation*"]

[tool.black]
line-length = 100
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

## Implementation Steps for Recreation

### Phase 1: Project Setup (Day 1)

1. **Initialize Project Structure**

   ```bash
   mkdir llm-cost-recommendation
   cd llm-cost-recommendation
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Create Core Files**

   ```bash
   touch pyproject.toml setup.py requirements.txt
   touch .env.example .gitignore README.md
   mkdir -p llm_cost_recommendation/{agents,models,services}
   mkdir -p config data/{billing,inventory,metrics}
   ```

3. **Install Dependencies**

   ```bash
   pip install langchain langchain-openai langgraph pydantic python-dotenv
   pip install pandas numpy pyyaml structlog openpyxl boto3
   ```

### Phase 2: Core Models and Services (Days 2-3)

1. **Implement Data Models** (`llm_cost_recommendation/models/__init__.py`)

   - CloudProvider, ServiceType, RecommendationType enums
   - Resource, Metrics, BillingData, Recommendation models

2. **Configuration Management** (`llm_cost_recommendation/services/config.py`)

   - ConfigManager class
   - YAML loading and validation

3. **LLM Service** (`llm_cost_recommendation/services/llm.py`)
   - LLMService with OpenAI integration
   - Structured prompt handling

4. **Data Ingestion** (`llm_cost_recommendation/services/ingestion.py`)
   - CSV/JSON data loading
   - Sample data generation

### Phase 3: Agent System (Days 4-6)

1. **Base Agent Architecture** (`llm_cost_recommendation/agents/base.py`)
   - BaseAgent abstract class
   - Generic agent implementation

2. **Coordinator Agent** (`llm_cost_recommendation/agents/coordinator.py`)
   - Multi-agent orchestration
   - Recommendation aggregation and ranking

3. **Configuration Files** (`config/*.yaml`)
   - Create 35+ service agent configurations
   - AWS, Azure, GCP service definitions

### Phase 4: CLI and Export (Days 7-8)

1. **Command Line Interface** (`llm_cost_recommendation/cli.py`)
   - Argument parsing
   - Analysis pipeline orchestration

2. **Console Output** (`llm_cost_recommendation/console.py`)
   - Human-readable report formatting
   - Summary statistics

3. **Export Capabilities**
   - JSON (detailed), CSV (summary), Excel (multi-sheet)
   - Pandas integration for data manipulation

### Phase 5: Testing and Documentation (Days 9-10)

1. **Sample Data Creation**
   - Realistic test data for all three cloud providers
   - Various resource types and usage patterns

2. **Documentation**
   - README.md with usage examples
   - ARCHITECTURE.md with technical details
   - This REQUIREMENTS.md as implementation guide

3. **Testing**
   - Unit tests for core components
   - Integration tests with sample data
   - CLI testing with various scenarios

## Key Configuration Files Required

### Coordinator Configuration

- `config/coordinator.yaml`: Global settings and enabled services

### AWS Service Agents (17 configurations)

- `config/aws.ec2_agent.yaml`: EC2 instances

- `config/aws.ebs_agent.yaml`: EBS volumes
- `config/aws.s3_agent.yaml`: S3 buckets
- `config/aws.rds_agent.yaml`: RDS databases
- `config/aws.lambda_agent.yaml`: Lambda functions
- `config/aws.alb_agent.yaml`: Application Load Balancer
- `config/aws.nlb_agent.yaml`: Network Load Balancer
- `config/aws.cloudfront_agent.yaml`: CloudFront CDN
- `config/aws.dynamodb_agent.yaml`: DynamoDB tables
- `config/aws.efs_agent.yaml`: EFS file systems
- `config/aws.elasticip_agent.yaml`: Elastic IPs
- `config/aws.natgateway_agent.yaml`: NAT Gateways
- `config/aws.vpcendpoints_agent.yaml`: VPC Endpoints
- `config/aws.sqs_agent.yaml`: SQS queues
- `config/aws.sns_agent.yaml`: SNS topics
- `config/aws.gwlb_agent.yaml`: Gateway Load Balancer
- `config/aws.rds_snapshots_agent.yaml`: RDS snapshots

### Azure Service Agents (10 configurations)

- `config/azure.vm_agent.yaml`: Virtual Machines

- `config/azure.disk_agent.yaml`: Managed Disks
- `config/azure.storage_agent.yaml`: Storage Accounts
- `config/azure.sql_agent.yaml`: SQL Databases
- `config/azure.functions_agent.yaml`: Azure Functions
- `config/azure.loadbalancer_agent.yaml`: Load Balancer
- `config/azure.publicip_agent.yaml`: Public IPs
- `config/azure.natgateway_agent.yaml`: NAT Gateway
- `config/azure.cdn_agent.yaml`: CDN
- `config/azure.cosmos_agent.yaml`: Cosmos DB

### GCP Service Agents (8 configurations)

- `config/gcp.compute_agent.yaml`: Compute Engine

- `config/gcp.disk_agent.yaml`: Persistent Disks
- `config/gcp.storage_agent.yaml`: Cloud Storage
- `config/gcp.sql_agent.yaml`: Cloud SQL
- `config/gcp.functions_agent.yaml`: Cloud Functions
- `config/gcp.loadbalancer_agent.yaml`: Load Balancer
- `config/gcp.cdn_agent.yaml`: Cloud CDN
- `config/gcp.firestore_agent.yaml`: Firestore

## Success Criteria

The implementation is complete when:

1. **CLI Functionality**: System runs with `python -m llm_cost_recommendation --sample-data`
2. **Multi-format Export**: Generates JSON, CSV, and Excel reports successfully
3. **Multi-cloud Support**: Handles AWS, Azure, and GCP resources in configuration
4. **Agent System**: 35+ service agents load from configuration files
5. **LLM Integration**: OpenAI API generates realistic cost optimization recommendations
6. **Sample Data**: System works with generated test data out of the box
7. **Multi-account Support**: Processes resources from multiple accounts without filtering
8. **Documentation**: Complete README, ARCHITECTURE, and this REQUIREMENTS guide

This implementation creates a robust, extensible foundation for multi-cloud cost optimization that can be easily recreated and enhanced.
