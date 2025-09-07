"""
Configuration management for the LLM cost recommendation system.
"""

import os
import yaml
from typing import Dict, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

from ..models import (
    ServiceAgentConfig,
    GlobalConfig,
    ServiceType,
    RecommendationType,
    AgentCapability,
)


class LLMConfig(BaseModel):
    """LLM configuration"""

    provider: str = "openai"
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30


class ConfigManager:
    """Manages configuration for the entire system"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Initialize configurations
        self.llm_config = self._load_llm_config()
        self.global_config = self._load_global_config()
        self.service_configs = self._load_service_configs()

    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from environment variables"""
        return LLMConfig(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
        )

    def _load_global_config(self) -> GlobalConfig:
        """Load global configuration shared across all agents"""
        config_file = self.config_dir / "global" / "coordinator.yaml"

        if not config_file.exists():
            # Create default configuration
            default_config = {
                "enabled_services": [service.value for service in ServiceType],
                "similarity_threshold": 0.8,
                "savings_weight": 0.4,
                "risk_weight": 0.3,
                "confidence_weight": 0.2,
                "implementation_ease_weight": 0.1,
                "max_recommendations_per_service": 50,
                "include_low_impact": False,
                "cost_tiers": {
                    "minimal_cost": {"min": 0, "max": 10, "batch_adjustment": 2},
                    "low_cost": {"min": 10, "max": 100, "batch_adjustment": 0},
                    "medium_cost": {"min": 100, "max": 1000, "batch_adjustment": -1},
                    "high_cost": {
                        "min": 1000,
                        "max": float("inf"),
                        "batch_adjustment": -2,
                    },
                },
                "complexity_tiers": {
                    "simple": {"metric_threshold": 3, "base_batch_size": 6},
                    "moderate": {"metric_threshold": 8, "base_batch_size": 4},
                    "complex": {"metric_threshold": float("inf"), "base_batch_size": 2},
                },
                "batch_config": {
                    "min_batch_size": 1,
                    "max_batch_size": 10,
                    "default_batch_size": 4,
                    "single_resource_threshold_cost": 5000,
                },
            }

            with open(config_file, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        # Fix 'inf' strings from YAML to proper float('inf')
        config_data = self._fix_yaml_inf_values(config_data)

        return GlobalConfig(**config_data)

    def _fix_yaml_inf_values(self, data):
        """Recursively fix 'inf' strings from YAML to proper float('inf')"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.lower() == "inf":
                    data[key] = float("inf")
                elif isinstance(value, (dict, list)):
                    data[key] = self._fix_yaml_inf_values(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.lower() == "inf":
                    data[i] = float("inf")
                elif isinstance(item, (dict, list)):
                    data[i] = self._fix_yaml_inf_values(item)
        return data

    def _get_agent_config_path(self, service: ServiceType) -> Path:
        """Get the config file path for a service agent based on new hierarchical structure"""
        service_str = service.value.lower()

        # Handle default service type
        if service_str == "default":
            return self.config_dir / "agents" / "default" / "default.yaml"

        # Map service types to cloud provider
        if service_str.startswith("aws."):
            cloud = "aws"
            service_name = service_str[4:]  # Remove 'aws.' prefix
        elif service_str.startswith("azure."):
            cloud = "azure"
            service_name = service_str[6:]  # Remove 'azure.' prefix
        elif service_str.startswith("gcp."):
            cloud = "gcp"
            service_name = service_str[4:]  # Remove 'gcp.' prefix
        else:
            # Legacy service names without cloud prefix - assume AWS
            cloud = "aws"
            service_name = service_str

        return self.config_dir / "agents" / cloud / f"{service_name}.yaml"

    def _load_service_configs(self) -> Dict[ServiceType, ServiceAgentConfig]:
        """Load service agent configurations"""
        configs = {}

        for service in ServiceType.get_all_services():
            config_file = self._get_agent_config_path(service)

            if not config_file.exists():
                # Create default configuration for this service
                default_config = self._create_default_service_config(service)

                # Ensure directory exists
                config_file.parent.mkdir(parents=True, exist_ok=True)

                with open(config_file, "w") as f:
                    yaml.dump(default_config, f, default_flow_style=False)

            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Convert to ServiceAgentConfig
            capability_data = config_data.pop("capability")
            capability = AgentCapability(**capability_data)

            config_data["capability"] = capability
            configs[service] = ServiceAgentConfig(**config_data)

        return configs

    def _create_default_service_config(self, service: ServiceType) -> Dict:
        """Create default configuration for a service"""
        default_config = {
            "agent_id": "default_agent",
            "service": ServiceType.DEFAULT,
            "enabled": True,
            "capability": {
                "service": ServiceType.DEFAULT,
                "supported_recommendation_types": [
                    RecommendationType.COST_ANALYSIS.value,
                    RecommendationType.GENERAL_OPTIMIZATION.value,
                ],
                "required_metrics": ["monthly_cost"],
                "optional_metrics": [
                    "cost_per_hour",
                    "usage_hours",
                    "network_in",
                    "network_out",
                ],
                "thresholds": {
                    "cost_threshold": 1.0,
                    "usage_threshold": 0.1,
                },
                "analysis_window_days": 30,
            },
            "base_prompt": "You are a cloud cost optimization specialist with expertise across multiple cloud providers.",
            "service_specific_prompt": """
You are analyzing a cloud resource that doesn't have a specialized agent yet.

Provide general cost optimization recommendations based on:
1. Resource utilization patterns
2. Cost trends and billing data
3. Industry best practices
4. General cloud optimization principles

Focus on:
- Identifying obviously underutilized resources
- Suggesting general cost optimization strategies
- Recommending further analysis with specialized tools
- Highlighting potential savings opportunities

Be conservative in your recommendations since you don't have service-specific expertise.
Always suggest consulting with service-specific specialists for detailed analysis.

Provide actionable but general recommendations with cost estimates when possible.
""",
            "temperature": 0.2,
            "max_tokens": 2000,
            "min_cost_threshold": 1.0,
            "confidence_threshold": 0.6,
        }

        # Define service-specific configurations
        service_configs = {
            ServiceType.AWS.EC2: {
                "agent_id": "ec2_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.PURCHASING_OPTION.value,
                        RecommendationType.IDLE_RESOURCE.value,
                    ],
                    "required_metrics": [
                        "cpu_utilization_p50",
                        "cpu_utilization_p95",
                        "memory_utilization_p50",
                    ],
                    "optional_metrics": ["network_in", "network_out"],
                    "thresholds": {
                        "cpu_idle_threshold": 5.0,
                        "cpu_low_threshold": 20.0,
                        "memory_low_threshold": 30.0,
                        "uptime_threshold": 0.95,
                    },
                    "analysis_window_days": 30,
                },
                "base_prompt": "You are an expert AWS cost optimization specialist focusing on EC2 instances.",
                "service_specific_prompt": """
Analyze EC2 instances for cost optimization opportunities. Consider:
1. CPU and memory utilization patterns
2. Instance family and generation efficiency
3. Purchase options (On-Demand vs Reserved vs Spot)
4. Idle or underutilized instances
5. Right-sizing opportunities based on actual usage

Provide specific recommendations with exact instance types and cost calculations.
""",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7,
            },
            ServiceType.AWS.EBS: {
                "agent_id": "ebs_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.STORAGE_CLASS.value,
                        RecommendationType.IDLE_RESOURCE.value,
                    ],
                    "required_metrics": [
                        "iops_read",
                        "iops_write",
                        "throughput_read",
                        "throughput_write",
                    ],
                    "optional_metrics": ["burst_credits"],
                    "thresholds": {
                        "iops_utilization_threshold": 10.0,
                        "throughput_utilization_threshold": 10.0,
                        "idle_threshold": 1.0,
                    },
                    "analysis_window_days": 30,
                },
                "base_prompt": "You are an expert AWS cost optimization specialist focusing on EBS volumes.",
                "service_specific_prompt": """
Analyze EBS volumes for cost optimization opportunities. Consider:
1. IOPS and throughput utilization
2. Volume type optimization (gp2 vs gp3 vs io1 vs io2)
3. Provisioned vs actual usage
4. Unattached or unused volumes
5. Snapshot management

Provide specific recommendations with volume types and cost calculations.
""",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7,
            },
            ServiceType.AWS.S3: {
                "agent_id": "s3_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.STORAGE_CLASS.value,
                        RecommendationType.LIFECYCLE.value,
                    ],
                    "required_metrics": [
                        "storage_used",
                        "requests_get",
                        "requests_put",
                    ],
                    "optional_metrics": ["data_retrieval", "data_transfer"],
                    "thresholds": {
                        "ia_access_threshold": 30,  # days
                        "glacier_access_threshold": 90,  # days
                        "deep_archive_threshold": 180,  # days
                    },
                    "analysis_window_days": 90,
                },
                "base_prompt": "You are an expert AWS cost optimization specialist focusing on S3 storage.",
                "service_specific_prompt": """
Analyze S3 buckets for cost optimization opportunities. Consider:
1. Storage class optimization (Standard -> IA -> Glacier -> Deep Archive)
2. Lifecycle policies for automatic transitions
3. Access patterns and retrieval costs
4. Incomplete multipart uploads
5. Duplicate or redundant data

Provide specific recommendations with storage classes and lifecycle rules.
""",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7,
            },
            ServiceType.AWS.RDS: {
                "agent_id": "rds_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.PURCHASING_OPTION.value,
                        RecommendationType.STORAGE_CLASS.value,
                    ],
                    "required_metrics": [
                        "cpu_utilization_p50",
                        "cpu_utilization_p95",
                        "memory_utilization_p50",
                    ],
                    "optional_metrics": [
                        "read_iops",
                        "write_iops",
                        "network_throughput",
                    ],
                    "thresholds": {
                        "cpu_low_threshold": 20.0,
                        "memory_low_threshold": 30.0,
                        "iops_utilization_threshold": 20.0,
                    },
                    "analysis_window_days": 30,
                },
                "base_prompt": "You are an expert AWS cost optimization specialist focusing on RDS databases.",
                "service_specific_prompt": """
Analyze RDS instances for cost optimization opportunities. Consider:
1. CPU, memory, and IOPS utilization patterns
2. Instance class optimization
3. Purchase options (On-Demand vs Reserved)
4. Storage type optimization (GP2 vs GP3 vs Provisioned IOPS)
5. Multi-AZ vs Single-AZ based on requirements

Provide specific recommendations with instance classes and cost calculations.
""",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7,
            },
            ServiceType.AWS.LAMBDA: {
                "agent_id": "lambda_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value
                    ],
                    "required_metrics": [
                        "invocation_count",
                        "duration_avg",
                        "memory_used",
                    ],
                    "optional_metrics": ["errors", "throttles"],
                    "thresholds": {
                        "memory_utilization_threshold": 60.0,
                        "duration_efficiency_threshold": 80.0,
                    },
                    "analysis_window_days": 30,
                },
                "base_prompt": "You are an expert AWS cost optimization specialist focusing on Lambda functions.",
                "service_specific_prompt": """
Analyze Lambda functions for cost optimization opportunities. Consider:
1. Memory allocation vs actual usage
2. Execution duration patterns
3. Invocation frequency and patterns
4. Cold start optimization
5. Cost per invocation analysis

Provide specific recommendations with memory settings and architectural improvements.
""",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7,
            },
        }

        # For services not explicitly defined, create a default agent configuration
        config = service_configs.get(service, default_config)

        return config

    def get_service_config(
        self, service: Union[ServiceType.AWS, ServiceType.Azure, ServiceType.GCP, str]
    ) -> Optional[ServiceAgentConfig]:
        """Get configuration for a service"""
        if isinstance(service, str):
            # Convert string to ServiceType using from_string method
            service_obj = ServiceType.from_string(service)
            if not service_obj:
                return None
        else:
            service_obj = service

        return self.service_configs.get(service_obj)
