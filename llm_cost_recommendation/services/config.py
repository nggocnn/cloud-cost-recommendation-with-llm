"""
Configuration management for the LLM cost recommendation system.
"""
import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

from ..models import ServiceAgentConfig, CoordinatorConfig, ServiceType, RecommendationType, AgentCapability


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
        self.coordinator_config = self._load_coordinator_config()
        self.service_configs = self._load_service_configs()
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from environment variables"""
        return LLMConfig(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
        )
    
    def _load_coordinator_config(self) -> CoordinatorConfig:
        """Load coordinator configuration"""
        config_file = self.config_dir / "coordinator.yaml"
        
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
                "include_low_impact": False
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return CoordinatorConfig(**config_data)
    
    def _load_service_configs(self) -> Dict[ServiceType, ServiceAgentConfig]:
        """Load service agent configurations"""
        configs = {}
        
        for service in ServiceType:
            config_file = self.config_dir / f"{service.value.lower()}_agent.yaml"
            
            if not config_file.exists():
                # Create default configuration for this service
                default_config = self._create_default_service_config(service)
                
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert to ServiceAgentConfig
            capability_data = config_data.pop('capability')
            capability = AgentCapability(**capability_data)
            
            config_data['capability'] = capability
            configs[service] = ServiceAgentConfig(**config_data)
        
        return configs
    
    def _create_default_service_config(self, service: ServiceType) -> Dict:
        """Create default configuration for a service"""
        
        # Define service-specific configurations
        service_configs = {
            ServiceType.EC2: {
                "agent_id": "ec2_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.PURCHASING_OPTION.value,
                        RecommendationType.IDLE_RESOURCE.value
                    ],
                    "required_metrics": ["cpu_utilization_p50", "cpu_utilization_p95", "memory_utilization_p50"],
                    "optional_metrics": ["network_in", "network_out"],
                    "thresholds": {
                        "cpu_idle_threshold": 5.0,
                        "cpu_low_threshold": 20.0,
                        "memory_low_threshold": 30.0,
                        "uptime_threshold": 0.95
                    },
                    "analysis_window_days": 30
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
                "confidence_threshold": 0.7
            },
            
            ServiceType.EBS: {
                "agent_id": "ebs_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.STORAGE_CLASS.value,
                        RecommendationType.IDLE_RESOURCE.value
                    ],
                    "required_metrics": ["iops_read", "iops_write", "throughput_read", "throughput_write"],
                    "optional_metrics": ["burst_credits"],
                    "thresholds": {
                        "iops_utilization_threshold": 10.0,
                        "throughput_utilization_threshold": 10.0,
                        "idle_threshold": 1.0
                    },
                    "analysis_window_days": 30
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
                "confidence_threshold": 0.7
            },
            
            ServiceType.S3: {
                "agent_id": "s3_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.STORAGE_CLASS.value,
                        RecommendationType.LIFECYCLE.value
                    ],
                    "required_metrics": ["storage_used", "requests_get", "requests_put"],
                    "optional_metrics": ["data_retrieval", "data_transfer"],
                    "thresholds": {
                        "ia_access_threshold": 30,  # days
                        "glacier_access_threshold": 90,  # days
                        "deep_archive_threshold": 180  # days
                    },
                    "analysis_window_days": 90
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
                "confidence_threshold": 0.7
            },
            
            ServiceType.RDS: {
                "agent_id": "rds_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value,
                        RecommendationType.PURCHASING_OPTION.value,
                        RecommendationType.STORAGE_CLASS.value
                    ],
                    "required_metrics": ["cpu_utilization_p50", "cpu_utilization_p95", "memory_utilization_p50"],
                    "optional_metrics": ["read_iops", "write_iops", "network_throughput"],
                    "thresholds": {
                        "cpu_low_threshold": 20.0,
                        "memory_low_threshold": 30.0,
                        "iops_utilization_threshold": 20.0
                    },
                    "analysis_window_days": 30
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
                "confidence_threshold": 0.7
            },
            
            ServiceType.LAMBDA: {
                "agent_id": "lambda_agent",
                "service": service.value,
                "enabled": True,
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [
                        RecommendationType.RIGHTSIZING.value
                    ],
                    "required_metrics": ["invocation_count", "duration_avg", "memory_used"],
                    "optional_metrics": ["errors", "throttles"],
                    "thresholds": {
                        "memory_utilization_threshold": 60.0,
                        "duration_efficiency_threshold": 80.0
                    },
                    "analysis_window_days": 30
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
                "confidence_threshold": 0.7
            }
        }
        
        # For services not explicitly defined, create a basic configuration
        if service not in service_configs:
            return {
                "agent_id": f"{service.value.lower()}_agent",
                "service": service.value,
                "enabled": False,  # Disabled by default for undefined services
                "capability": {
                    "service": service.value,
                    "supported_recommendation_types": [RecommendationType.RIGHTSIZING.value],
                    "required_metrics": [],
                    "optional_metrics": [],
                    "thresholds": {},
                    "analysis_window_days": 30
                },
                "base_prompt": f"You are an expert AWS cost optimization specialist focusing on {service.value}.",
                "service_specific_prompt": f"Analyze {service.value} resources for cost optimization opportunities.",
                "temperature": 0.1,
                "max_tokens": 2000,
                "min_cost_threshold": 1.0,
                "confidence_threshold": 0.7
            }
        
        return service_configs[service]
    
    def get_service_config(self, service: ServiceType) -> Optional[ServiceAgentConfig]:
        """Get configuration for a specific service"""
        return self.service_configs.get(service)
    
    def get_enabled_services(self) -> List[ServiceType]:
        """Get list of enabled services"""
        return [service for service, config in self.service_configs.items() if config.enabled]
    
    def update_service_config(self, service: ServiceType, config: ServiceAgentConfig):
        """Update configuration for a service"""
        self.service_configs[service] = config
        
        # Save to file
        config_file = self.config_dir / f"{service.value.lower()}_agent.yaml"
        config_dict = config.dict()
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
