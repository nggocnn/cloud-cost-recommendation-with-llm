"""
Pytest configuration and shared fixtures for the LLM Cost Recommendation System.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

# Test imports
from llm_cost_recommendation.services.config import ConfigManager
from llm_cost_recommendation.services.llm import LLMService
from llm_cost_recommendation.services.ingestion import DataIngestionService
from llm_cost_recommendation.agents.coordinator import CoordinatorAgent
from llm_cost_recommendation.models import Resource, BillingData, Metrics
from llm_cost_recommendation.models.types import (
    ServiceType,
    CloudProvider,
    RiskLevel,
    RecommendationType,
)
from llm_cost_recommendation.models.recommendations import (
    Recommendation,
    RecommendationReport,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dir(temp_dir):
    """Create a sample configuration directory structure."""
    config_dir = temp_dir / "config"

    # Create directory structure
    (config_dir / "agents" / "aws").mkdir(parents=True, exist_ok=True)
    (config_dir / "agents" / "azure").mkdir(parents=True, exist_ok=True)
    (config_dir / "agents" / "gcp").mkdir(parents=True, exist_ok=True)
    (config_dir / "agents" / "default").mkdir(parents=True, exist_ok=True)
    (config_dir / "global").mkdir(parents=True, exist_ok=True)

    # Create coordinator config
    coordinator_config = {
        "enabled_services": ["AWS.EC2", "AWS.S3", "AWS.RDS"],
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
            "high_cost": {"min": 1000, "max": float("inf"), "batch_adjustment": -2},
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

    with open(config_dir / "global" / "coordinator.yaml", "w") as f:
        import yaml

        yaml.dump(coordinator_config, f)

    # Create sample agent configs
    ec2_config = {
        "agent_id": "aws.ec2_agent",
        "service": "AWS.EC2",
        "enabled": True,
        "base_prompt": "You are a cloud cost optimization expert.",
        "service_specific_prompt": "Analyze EC2 instance for cost optimization opportunities.",
        "capability": {
            "service": "AWS.EC2",
            "supported_recommendation_types": ["rightsizing", "purchasing_option"],
            "required_metrics": ["cpu_utilization_p95", "memory_utilization_p95"],
            "optional_metrics": ["network_in", "network_out"],
            "thresholds": {"cpu_threshold": 80.0, "memory_threshold": 80.0},
            "analysis_window_days": 30,
        },
    }

    with open(config_dir / "agents" / "aws" / "ec2.yaml", "w") as f:
        import yaml

        yaml.dump(ec2_config, f)

    # Create S3 agent config
    s3_config = {
        "agent_id": "aws.s3_agent",
        "service": "AWS.S3",
        "enabled": True,
        "base_prompt": "You are a cloud cost optimization expert.",
        "service_specific_prompt": "Analyze S3 bucket for cost optimization opportunities.",
        "capability": {
            "service": "AWS.S3",
            "supported_recommendation_types": ["lifecycle", "storage_class"],
            "required_metrics": [],
            "optional_metrics": ["storage_size", "request_count"],
            "thresholds": {"access_frequency": 0.1},
            "analysis_window_days": 30,
        },
    }

    with open(config_dir / "agents" / "aws" / "s3.yaml", "w") as f:
        import yaml

        yaml.dump(s3_config, f)

    # Create RDS agent config
    rds_config = {
        "agent_id": "aws.rds_agent",
        "service": "AWS.RDS",
        "enabled": True,
        "base_prompt": "You are a cloud cost optimization expert.",
        "service_specific_prompt": "Analyze RDS instance for cost optimization opportunities.",
        "capability": {
            "service": "AWS.RDS",
            "supported_recommendation_types": ["rightsizing", "purchasing_option"],
            "required_metrics": ["cpu_utilization_p95"],
            "optional_metrics": ["memory_utilization_p95", "connection_count"],
            "thresholds": {"cpu_threshold": 80.0, "memory_threshold": 80.0},
            "analysis_window_days": 30,
        },
    }

    with open(config_dir / "agents" / "aws" / "rds.yaml", "w") as f:
        import yaml

        yaml.dump(rds_config, f)

    # Create DEFAULT agent config for fallback coverage
    default_config = {
        "agent_id": "default_agent",
        "service": "DEFAULT",
        "enabled": True,
        "base_prompt": "You are a cloud cost optimization expert.",
        "service_specific_prompt": "Analyze resource for generic cost optimization opportunities.",
        "capability": {
            "service": "DEFAULT",
            "supported_recommendation_types": ["rightsizing"],
            "required_metrics": [],
            "optional_metrics": ["cpu_utilization_p95", "memory_utilization_p95"],
            "thresholds": {"cpu_threshold": 80.0, "memory_threshold": 80.0},
            "analysis_window_days": 30,
        },
    }

    with open(config_dir / "agents" / "default" / "agent.yaml", "w") as f:
        import yaml

        yaml.dump(default_config, f)

    return config_dir


@pytest.fixture
def sample_data_dir(temp_dir):
    """Create a sample data directory structure."""
    data_dir = temp_dir / "data"

    # Create directory structure
    (data_dir / "billing").mkdir(parents=True, exist_ok=True)
    (data_dir / "inventory").mkdir(parents=True, exist_ok=True)
    (data_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # Create sample billing data
    billing_data = [
        {
            "resource_id": "i-1234567890abcdef0",
            "service": "AWS.EC2",
            "region": "us-east-1",
            "unblended_cost": 150.50,
            "usage_type": "BoxUsage:t3.large",
            "period": "2024-09-01",
            "bill_period_start": "2024-09-01T00:00:00Z",
            "bill_period_end": "2024-09-30T23:59:59Z",
            "usage_amount": 24.0,
            "usage_unit": "Hrs",
            "amortized_cost": 150.50,
        }
    ]

    with open(data_dir / "billing" / "sample_billing.csv", "w") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=billing_data[0].keys())
        writer.writeheader()
        writer.writerows(billing_data)

    # Create sample inventory data
    inventory_data = [
        {
            "resource_id": "i-1234567890abcdef0",
            "service": "AWS.EC2",
            "cloud_provider": "AWS",
            "region": "us-east-1",
            "account_id": "123456789012",
            "tags": {"Environment": "production", "Team": "backend"},
            "properties": {"InstanceType": "t3.large", "State": "running"},
        }
    ]

    with open(data_dir / "inventory" / "sample_inventory.json", "w") as f:
        json.dump(inventory_data, f, indent=2)

    # Create sample metrics data
    metrics_data = [
        {
            "resource_id": "i-1234567890abcdef0",
            "timestamp": "2024-09-01T00:00:00Z",
            "period_days": 30,
            "cpu_utilization_p50": 25.5,
            "cpu_utilization_p90": 45.2,
            "cpu_utilization_p95": 52.8,
            "memory_utilization_p50": 30.3,
            "memory_utilization_p90": 55.7,
            "memory_utilization_p95": 62.1,
            "is_idle": False,
        }
    ]

    with open(data_dir / "metrics" / "sample_metrics.csv", "w") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
        writer.writeheader()
        writer.writerows(metrics_data)

    return data_dir


@pytest.fixture
def mock_llm_service():
    """Create a mocked LLM service."""
    llm_service = MagicMock(spec=LLMService)

    # Mock the generate_recommendation method to return a realistic LLMResponse
    async def mock_generate_recommendation(*args, **kwargs):
        from llm_cost_recommendation.services.llm import LLMResponse

        return LLMResponse(
            content=json.dumps(
                {
                    "recommendations": [
                        {
                            "recommendation_type": "rightsizing",
                            "current_config": {
                                "instance_type": "t3.large",
                                "vcpu": 2,
                                "memory_gb": 8,
                            },
                            "recommended_config": {
                                "instance_type": "t3.medium",
                                "vcpu": 2,
                                "memory_gb": 4,
                            },
                            "current_monthly_cost": 100.00,
                            "estimated_monthly_cost": 54.70,
                            "estimated_monthly_savings": 45.30,
                            "annual_savings": 543.60,
                            "confidence_score": 0.85,
                            "risk_level": "LOW",
                            "impact_description": "Minimal performance impact expected due to low memory usage patterns.",
                            "rollback_plan": "Monitor performance for 24 hours, rollback to t3.large if any issues detected.",
                            "rationale": "Current CPU utilization averages 25% and memory usage is consistently below 60%, indicating over-provisioning.",
                            "implementation_steps": [
                                "Stop the instance during maintenance window",
                                "Change instance type to t3.medium",
                                "Start the instance and verify application functionality",
                            ],
                            "prerequisites": [
                                "Verify maintenance window",
                                "Notify stakeholders",
                            ],
                        }
                    ]
                }
            ),
            model="test-model",
            response_time_ms=100.0,
        )

    # Mock both methods that might be called
    llm_service.generate_recommendation = AsyncMock(
        side_effect=mock_generate_recommendation
    )
    llm_service.analyze = AsyncMock(side_effect=mock_generate_recommendation)
    return llm_service


@pytest.fixture
def sample_resource():
    """Create a sample resource for testing."""
    return Resource(
        resource_id="i-1234567890abcdef0",
        service=ServiceType.AWS.EC2,
        cloud_provider=CloudProvider.AWS,
        region="us-east-1",
        account_id="123456789012",
        tags={"Environment": "production", "Team": "backend"},
        properties={"InstanceType": "t3.large", "State": "running"},
    )


@pytest.fixture
def sample_billing_data():
    """Create sample billing data for testing."""
    return [
        BillingData(
            resource_id="i-1234567890abcdef0",
            service="AWS.EC2",
            region="us-east-1",
            unblended_cost=150.50,
            usage_type="BoxUsage:t3.large",
            period="2024-09-01",
            bill_period_start="2024-09-01T00:00:00Z",
            bill_period_end="2024-09-30T23:59:59Z",
            usage_amount=24.0,
            usage_unit="Hrs",
            amortized_cost=150.50,
        )
    ]


@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return [
        Metrics(
            resource_id="i-1234567890abcdef0",
            timestamp="2024-09-01T00:00:00Z",
            period_days=30,
            cpu_utilization_p50=25.5,
            cpu_utilization_p90=45.2,
            cpu_utilization_p95=52.8,
            memory_utilization_p50=30.3,
            memory_utilization_p90=55.7,
            memory_utilization_p95=62.1,
            is_idle=False,
        )
    ]


@pytest.fixture
def sample_recommendation():
    """Create a sample recommendation for testing."""
    return Recommendation(
        id="rec_001",
        resource_id="i-1234567890abcdef0",
        service="AWS.EC2",
        recommendation_type=RecommendationType.RIGHTSIZING,
        current_config={"instance_type": "t3.large"},
        recommended_config={"instance_type": "t3.medium"},
        current_monthly_cost=150.50,
        estimated_monthly_cost=105.20,
        estimated_monthly_savings=45.30,
        annual_savings=543.60,
        risk_level=RiskLevel.LOW,
        impact_description="Minimal impact - instance will be temporarily unavailable during resize",
        rollback_plan="Change instance type back to t3.large if performance issues occur",
        rationale="Current CPU utilization averages 25%, indicating over-provisioning.",
        confidence_score=0.85,
        agent_id="aws.ec2_agent",
        implementation_steps=[
            "Stop the instance during maintenance window",
            "Change instance type to t3.medium",
            "Start the instance and verify application functionality",
        ],
    )


@pytest.fixture
def config_manager(sample_config_dir):
    """Create a ConfigManager instance with sample configuration."""
    return ConfigManager(str(sample_config_dir))


@pytest.fixture
def data_service(sample_data_dir):
    """Create a DataIngestionService instance with sample data."""
    return DataIngestionService(str(sample_data_dir))


@pytest.fixture
async def coordinator_agent(config_manager, mock_llm_service):
    """Create a CoordinatorAgent instance for testing."""
    return CoordinatorAgent(config_manager, mock_llm_service)


@pytest.fixture
def api_test_data():
    """Create test data for API endpoints."""
    return {
        "resources": [
            {
                "resource_id": "i-test-12345",
                "service": "AWS.EC2",
                "cloud_provider": "AWS",
                "region": "us-east-1",
                "account_id": "123456789012",
                "tags": {"Environment": "test"},
                "properties": {"InstanceType": "t3.large", "State": "running"},
            }
        ],
        "billing": [
            {
                "resource_id": "i-test-12345",
                "service": "AWS.EC2",
                "region": "us-east-1",
                "usage_type": "BoxUsage:t3.large",
                "usage_amount": 24.0,
                "usage_unit": "Hours",
                "unblended_cost": 100.00,
                "amortized_cost": 100.00,
                "bill_period_start": "2024-09-01T00:00:00Z",
                "bill_period_end": "2024-09-30T23:59:59Z",
            }
        ],
        "metrics": [
            {
                "resource_id": "i-test-12345",
                "timestamp": "2024-09-01T00:00:00Z",
                "period_days": 30,
                "cpu_utilization_p50": 20.0,
                "cpu_utilization_p90": 40.0,
                "cpu_utilization_p95": 50.0,
                "is_idle": False,
            }
        ],
        "individual_processing": False,
        "max_recommendations": 5,
    }


@pytest.fixture
def minimal_valid_request():
    """Create minimal valid API request for testing."""
    return {
        "resources": [
            {
                "resource_id": "test-resource-001",
                "service": "AWS.EC2",
                "cloud_provider": "AWS",
                "region": "us-east-1",
                "tags": {},
                "properties": {},
            }
        ]
    }


# Helper functions for testing
def assert_valid_recommendation(recommendation: Dict[str, Any]):
    """Assert that a recommendation has all required fields with valid values."""
    required_fields = [
        "recommendation_id",
        "resource_id",
        "type",
        "monthly_savings",
        "annual_savings",
        "confidence",
        "risk_level",
        "rationale",
    ]

    for field in required_fields:
        assert field in recommendation, f"Missing required field: {field}"

    # Validate field types and ranges
    assert isinstance(recommendation["monthly_savings"], (int, float))
    assert isinstance(recommendation["annual_savings"], (int, float))
    assert 0 <= recommendation["confidence"] <= 1
    assert recommendation["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def assert_valid_report(report: Dict[str, Any]):
    """Assert that a report has all required fields with valid values."""
    required_fields = ["report_id", "timestamp", "summary", "recommendations"]

    for field in required_fields:
        assert field in report, f"Missing required field: {field}"

    assert isinstance(report["recommendations"], list)

    # Validate each recommendation in the report
    for rec in report["recommendations"]:
        assert_valid_recommendation(rec)
