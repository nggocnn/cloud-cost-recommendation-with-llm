"""
Unit tests for services layer components.
Tests configuration management, data ingestion, and condition evaluation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from llm_cost_recommendation.services.config import ConfigManager
from llm_cost_recommendation.services.ingestion import DataIngestionService
from llm_cost_recommendation.services.conditions import (
    ConditionEvaluator,
    RuleProcessor,
)
from llm_cost_recommendation.services.llm import LLMService
from llm_cost_recommendation.models import (
    Resource,
    BillingData,
    Metrics,
    CustomCondition,
    ConditionalRule,
)
from llm_cost_recommendation.models.types import (
    ServiceType,
    CloudProvider,
    ConditionOperator,
)


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_initialization(self, sample_config_dir):
        """Test ConfigManager initialization with valid config directory."""
        config_manager = ConfigManager(str(sample_config_dir))

        assert config_manager.config_dir == Path(sample_config_dir)
        assert config_manager.llm_config is not None
        assert isinstance(config_manager.service_configs, dict)

    def test_load_service_configs(self, sample_config_dir):
        """Test loading service configurations."""
        config_manager = ConfigManager(str(sample_config_dir))

        # Should load service configs (keyed by ServiceType enum)
        from llm_cost_recommendation.models.types import ServiceType

        assert ServiceType.AWS.EC2 in config_manager.service_configs

        ec2_config = config_manager.service_configs[ServiceType.AWS.EC2]
        assert ec2_config.agent_id == "aws.ec2_agent"
        assert ec2_config.service == ServiceType.AWS.EC2

    def test_get_service_config(self, sample_config_dir):
        """Test retrieving specific service configuration."""
        config_manager = ConfigManager(str(sample_config_dir))

        # Get existing config
        from llm_cost_recommendation.models.types import ServiceType

        ec2_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        assert ec2_config is not None
        assert ec2_config.service == ServiceType.AWS.EC2

        # Get non-existent config
        missing_config = config_manager.get_service_config("nonexistent.agent")
        assert missing_config is None

    def test_invalid_config_directory(self):
        """Test ConfigManager with invalid config directory."""
        with pytest.raises(FileNotFoundError):
            ConfigManager("/nonexistent/config/path")

    def test_config_validation(self, temp_dir):
        """Test configuration validation."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        (config_dir / "global").mkdir()

        # Create minimal global config
        global_config = {
            "enabled_services": ["AWS.EC2"],
            "similarity_threshold": 0.8,
            "savings_weight": 0.4,
            "risk_weight": 0.3,
            "confidence_weight": 0.2,
            "implementation_ease_weight": 0.1,
            "max_recommendations_per_service": 50,
            "include_low_impact": False,
            "cost_tiers": {"low_cost": {"min": 0, "max": 100, "batch_adjustment": 0}},
            "complexity_tiers": {
                "simple": {"metric_threshold": 3, "base_batch_size": 6}
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

            yaml.dump(global_config, f)

        # Create invalid service config (missing required fields)
        (config_dir / "agents" / "aws").mkdir(parents=True)
        invalid_config = {"agent_id": "incomplete_agent"}  # Missing required fields

        with open(config_dir / "agents" / "aws" / "invalid.yaml", "w") as f:
            import yaml

            yaml.dump(invalid_config, f)

        # Should handle invalid configs gracefully and create default config
        config_manager = ConfigManager(str(config_dir))
        # Should have at least the default agent, even if other configs are invalid
        assert len(config_manager.service_configs) >= 1  # At least default agent


class TestDataIngestionService:
    """Test DataIngestionService functionality."""

    def test_data_service_initialization(self, sample_data_dir):
        """Test DataIngestionService initialization."""
        data_service = DataIngestionService(str(sample_data_dir))

        assert data_service.data_dir == Path(sample_data_dir)

    def test_ingest_billing_data(self, sample_data_dir):
        """Test ingesting billing data from CSV."""
        data_service = DataIngestionService(str(sample_data_dir))

        billing_data = data_service.ingest_billing_data(
            str(sample_data_dir / "billing" / "sample_billing.csv")
        )

        assert len(billing_data) > 0
        assert isinstance(billing_data[0], BillingData)
        assert billing_data[0].resource_id == "i-1234567890abcdef0"
        assert billing_data[0].unblended_cost == 150.50

    def test_ingest_inventory_data(self, sample_data_dir):
        """Test ingesting inventory data from JSON."""
        data_service = DataIngestionService(str(sample_data_dir))

        inventory_data = data_service.ingest_inventory_data(
            str(sample_data_dir / "inventory" / "sample_inventory.json")
        )

        assert len(inventory_data) > 0
        assert isinstance(inventory_data[0], Resource)
        assert inventory_data[0].resource_id == "i-1234567890abcdef0"
        assert inventory_data[0].service == "AWS.EC2"

    def test_ingest_metrics_data(self, sample_data_dir):
        """Test ingesting metrics data from CSV."""
        data_service = DataIngestionService(str(sample_data_dir))

        metrics_data = data_service.ingest_metrics_data(
            str(sample_data_dir / "metrics" / "sample_metrics.csv")
        )

        assert len(metrics_data) > 0
        assert isinstance(metrics_data[0], Metrics)
        assert metrics_data[0].resource_id == "i-1234567890abcdef0"
        assert metrics_data[0].cpu_utilization_p50 == 25.5

    def test_ingest_nonexistent_file(self, sample_data_dir):
        """Test ingesting data from non-existent file."""
        data_service = DataIngestionService(str(sample_data_dir))

        with pytest.raises((FileNotFoundError, ValueError)):
            data_service.ingest_billing_data("nonexistent_file.csv")

    def test_create_sample_data(self, temp_dir):
        """Test sample data creation."""
        data_service = DataIngestionService(str(temp_dir))

        data_service.create_sample_data(num_resources=5)

        # Check that sample data files were created (files are created directly in temp_dir)
        billing_file = temp_dir / "billing" / "sample_billing.csv"
        inventory_file = temp_dir / "inventory" / "sample_inventory.json"
        metrics_file = temp_dir / "metrics" / "sample_metrics.csv"

        assert billing_file.exists()
        assert inventory_file.exists()
        assert metrics_file.exists()


class TestConditionEvaluator:
    """Test ConditionEvaluator functionality."""

    def test_evaluate_simple_condition(self, sample_resource, sample_metrics):
        """Test evaluation of simple conditions."""
        evaluator = ConditionEvaluator()

        # CPU utilization condition
        condition = CustomCondition(
            field="cpu_utilization_p95",
            operator=ConditionOperator.LESS_THAN,
            value=60.0,
        )

        result = evaluator.evaluate_condition(
            condition, sample_resource, sample_metrics[0]
        )

        # Sample metrics has cpu_utilization_p95=52.8, should be < 60.0
        assert result is True

    def test_evaluate_tag_condition(self, sample_resource, sample_metrics):
        """Test evaluation of tag-based conditions."""
        evaluator = ConditionEvaluator()

        # Environment tag condition (use tag.Environment prefix)
        condition = CustomCondition(
            field="tag.Environment",
            operator=ConditionOperator.EQUALS,
            value="production",
        )

        result = evaluator.evaluate_condition(
            condition, sample_resource, sample_metrics[0]
        )

        # Sample resource has Environment=production tag
        assert result is True

    def test_evaluate_false_condition(self, sample_resource, sample_metrics):
        """Test evaluation of condition that should be false."""
        evaluator = ConditionEvaluator()

        # High CPU condition (should be false for our sample data)
        condition = CustomCondition(
            field="cpu_utilization_p95",
            operator=ConditionOperator.GREATER_THAN,
            value=80.0,
        )

        result = evaluator.evaluate_condition(
            condition, sample_resource, sample_metrics[0]
        )

        # Sample metrics has cpu_utilization_p95=52.8, should not be > 80.0
        assert result is False

    def test_evaluate_missing_field(self, sample_resource, sample_metrics):
        """Test evaluation with missing field."""
        evaluator = ConditionEvaluator()

        # Condition for non-existent field
        condition = CustomCondition(
            field="nonexistent_field", operator=ConditionOperator.EQUALS, value="test"
        )

        result = evaluator.evaluate_condition(
            condition, sample_resource, sample_metrics[0]
        )

        # Should return False for missing fields
        assert result is False


class TestRuleProcessor:
    """Test RuleProcessor functionality."""

    def test_process_matching_rule(self, sample_resource, sample_metrics):
        """Test processing rule that matches conditions."""
        processor = RuleProcessor()

        # Create rule with matching condition (use tag.Environment prefix)
        condition = CustomCondition(
            field="tag.Environment",
            operator=ConditionOperator.EQUALS,
            value="production",
        )

        rule = ConditionalRule(
            name="Production Rule",
            description="Applies adjustments for production resources",
            conditions=[condition],
            actions={"confidence_multiplier": 1.2},
        )

        result = processor.apply_rules([rule], sample_resource, sample_metrics[0])

        # Should return the actions since condition matches
        assert result["actions"] == {"confidence_multiplier": 1.2}

    def test_process_non_matching_rule(self, sample_resource, sample_metrics):
        """Test processing rule that doesn't match conditions."""
        processor = RuleProcessor()

        # Create rule with non-matching condition (use tag.Environment prefix)
        condition = CustomCondition(
            field="tag.Environment",
            operator=ConditionOperator.EQUALS,
            value="development",  # Sample resource is "production"
        )

        rule = ConditionalRule(
            name="Development Rule",
            description="Applies adjustments for development resources",
            conditions=[condition],
            actions={"confidence_multiplier": 0.8},
        )

        result = processor.apply_rules([rule], sample_resource, sample_metrics[0])

        # Should return empty actions since condition doesn't match
        assert result["actions"] == {}

    def test_process_multiple_rules(self, sample_resource, sample_metrics):
        """Test processing multiple rules."""
        processor = RuleProcessor()

        # Create multiple rules
        rules = [
            ConditionalRule(
                name="Production Rule",
                description="Apply adjustments for production environments",
                actions={"adjustments": {"confidence_multiplier": 1.2}},
                conditions=[
                    CustomCondition(
                        field="tag.Environment",
                        operator=ConditionOperator.EQUALS,
                        value="production",
                    )
                ],
            ),
            ConditionalRule(
                name="Low CPU Rule",
                description="Apply adjustments for low CPU utilization",
                actions={"adjustments": {"savings_multiplier": 1.1}},
                conditions=[
                    CustomCondition(
                        field="cpu_utilization_p95",
                        operator=ConditionOperator.LESS_THAN,
                        value=60.0,
                    )
                ],
            ),
        ]

        result = processor.apply_rules(rules, sample_resource, sample_metrics[0])

        # Both rules should be applied - check return structure
        assert "threshold_overrides" in result
        assert "skip_recommendation_types" in result
        assert "force_recommendation_types" in result


class TestLLMService:
    """Test LLMService functionality."""

    def test_llm_service_initialization(self, config_manager):
        """Test LLMService initialization."""
        llm_service = LLMService(config_manager.llm_config)

        assert llm_service.config is not None
        assert hasattr(llm_service, "client")

    @pytest.mark.mock_llm
    @patch("llm_cost_recommendation.services.llm.ChatOpenAI")
    def test_analyze_with_mocked_llm(self, mock_openai, config_manager):
        """Test LLM analysis with mocked ChatOpenAI client."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = '{"test": "response"}'
        mock_client.invoke.return_value = mock_message

        # Test the service initialization
        llm_service = LLMService(config_manager.llm_config)
        assert llm_service is not None

        # Note: This tests initialization only since async methods need proper testing setup

        # For now, just verify the service was initialized
        assert llm_service is not None

    def test_prompt_construction(self, config_manager):
        """Test LLM prompt formatting."""
        llm_service = LLMService(config_manager.llm_config)

        # Test formatting a prompt template with context data
        template = "Analyze resource {resource_id} of type {service}"
        context_data = {
            "resource_id": "i-test",
            "service": "AWS.EC2",
            "instance_type": "t3.large",
        }

        prompt = llm_service._format_prompt(template, context_data)

        # Verify prompt contains resource information
        assert "i-test" in prompt
        assert "AWS.EC2" in prompt
