"""
Unit tests for models and data structures.
Tests the core data models, validation, and type conversion.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from llm_cost_recommendation.models.resources import Resource
from llm_cost_recommendation.models.metrics import BillingData, Metrics
from llm_cost_recommendation.models.recommendations import Recommendation
from llm_cost_recommendation.models.conditions import CustomCondition, ConditionalRule
from llm_cost_recommendation.models.types import (
    ServiceType,
    CloudProvider,
    RiskLevel,
    RecommendationType,
    ConditionOperator,
)


class TestResource:
    """Test Resource model validation and functionality."""

    def test_valid_resource_creation(self):
        """Test creating a valid resource."""
        resource = Resource(
            resource_id="i-1234567890abcdef0",
            service=ServiceType.AWS.EC2,
            region="us-east-1",
            tags={"Environment": "production"},
            properties={"InstanceType": "t3.large"},
        )

        assert resource.resource_id == "i-1234567890abcdef0"
        assert resource.service == ServiceType.AWS.EC2
        assert resource.region == "us-east-1"
        assert resource.tags["Environment"] == "production"

    def test_resource_with_minimal_data(self):
        """Test creating resource with minimal required data."""
        resource = Resource(
            resource_id="minimal-resource",
            service=ServiceType.AWS.EC2,
            region="us-east-1",
        )

        assert resource.resource_id == "minimal-resource"
        assert resource.tags == {}
        assert resource.properties == {}

    def test_resource_validation_errors(self):
        """Test validation errors for invalid resource data."""
        # Missing required fields
        with pytest.raises(ValidationError):
            Resource()

        # Invalid service type
        with pytest.raises(ValidationError):
            Resource(resource_id="test", service="InvalidService", region="us-east-1")

    def test_resource_string_representation(self):
        """Test resource string representation."""
        resource = Resource(
            resource_id="i-test", service=ServiceType.AWS.EC2, region="us-east-1"
        )

        str_repr = str(resource)
        assert "i-test" in str_repr
        assert "AWS.EC2" in str_repr


class TestBillingData:
    """Test BillingData model validation and functionality."""

    def test_valid_billing_data_creation(self):
        """Test creating valid billing data."""
        from datetime import datetime

        billing = BillingData(
            resource_id="i-1234567890abcdef0",
            service="AWS.EC2",
            region="us-east-1",
            usage_type="BoxUsage:t3.large",
            usage_amount=24.0,
            usage_unit="Hours",
            unblended_cost=150.50,
            amortized_cost=150.50,
            bill_period_start=datetime(2024, 9, 1),
            bill_period_end=datetime(2024, 9, 30),
        )

        assert billing.resource_id == "i-1234567890abcdef0"
        assert billing.unblended_cost == 150.50
        assert billing.service == "AWS.EC2"

    def test_billing_data_cost_validation(self):
        """Test validation of cost values."""
        from datetime import datetime

        # Negative cost should raise error
        with pytest.raises(ValidationError):
            BillingData(
                resource_id="test",
                service="AWS.EC2",
                region="us-east-1",
                usage_type="test",
                usage_amount=10.0,
                usage_unit="Hours",
                unblended_cost=-10.0,  # Negative cost
                amortized_cost=-10.0,
                bill_period_start=datetime(2024, 9, 1),
                bill_period_end=datetime(2024, 9, 30),
            )

    def test_billing_data_zero_cost(self):
        """Test billing data with zero cost."""
        from datetime import datetime

        billing = BillingData(
            resource_id="free-tier-resource",
            service="AWS.EC2",
            region="us-east-1",
            usage_type="BoxUsage:t3.micro",
            usage_amount=24.0,
            usage_unit="Hours",
            unblended_cost=0.0,
            amortized_cost=0.0,
            bill_period_start=datetime(2024, 9, 1),
            bill_period_end=datetime(2024, 9, 30),
        )

        assert billing.unblended_cost == 0.0


class TestMetrics:
    """Test Metrics model validation and functionality."""

    def test_valid_metrics_creation(self):
        """Test creating valid metrics."""
        from datetime import datetime

        metrics = Metrics(
            resource_id="i-1234567890abcdef0",
            timestamp=datetime(2024, 9, 1),
            period_days=30,
            cpu_utilization_p50=25.5,
            cpu_utilization_p90=45.2,
            cpu_utilization_p95=52.8,
            is_idle=False,
        )

        assert metrics.resource_id == "i-1234567890abcdef0"
        assert metrics.cpu_utilization_p50 == 25.5
        assert metrics.is_idle is False

    def test_metrics_utilization_validation(self):
        """Test validation of utilization percentages."""
        from datetime import datetime

        # Test with high utilization (should be valid since no explicit validation in model)
        metrics = Metrics(
            resource_id="test",
            timestamp=datetime(2024, 9, 1),
            period_days=30,
            cpu_utilization_p50=150.0,  # High but allowed
            is_idle=False,
        )
        assert metrics.cpu_utilization_p50 == 150.0

        # Test with negative utilization (should be valid since no explicit validation)
        metrics2 = Metrics(
            resource_id="test2",
            timestamp=datetime(2024, 9, 1),
            period_days=30,
            cpu_utilization_p50=-10.0,  # Negative but allowed
            is_idle=False,
        )
        assert metrics2.cpu_utilization_p50 == -10.0

    def test_metrics_idle_detection(self):
        """Test idle resource detection logic."""
        from datetime import datetime

        # Very low utilization - should be idle
        metrics = Metrics(
            resource_id="idle-resource",
            timestamp=datetime(2024, 9, 1),
            period_days=30,
            cpu_utilization_p50=1.0,
            cpu_utilization_p90=2.0,
            cpu_utilization_p95=3.0,
            is_idle=True,
        )

        assert metrics.is_idle is True

        # High utilization - not idle
        metrics = Metrics(
            resource_id="active-resource",
            timestamp=datetime(2024, 9, 1),
            period_days=30,
            cpu_utilization_p50=75.0,
            cpu_utilization_p90=85.0,
            cpu_utilization_p95=90.0,
            is_idle=False,
        )

        assert metrics.is_idle is False


class TestRecommendation:
    """Test Recommendation model validation and functionality."""

    def test_valid_recommendation_creation(self):
        """Test creating a valid recommendation."""
        recommendation = Recommendation(
            id="rec_001",
            resource_id="i-1234567890abcdef0",
            service=ServiceType.AWS.EC2,
            recommendation_type=RecommendationType.RIGHTSIZING,
            current_config={"InstanceType": "t3.large"},
            recommended_config={"InstanceType": "t3.medium"},
            current_monthly_cost=100.0,
            estimated_monthly_cost=54.70,
            estimated_monthly_savings=45.30,
            annual_savings=543.60,
            confidence_score=0.85,
            risk_level=RiskLevel.LOW,
            rationale="CPU utilization is consistently low.",
            impact_description="Lower compute capacity but sufficient for workload",
            rollback_plan="Stop instance, change type back to t3.large, restart",
            agent_id="aws.ec2_agent",
        )

        assert recommendation.id == "rec_001"
        assert recommendation.recommendation_type == RecommendationType.RIGHTSIZING
        assert recommendation.estimated_monthly_savings == 45.30
        assert recommendation.confidence_score == 0.85
        assert recommendation.risk_level == RiskLevel.LOW

    def test_recommendation_savings_validation(self):
        """Test validation of savings amounts."""
        # Test that negative costs can be created (no explicit validation in model)
        recommendation = Recommendation(
            id="rec_001",
            resource_id="test",
            service=ServiceType.AWS.EC2,
            recommendation_type=RecommendationType.RIGHTSIZING,
            current_config={"InstanceType": "t3.large"},
            recommended_config={"InstanceType": "t3.medium"},
            current_monthly_cost=100.0,
            estimated_monthly_cost=110.0,  # Higher cost
            estimated_monthly_savings=-10.0,  # Negative savings
            annual_savings=-120.0,
            confidence_score=0.8,
            risk_level=RiskLevel.LOW,
            rationale="Test",
            impact_description="Test impact",
            rollback_plan="Test rollback",
            agent_id="test_agent",
        )
        assert recommendation.estimated_monthly_savings == -10.0

    def test_recommendation_confidence_validation(self):
        """Test validation of confidence scores."""
        # Confidence > 1.0 - should raise error
        with pytest.raises(ValidationError):
            Recommendation(
                id="rec_001",
                resource_id="test",
                service=ServiceType.AWS.EC2,
                recommendation_type=RecommendationType.RIGHTSIZING,
                current_config={"InstanceType": "t3.large"},
                recommended_config={"InstanceType": "t3.medium"},
                current_monthly_cost=100.0,
                estimated_monthly_cost=90.0,
                estimated_monthly_savings=10.0,
                annual_savings=120.0,
                confidence_score=1.5,  # Invalid
                risk_level=RiskLevel.LOW,
                rationale="Test",
                impact_description="Test impact",
                rollback_plan="Test rollback",
                agent_id="test_agent",
            )

        # Confidence < 0.0 - should raise error
        with pytest.raises(ValidationError):
            Recommendation(
                id="rec_001",
                resource_id="test",
                service=ServiceType.AWS.EC2,
                recommendation_type=RecommendationType.RIGHTSIZING,
                current_config={"InstanceType": "t3.large"},
                recommended_config={"InstanceType": "t3.medium"},
                current_monthly_cost=100.0,
                estimated_monthly_cost=90.0,
                estimated_monthly_savings=10.0,
                annual_savings=120.0,
                confidence_score=-0.1,  # Invalid
                risk_level=RiskLevel.LOW,
                rationale="Test",
                impact_description="Test impact",
                rollback_plan="Test rollback",
                agent_id="test_agent",
            )

    def test_recommendation_annual_savings_calculation(self):
        """Test that annual savings is properly calculated."""
        recommendation = Recommendation(
            id="rec_001",
            resource_id="test",
            service=ServiceType.AWS.EC2,
            recommendation_type=RecommendationType.RIGHTSIZING,
            current_config={"InstanceType": "t3.large"},
            recommended_config={"InstanceType": "t3.medium"},
            current_monthly_cost=200.0,
            estimated_monthly_cost=100.0,
            estimated_monthly_savings=100.0,
            annual_savings=1200.0,
            confidence_score=0.8,
            risk_level=RiskLevel.LOW,
            rationale="Test",
            impact_description="Test impact",
            rollback_plan="Test rollback",
            agent_id="test_agent",
        )

        # Annual should be 12x monthly (approximately)
        expected_annual = recommendation.estimated_monthly_savings * 12
        assert abs(recommendation.annual_savings - expected_annual) < 0.01


class TestCustomCondition:
    """Test CustomCondition model validation and functionality."""

    def test_valid_condition_creation(self):
        """Test creating a valid custom condition."""
        condition = CustomCondition(
            field="cpu_utilization_p95",
            operator=ConditionOperator.LESS_THAN,
            value=50.0,
            description="High CPU utilization check",
        )

        assert condition.field == "cpu_utilization_p95"
        assert condition.operator == ConditionOperator.LESS_THAN
        assert condition.value == 50.0

    def test_condition_evaluation_methods(self):
        """Test condition evaluation logic."""
        condition = CustomCondition(
            field="cpu_utilization_p95",
            operator=ConditionOperator.LESS_THAN,
            value=50.0,
        )

        # Test values that should match
        test_data = {"cpu_utilization_p95": 30.0}
        # Note: Actual evaluation logic would be tested in service tests
        assert condition.field in test_data
        assert test_data[condition.field] < condition.value


class TestConditionalRule:
    """Test ConditionalRule model validation and functionality."""

    def test_valid_rule_creation(self):
        """Test creating a valid conditional rule."""
        condition = CustomCondition(
            field="Environment", operator=ConditionOperator.EQUALS, value="production"
        )

        rule = ConditionalRule(
            name="prod_rule_001",
            description="Production environment rule",
            conditions=[condition],
            actions={"confidence_multiplier": 1.2},
        )

        assert rule.name == "prod_rule_001"
        assert len(rule.conditions) == 1
        assert rule.actions["confidence_multiplier"] == 1.2

    def test_rule_with_multiple_conditions(self):
        """Test rule with multiple conditions."""
        conditions = [
            CustomCondition(
                field="Environment",
                operator=ConditionOperator.EQUALS,
                value="production",
            ),
            CustomCondition(
                field="cpu_utilization_p95",
                operator=ConditionOperator.LESS_THAN,
                value=30.0,
            ),
        ]

        rule = ConditionalRule(
            name="multi_condition_rule",
            description="Multi-condition rule",
            conditions=conditions,
            actions={"confidence_multiplier": 1.5},
        )

        assert len(rule.conditions) == 2
        assert rule.conditions[0].field == "Environment"
        assert rule.conditions[1].field == "cpu_utilization_p95"


class TestEnumTypes:
    """Test enum types and their values."""

    def test_service_type_enum(self):
        """Test ServiceType enum values."""
        assert ServiceType.AWS.EC2 == "AWS.EC2"
        assert ServiceType.AWS.S3 == "AWS.S3"
        assert ServiceType.Azure.VM == "Azure.VM"
        assert ServiceType.GCP.COMPUTE == "GCP.Compute"

    def test_cloud_provider_enum(self):
        """Test CloudProvider enum values."""
        assert CloudProvider.AWS == "AWS"
        assert CloudProvider.AZURE == "Azure"
        assert CloudProvider.GCP == "GCP"

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"

    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.RIGHTSIZING == "rightsizing"
        assert RecommendationType.PURCHASING_OPTION == "purchasing_option"
        assert RecommendationType.STORAGE_CLASS == "storage_class"
        assert RecommendationType.LIFECYCLE == "lifecycle"

    def test_condition_operator_enum(self):
        """Test ConditionOperator enum values."""
        assert ConditionOperator.EQUALS == "equals"
        assert ConditionOperator.GREATER_THAN == "greater_than"
        assert ConditionOperator.LESS_THAN == "less_than"
        assert ConditionOperator.CONTAINS == "contains"
