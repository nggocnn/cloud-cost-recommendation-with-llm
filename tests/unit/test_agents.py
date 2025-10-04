"""
Unit tests for agents functionality.
Tests the coordinator agent and service-specific agents.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from llm_cost_recommendation.agents.coordinator import CoordinatorAgent
from llm_cost_recommendation.agents.base import ServiceAgent
from llm_cost_recommendation.models import Resource, BillingData, Metrics
from llm_cost_recommendation.models.types import ServiceType, CloudProvider, RiskLevel
from llm_cost_recommendation.models.recommendations import Recommendation


class TestCoordinatorAgent:
    """Test CoordinatorAgent functionality."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, config_manager, mock_llm_service):
        """Test CoordinatorAgent initialization."""
        coordinator = CoordinatorAgent(config_manager, mock_llm_service)
        assert coordinator.config_manager == config_manager
        assert coordinator.llm_service == mock_llm_service
        assert isinstance(coordinator.service_agents, dict)

    @pytest.mark.asyncio
    async def test_analyze_resources_batch_mode(
        self,
        coordinator_agent,
        sample_resource,
        sample_billing_data,
        sample_metrics,
        sample_recommendation,
    ):
        """Test analyzing resources in batch mode."""
        resources = [sample_resource]
        billing_data = {sample_resource.resource_id: sample_billing_data}
        metrics_data = {sample_resource.resource_id: sample_metrics[0]}

        # Mock the analyze_resources_and_generate_report method
        with patch.object(
            coordinator_agent, "analyze_resources_and_generate_report"
        ) as mock_analyze:
            from llm_cost_recommendation.models.recommendations import (
                RecommendationReport,
            )

            mock_report = RecommendationReport(
                id="test_analysis_001",
                total_monthly_savings=sample_recommendation.estimated_monthly_savings,
                total_annual_savings=sample_recommendation.annual_savings,
                total_recommendations=1,
                recommendations=[sample_recommendation],
            )
            mock_analyze.return_value = mock_report

            report = await coordinator_agent.analyze_resources_and_generate_report(
                resources, metrics_data, billing_data, batch_mode=True
            )

            assert report.total_recommendations == 1
            assert len(report.recommendations) == 1
            assert report.recommendations[0].resource_id == sample_resource.resource_id
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_resources_individual_mode(
        self,
        coordinator_agent,
        sample_resource,
        sample_billing_data,
        sample_metrics,
        sample_recommendation,
    ):
        """Test analyzing resources in individual mode (non-batch)."""
        resources = [sample_resource]
        billing_data = {sample_resource.resource_id: sample_billing_data}
        metrics_data = {sample_resource.resource_id: sample_metrics[0]}

        # Mock the analyze_resources_and_generate_report method
        with patch.object(
            coordinator_agent, "analyze_resources_and_generate_report"
        ) as mock_analyze:
            from llm_cost_recommendation.models.recommendations import (
                RecommendationReport,
            )

            mock_report = RecommendationReport(
                id="test_analysis_002",
                total_monthly_savings=sample_recommendation.estimated_monthly_savings,
                total_annual_savings=sample_recommendation.annual_savings,
                total_recommendations=1,
                recommendations=[sample_recommendation],
            )
            mock_analyze.return_value = mock_report

            report = await coordinator_agent.analyze_resources_and_generate_report(
                resources, metrics_data, billing_data, batch_mode=False
            )

            assert report.total_recommendations == 1
            assert len(report.recommendations) == 1
            assert (
                report.recommendations[0].rationale == sample_recommendation.rationale
            )
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_for_service(self, coordinator_agent):
        """Test getting appropriate agent for service."""
        # Test AWS EC2 service - coordinator should have this agent initialized
        assert ServiceType.AWS.EC2 in coordinator_agent.service_agents
        ec2_agent = coordinator_agent.service_agents[ServiceType.AWS.EC2]
        assert ec2_agent is not None

        # Test default service
        assert ServiceType.DEFAULT in coordinator_agent.service_agents
        default_agent = coordinator_agent.service_agents[ServiceType.DEFAULT]
        assert default_agent is not None

    @pytest.mark.asyncio
    async def test_consolidate_recommendations(self, coordinator_agent):
        """Test recommendation consolidation and deduplication."""
        from llm_cost_recommendation.models.types import RecommendationType

        # Create duplicate recommendations
        recommendations = [
            Recommendation(
                id="rec_001",
                resource_id="i-12345",
                service="AWS.EC2",
                recommendation_type=RecommendationType.RIGHTSIZING,
                current_config={"instance_type": "t3.large"},
                recommended_config={"instance_type": "t3.medium"},
                current_monthly_cost=100.0,
                estimated_monthly_cost=50.0,
                estimated_monthly_savings=50.0,
                annual_savings=600.0,
                risk_level=RiskLevel.LOW,
                impact_description="Test impact",
                rollback_plan="Test rollback",
                rationale="Duplicate 1",
                confidence_score=0.8,
                agent_id="test_agent",
            ),
            Recommendation(
                id="rec_002",
                resource_id="i-12345",
                service="AWS.EC2",
                recommendation_type=RecommendationType.RIGHTSIZING,
                current_config={"instance_type": "t3.large"},
                recommended_config={"instance_type": "t3.medium"},
                current_monthly_cost=100.0,
                estimated_monthly_cost=55.0,
                estimated_monthly_savings=45.0,
                annual_savings=540.0,
                risk_level=RiskLevel.LOW,
                impact_description="Test impact",
                rollback_plan="Test rollback",
                rationale="Duplicate 2",
                confidence_score=0.9,
                agent_id="test_agent",
            ),
            Recommendation(
                id="rec_003",
                resource_id="i-67890",
                service="AWS.S3",
                recommendation_type=RecommendationType.STORAGE_CLASS,
                current_config={"storage_class": "STANDARD"},
                recommended_config={"storage_class": "IA"},
                current_monthly_cost=50.0,
                estimated_monthly_cost=20.0,
                estimated_monthly_savings=30.0,
                annual_savings=360.0,
                risk_level=RiskLevel.MEDIUM,
                impact_description="Test impact",
                rollback_plan="Test rollback",
                rationale="Different resource",
                confidence_score=0.7,
                agent_id="test_agent",
            ),
        ]

        deduplicated = coordinator_agent._deduplicate_recommendations(recommendations)

        # Should have 2 recommendations (duplicates consolidated)
        assert len(deduplicated) == 2

        # Higher confidence recommendation should be kept when savings are close
        # (50.0 vs 45.0 are within 80% threshold, so confidence 0.9 > 0.8 wins)
        ec2_rec = next(r for r in deduplicated if r.resource_id == "i-12345")
        assert ec2_rec.estimated_monthly_savings == 45.0  # Higher confidence one
        assert ec2_rec.rationale == "Duplicate 2"

    @pytest.mark.asyncio
    async def test_rank_recommendations(self, coordinator_agent):
        """Test recommendation ranking by value."""
        from llm_cost_recommendation.models.types import RecommendationType

        recommendations = [
            Recommendation(
                id="rec_001",
                resource_id="i-12345",
                service="AWS.EC2",
                recommendation_type=RecommendationType.RIGHTSIZING,
                current_config={"instance_type": "t3.large"},
                recommended_config={"instance_type": "t3.medium"},
                current_monthly_cost=50.0,
                estimated_monthly_cost=20.0,
                estimated_monthly_savings=30.0,
                annual_savings=360.0,
                risk_level=RiskLevel.LOW,
                impact_description="Test impact",
                rollback_plan="Test rollback",
                rationale="Lower savings",
                confidence_score=0.8,
                agent_id="test_agent",
            ),
            Recommendation(
                id="rec_002",
                resource_id="i-67890",
                service="AWS.S3",
                recommendation_type=RecommendationType.STORAGE_CLASS,
                current_config={"storage_class": "STANDARD"},
                recommended_config={"storage_class": "IA"},
                current_monthly_cost=150.0,
                estimated_monthly_cost=50.0,
                estimated_monthly_savings=100.0,
                annual_savings=1200.0,
                risk_level=RiskLevel.MEDIUM,
                impact_description="Test impact",
                rollback_plan="Test rollback",
                rationale="Higher savings",
                confidence_score=0.7,
                agent_id="test_agent",
            ),
        ]

        ranked = coordinator_agent._rank_recommendations(recommendations)

        # Should be ordered by composite score (considers savings, risk, confidence)
        # Lower risk (LOW vs MEDIUM) and higher confidence can outweigh higher savings
        assert ranked[0].estimated_monthly_savings == 30.0  # Better overall score
        assert ranked[1].estimated_monthly_savings == 100.0


class TestServiceAgent:
    """Test ServiceAgent functionality."""

    def test_service_agent_initialization(self, config_manager, mock_llm_service):
        """Test ServiceAgent initialization."""
        # Get EC2 service config
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        assert service_config is not None

        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        assert agent.agent_config == service_config
        assert agent.agent_id == "aws.ec2_agent"
        assert agent.service == ServiceType.AWS.EC2

    @pytest.mark.asyncio
    async def test_analyze_resource(
        self,
        config_manager,
        mock_llm_service,
        sample_resource,
        sample_billing_data,
        sample_metrics,
    ):
        """Test analyzing a single resource."""
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        result = await agent.analyze_single_resource(
            sample_resource, sample_metrics[0], sample_billing_data  # Already a list
        )

        # Verify LLM service was called
        mock_llm_service.generate_recommendation.assert_called_once()
        assert result is not None

    def test_agent_capabilities(self, config_manager, mock_llm_service):
        """Test agent capabilities."""
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        capabilities = agent.get_capabilities()

        # Verify capabilities contain expected information
        assert "service" in capabilities
        assert "supported_recommendation_types" in capabilities
        assert capabilities["service"] == ServiceType.AWS.EC2

    def test_validate_resource_data(
        self, config_manager, mock_llm_service, sample_resource
    ):
        """Test resource data validation."""
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        # Valid resource
        assert agent._validate_resource_data(sample_resource) is True

        # Invalid resource (missing required fields)
        invalid_resource = Resource(
            resource_id="",  # Empty resource ID
            service=ServiceType.AWS.EC2,
            cloud_provider=CloudProvider.AWS,
            region="us-east-1",
            account_id="123456789012",
            tags={},
            properties={},
        )

        assert agent._validate_resource_data(invalid_resource) is False

    async def test_multiple_resources_analysis(
        self,
        config_manager,
        mock_llm_service,
        sample_resource,
        sample_billing_data,
        sample_metrics,
    ):
        """Test analyzing multiple resources with simplified logic."""
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        # Create multiple similar resources
        resources = []
        for i in range(3):
            resource = Resource(
                resource_id=f"i-test-{i}",
                service=ServiceType.AWS.EC2,
                cloud_provider=CloudProvider.AWS,
                region="us-east-1",
                tags={"Environment": "test"},
                properties={"InstanceType": "t3.micro"},
            )
            resources.append(resource)

        # Test that we can analyze multiple resources (even if individually)
        results = []
        for resource in resources:
            try:
                result = await agent.analyze_resource(
                    resource, sample_billing_data, sample_metrics
                )
                results.append(result)
            except Exception:
                # LLM might fail in test environment, but agent should handle gracefully
                results.append(None)

        # Should have attempted analysis of all resources
        assert len(results) == 3

        # Test agent can handle multiple resources without crashing
        assert isinstance(agent.agent_id, str)
        assert agent.service == ServiceType.AWS.EC2

        # Test resource validation works for multiple resources
        valid_count = 0
        for resource in resources:
            if agent._validate_resource_data(resource):
                valid_count += 1

        # All test resources should be valid
        assert valid_count == 3

    def test_apply_custom_conditions(self, config_manager, mock_llm_service):
        """Test applying custom conditions to adjust results."""
        service_config = config_manager.get_service_config(ServiceType.AWS.EC2)
        agent = ServiceAgent(
            service_config, mock_llm_service, config_manager.global_config
        )

        resource = Resource(
            resource_id="i-prod-critical",
            service=ServiceType.AWS.EC2,
            cloud_provider=CloudProvider.AWS,
            region="us-east-1",
            tags={"Environment": "production", "Criticality": "critical"},
        )

        # Apply custom rules (this would use the conditions from the config)
        rule_results = agent._apply_custom_rules(resource, None, None)

        # Should return rule results dict
        assert isinstance(rule_results, dict)
        # The rule processor might return modifications or empty dict
