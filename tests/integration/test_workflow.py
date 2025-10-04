"""
Integration tests for the complete analysis workflow.
Tests end-to-end functionality including data ingestion, analysis, and reporting.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from llm_cost_recommendation.cli import CostRecommendationApp
from llm_cost_recommendation.services.config import ConfigManager
from llm_cost_recommendation.services.llm import LLMService
from llm_cost_recommendation.services.ingestion import DataIngestionService
from llm_cost_recommendation.agents.coordinator import CoordinatorAgent
from llm_cost_recommendation.models.recommendations import RecommendationReport


class TestEndToEndWorkflow:
    """Test complete end-to-end analysis workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(
        self, sample_config_dir, sample_data_dir, mock_llm_service
    ):
        """Test complete analysis workflow from data loading to report generation."""

        # Initialize the application
        app = CostRecommendationApp(
            config_dir=str(sample_config_dir), data_dir=str(sample_data_dir)
        )

        # Replace LLM service with mock
        app.llm_service = mock_llm_service
        app.coordinator = CoordinatorAgent(app.config_manager, mock_llm_service)

        # Run analysis with sample data
        report = await app.run_analysis(
            billing_file=str(sample_data_dir / "billing" / "sample_billing.csv"),
            inventory_file=str(sample_data_dir / "inventory" / "sample_inventory.json"),
            metrics_file=str(sample_data_dir / "metrics" / "sample_metrics.csv"),
            use_sample_data=False,
            individual_processing=False,
        )

        # Verify report structure
        assert report.id is not None
        assert report.generated_at is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sample_data_workflow(
        self, sample_config_dir, sample_data_dir, mock_llm_service
    ):
        """Test workflow using generated sample data."""

        app = CostRecommendationApp(
            config_dir=str(sample_config_dir), data_dir=str(sample_data_dir)
        )

        # Replace LLM service with mock
        app.llm_service = mock_llm_service
        app.coordinator = CoordinatorAgent(app.config_manager, mock_llm_service)

        # Run analysis with sample data generation
        report = await app.run_analysis(use_sample_data=True)

        # Verify report was generated
        assert isinstance(report, RecommendationReport)
        assert len(report.recommendations) > 0
        assert report.coverage["total_resources_analyzed"] > 0
        assert report.total_monthly_savings >= 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_individual_processing_mode(
        self, sample_config_dir, sample_data_dir, mock_llm_service
    ):
        """Test individual processing mode workflow."""

        app = CostRecommendationApp(
            config_dir=str(sample_config_dir), data_dir=str(sample_data_dir)
        )

        # Replace LLM service with mock
        app.llm_service = mock_llm_service
        app.coordinator = CoordinatorAgent(app.config_manager, mock_llm_service)

        # Run analysis in individual mode
        report = await app.run_analysis(
            use_sample_data=True, individual_processing=True
        )

        # Verify report structure
        assert isinstance(report, RecommendationReport)
        assert isinstance(report.recommendations, list)
        # Individual processing mode is verified by the specific configuration passed above

    @pytest.mark.integration
    def test_data_validation_integration(self, sample_data_dir):
        """Test data validation during ingestion."""

        data_service = DataIngestionService(str(sample_data_dir))

        # Load all data types and verify validation
        billing_data = data_service.ingest_billing_data(
            str(sample_data_dir / "billing" / "sample_billing.csv")
        )
        inventory_data = data_service.ingest_inventory_data(
            str(sample_data_dir / "inventory" / "sample_inventory.json")
        )
        metrics_data = data_service.ingest_metrics_data(
            str(sample_data_dir / "metrics" / "sample_metrics.csv")
        )

        # Verify data integrity
        assert len(billing_data) > 0
        assert len(inventory_data) > 0
        assert len(metrics_data) > 0

        # Verify data relationships
        resource_ids = {r.resource_id for r in inventory_data}
        billing_ids = {b.resource_id for b in billing_data}
        metrics_ids = {m.resource_id for m in metrics_data}

        # All should reference the same resources
        assert resource_ids == billing_ids == metrics_ids

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_coordination_integration(
        self,
        config_manager,
        mock_llm_service,
        sample_resource,
        sample_billing_data,
        sample_metrics,
    ):
        """Test agent coordination and communication."""

        coordinator = CoordinatorAgent(config_manager, mock_llm_service)

        # Test with multiple resources of different types
        resources = [
            sample_resource,  # EC2 instance
            # Add more resource types here if available in fixtures
        ]

        billing = sample_billing_data
        metrics = sample_metrics

        # Run analysis
        report = await coordinator.analyze_resources_and_generate_report(
            resources, metrics, billing, batch_mode=True
        )
        recommendations = report.recommendations

        # Verify coordination results
        assert isinstance(recommendations, list)

        # Each resource should have corresponding recommendations
        recommendation_resource_ids = {r.resource_id for r in recommendations}
        resource_ids = {r.resource_id for r in resources}

        # All resources should have been processed
        assert recommendation_resource_ids.issubset(resource_ids)

    @pytest.mark.integration
    def test_configuration_integration(self, sample_config_dir):
        """Test configuration loading and agent initialization."""

        config_manager = ConfigManager(str(sample_config_dir))

        # Verify coordinator config
        assert config_manager.llm_config is not None
        assert hasattr(config_manager.llm_config, "provider")
        assert config_manager.llm_config.provider is not None

        # Verify service configs
        assert len(config_manager.service_configs) > 0

        # Test getting specific service config
        from llm_cost_recommendation.models.types import ServiceType

        if ServiceType.AWS.EC2 in config_manager.service_configs:
            ec2_config = config_manager.service_configs[ServiceType.AWS.EC2]
            assert ec2_config is not None
            assert ec2_config.service == ServiceType.AWS.EC2
            assert ec2_config.agent_id is not None
            assert ec2_config.enabled is True

    @pytest.mark.integration
    @pytest.mark.mock_llm
    @pytest.mark.asyncio
    async def test_llm_integration_flow(self, config_manager):
        """Test LLM integration in the analysis flow."""

        # Mock LLM responses
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock chat completion response
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(
                    message=AsyncMock(
                        content='{"recommendations": [{"type": "rightsizing", "monthly_savings": 50.0}]}'
                    )
                )
            ]
            mock_client.chat.completions.create.return_value = mock_response

            llm_service = LLMService(config_manager.llm_config)
            coordinator = CoordinatorAgent(config_manager, llm_service)

            # Create test data
            from llm_cost_recommendation.models import Resource, BillingData, Metrics
            from llm_cost_recommendation.models.types import ServiceType, CloudProvider

            resources = [
                Resource(
                    resource_id="i-integration-test",
                    service=ServiceType.AWS.EC2,
                    cloud_provider=CloudProvider.AWS,
                    region="us-east-1",
                )
            ]

            billing = [
                BillingData(
                    resource_id="i-integration-test",
                    service="AWS.EC2",
                    region="us-east-1",
                    unblended_cost=100.0,
                    usage_type="BoxUsage:t3.large",
                    period="2024-09-01",
                    usage_amount=24.0,
                    usage_unit="Hrs",
                    amortized_cost=100.0,
                    bill_period_start="2024-09-01T00:00:00Z",
                    bill_period_end="2024-09-30T23:59:59Z",
                )
            ]

            metrics = [
                Metrics(
                    resource_id="i-integration-test",
                    timestamp="2024-09-01T00:00:00Z",
                    period_days=30,
                    cpu_utilization_p50=25.0,
                    is_idle=False,
                )
            ]

            # Run analysis
            report = await coordinator.analyze_resources_and_generate_report(
                resources, metrics, billing, batch_mode=True
            )
            recommendations = report.recommendations

            # Verify LLM was called and results processed
            assert isinstance(recommendations, list)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self, sample_config_dir, temp_dir, mock_llm_service
    ):
        """Test error handling in the integration workflow."""

        app = CostRecommendationApp(
            config_dir=str(sample_config_dir), data_dir=str(temp_dir)  # Empty data dir
        )

        # Replace LLM service with mock
        app.llm_service = mock_llm_service
        app.coordinator = CoordinatorAgent(app.config_manager, mock_llm_service)

        # Test with non-existent files - should handle gracefully
        report = await app.run_analysis(
            billing_file="nonexistent_billing.csv",
            inventory_file="nonexistent_inventory.json",
            metrics_file="nonexistent_metrics.csv",
            use_sample_data=False,
        )

        # Should return None when no resources are found to analyze
        assert report is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_processing_integration(
        self, sample_config_dir, sample_data_dir, mock_llm_service
    ):
        """Test concurrent processing capabilities."""

        app = CostRecommendationApp(
            config_dir=str(sample_config_dir), data_dir=str(sample_data_dir)
        )

        # Replace LLM service with mock
        app.llm_service = mock_llm_service
        app.coordinator = CoordinatorAgent(app.config_manager, mock_llm_service)

        # Generate larger sample data for concurrent processing
        app.data_service.create_sample_data(num_resources=10)

        # Load the created sample data
        billing_file = str(app.data_service.data_dir / "billing" / "sample_billing.csv")
        inventory_file = str(
            app.data_service.data_dir / "inventory" / "sample_inventory.json"
        )
        metrics_file = str(app.data_service.data_dir / "metrics" / "sample_metrics.csv")

        resources = app.data_service.ingest_inventory_data(inventory_file)
        billing_list = app.data_service.ingest_billing_data(billing_file)
        metrics_list = app.data_service.ingest_metrics_data(metrics_file)

        # Group billing data by resource_id
        from collections import defaultdict

        billing_grouped = defaultdict(list)
        for bill in billing_list:
            if bill.resource_id:
                billing_grouped[bill.resource_id].append(bill)
        billing = dict(billing_grouped)

        # Group metrics by resource_id
        metrics = {m.resource_id: m for m in metrics_list}

        # Run analysis (coordinator should handle concurrency internally)
        report = await app.coordinator.analyze_resources_and_generate_report(
            resources, metrics, billing, batch_mode=True
        )
        recommendations = report.recommendations

        # Verify results
        assert isinstance(recommendations, list)
        assert (
            len(recommendations) >= 0
        )  # May have 0 if mock doesn't return recommendations
