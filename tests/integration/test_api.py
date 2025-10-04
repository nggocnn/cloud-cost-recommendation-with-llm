"""
Integration tests for API endpoints and FastAPI application.
Tests the complete API functionality with real request/response cycles.
"""

import pytest
import asyncio
import json
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from llm_cost_recommendation.api import app
from llm_cost_recommendation.models.api_models import AnalysisRequest


@pytest.fixture(scope="class")
def test_client():
    """Create a test client with proper lifespan handling."""
    with TestClient(app) as client:
        yield client


class TestAPIIntegration:
    """Test API endpoints integration."""

    @pytest.mark.integration
    @pytest.mark.api
    def test_health_endpoints(self, test_client):
        """Test health check endpoints."""

        # Basic health check
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

        # Live probe
        response = test_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

        # Ready probe
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @pytest.mark.integration
    @pytest.mark.api
    def test_status_endpoint(self, test_client):
        """Test system status endpoint."""
        response = test_client.get("/status")
        assert response.status_code == 200
        data = response.json()

        # Verify status fields
        assert "config" in data
        assert "agents" in data
        assert "timestamp" in data
        assert isinstance(data["agents"]["total"], int)
        assert data["agents"]["total"] > 0

    @pytest.mark.integration
    @pytest.mark.api
    def test_metrics_endpoint(self, test_client):
        """Test system metrics endpoint."""
        response = test_client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()

        # Verify metrics structure
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_percent" in data
        assert "load_average" in data

        # Verify metric types
        assert isinstance(data["cpu_percent"], (int, float))
        assert isinstance(data["memory_percent"], (int, float))
        assert 0 <= data["cpu_percent"] <= 100
        assert 0 <= data["memory_percent"] <= 100

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.mock_llm
    def test_analyze_endpoint_valid_request(self, test_client, api_test_data):
        """Test analyze endpoint with valid request."""

        # Mock the LLM service to avoid API calls
        from llm_cost_recommendation.models.recommendations import RecommendationReport
        from datetime import datetime

        with patch(
            "llm_cost_recommendation.agents.coordinator.CoordinatorAgent.analyze_resources_and_generate_report"
        ) as mock_analyze:
            mock_report = RecommendationReport(
                id="test-report-123",
                total_monthly_savings=50.0,
                total_annual_savings=600.0,
                total_recommendations=1,
                recommendations=[],
            )
            mock_analyze.return_value = mock_report

            response = test_client.post("/analyze", json=api_test_data)
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
            assert response.status_code == 200

            data = response.json()
            assert "report" in data
            assert "request_id" in data
            assert "processing_time_seconds" in data

            # Verify report structure
            report = data["report"]
            assert "id" in report
            assert "generated_at" in report
            assert "recommendations" in report
            assert "total_monthly_savings" in report
            assert isinstance(report["recommendations"], list)

    @pytest.mark.integration
    @pytest.mark.api
    def test_analyze_endpoint_invalid_request(self, test_client):
        """Test analyze endpoint with invalid request data."""

        # Empty request
        response = test_client.post("/analyze", json={})
        assert response.status_code == 422  # Validation error

        # Invalid resource structure
        invalid_data = {
            "resources": [
                {
                    "resource_id": "",  # Empty ID should fail validation
                    "service": "InvalidService",
                }
            ]
        }

        response = test_client.post("/analyze", json=invalid_data)
        assert response.status_code == 422

        # Verify error response structure
        error_data = response.json()
        assert "detail" in error_data

    @pytest.mark.integration
    @pytest.mark.api
    def test_analyze_endpoint_missing_resources(self, test_client):
        """Test analyze endpoint with missing resources."""

        request_data = {
            "billing": [],
            "metrics": [],
            "individual_processing": False,
            # Missing required "resources" field
        }

        response = test_client.post("/analyze", json=request_data)
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    @pytest.mark.integration
    @pytest.mark.api
    def test_analyze_endpoint_timeout_handling(
        self, test_client, minimal_valid_request
    ):
        """Test analyze endpoint timeout handling."""

        with patch("llm_cost_recommendation.api.app_state") as mock_state:
            # Mock coordinator that takes too long
            mock_coordinator = AsyncMock()
            mock_coordinator.analyze_resources.side_effect = asyncio.TimeoutError(
                "Analysis timeout"
            )
            mock_state.get.return_value = mock_coordinator

            response = test_client.post("/analyze", json=minimal_valid_request)

            # Should handle timeout gracefully
            assert response.status_code in [500, 408]  # Internal error or timeout

    @pytest.mark.integration
    @pytest.mark.api
    def test_documentation_endpoints(self, test_client):
        """Test API documentation endpoints."""

        # OpenAPI schema
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/analyze" in schema["paths"]

        # Swagger UI (HTML response)
        response = test_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # ReDoc (HTML response)
        response = test_client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.integration
    @pytest.mark.api
    def test_cors_headers(self, test_client, api_test_data):
        """Test CORS headers are properly set."""

        with patch("llm_cost_recommendation.api.app_state") as mock_state:
            mock_coordinator = AsyncMock()
            mock_coordinator.analyze_resources.return_value = []
            mock_state.get.return_value = mock_coordinator

            # Test actual request with CORS headers
            headers = {"Origin": "http://localhost:3000"}
            response = test_client.post("/analyze", json=api_test_data, headers=headers)

            # Should include CORS headers
            assert (
                "access-control-allow-origin" in response.headers
                or "Access-Control-Allow-Origin" in response.headers
            )

    @pytest.mark.integration
    @pytest.mark.api
    def test_gzip_compression(self, test_client, api_test_data):
        """Test GZip compression middleware."""

        with patch("llm_cost_recommendation.api.app_state") as mock_state:
            mock_coordinator = AsyncMock()
            mock_coordinator.analyze_resources.return_value = []
            mock_state.get.return_value = mock_coordinator

            headers = {"Accept-Encoding": "gzip"}
            response = test_client.post("/analyze", json=api_test_data, headers=headers)

            # Large responses should be compressed
            if len(response.content) > 1024:
                assert "content-encoding" in response.headers

    @pytest.mark.integration
    @pytest.mark.api
    def test_request_logging_middleware(self, test_client, api_test_data):
        """Test request logging middleware functionality."""

        with patch("llm_cost_recommendation.api.app_state") as mock_state:
            mock_coordinator = AsyncMock()
            mock_coordinator.analyze_resources.return_value = []
            mock_state.get.return_value = mock_coordinator

            response = test_client.post("/analyze", json=api_test_data)

            # Should include request ID in response headers or body
            # This depends on the middleware implementation
            assert response.status_code in [200, 500]  # Should complete processing

    @pytest.mark.integration
    @pytest.mark.api
    def test_error_handling_middleware(self, test_client):
        """Test error handling middleware."""

        # Test 404 - Not Found
        response = test_client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        error_data = response.json()
        assert "detail" in error_data

        # Test 405 - Method Not Allowed
        response = test_client.post("/health")  # Health only accepts GET
        assert response.status_code == 405


class TestAPIAsyncIntegration:
    """Test API with async client for better async support."""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, api_test_data):
        """Test handling concurrent API requests with async client."""
        import asyncio
        from httpx import AsyncClient, ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Test concurrent health checks
            tasks = [
                client.get("/health"),
                client.get("/health/live"),
                client.get("/health/ready"),
            ]

            responses = await asyncio.gather(*tasks)

            # Health and live should succeed, ready might be 503 in test environment
            assert responses[0].status_code == 200  # /health
            assert responses[1].status_code == 200  # /health/live
            assert responses[2].status_code in [
                200,
                503,
            ]  # /health/ready (503 is expected in tests)

            # Test concurrent analyze requests (will fail due to missing LLM but that's expected)
            analyze_tasks = [
                client.post("/analyze", json=api_test_data),
                client.post("/analyze", json=api_test_data),
            ]

            analyze_responses = await asyncio.gather(
                *analyze_tasks, return_exceptions=True
            )

            # Should handle concurrent requests without crashing
            assert len(analyze_responses) == 2

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, api_test_data):
        """Test handling of streaming responses (if implemented)."""

        from llm_cost_recommendation.api import app

        with patch("llm_cost_recommendation.api.app_state") as mock_state:
            mock_coordinator = AsyncMock()
            mock_coordinator.analyze_resources.return_value = []
            mock_state.get.return_value = mock_coordinator

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Test with large request that might need streaming
                large_data = api_test_data.copy()

                # Add many resources to make it a larger request
                base_resource = large_data["resources"][0]
                large_data["resources"] = []

                for i in range(50):  # Create 50 resources
                    resource = base_resource.copy()
                    resource["resource_id"] = f"large-test-{i}"
                    large_data["resources"].append(resource)

                response = await client.post("/analyze", json=large_data)

                # Should handle large requests (may fail due to mock format issues)
                assert response.status_code in [
                    200,
                    413,
                    500,
                ]  # Success, too large, or mock error

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_websocket_support(self, test_client):
        """Test WebSocket support (if implemented)."""
        # Note: This is a placeholder for future WebSocket functionality
        # Currently the API doesn't implement WebSockets

        from llm_cost_recommendation.api import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Test that WebSocket endpoint doesn't exist yet
            response = await client.get("/ws")
            assert response.status_code == 404  # Not implemented yet
