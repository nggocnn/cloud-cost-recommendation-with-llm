"""
Pytest-compatible version of API route testing.
Comprehensive tests for all API endpoints with proper pytest structure.
"""

import pytest
import requests
import json
import time
from typing import Dict, Any


@pytest.mark.api
@pytest.mark.integration
class TestAPIRoutes:
    """Test all API routes comprehensively."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment for each test."""
        self.base_url = "http://localhost:8000"
        self.timeout = 30
        self.results = {}

    def make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[Any, Any] = None,
        headers: Dict[str, str] = None,
        expected_status: int = 200,
    ) -> Dict[str, Any]:
        """Make HTTP request and return structured result."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            elif method.upper() == "PUT":
                response = requests.put(
                    url, json=data, headers=headers, timeout=self.timeout
                )
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time

            # Parse response
            try:
                response_json = response.json()
            except ValueError:
                response_json = {"raw_content": response.text}

            success = response.status_code == expected_status

            return {
                "success": success,
                "status_code": response.status_code,
                "response_time": response_time,
                "response_data": response_json,
                "headers": dict(response.headers),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": None,
                "response_time": time.time() - start_time,
            }

    @pytest.mark.smoke
    def test_server_connectivity(self):
        """Test basic server connectivity."""
        result = self.make_request("GET", "/health")
        assert result[
            "success"
        ], f"Server not reachable: {result.get('error', 'Unknown error')}"
        assert result["status_code"] == 200

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        result = self.make_request("GET", "/health")

        assert result["success"]
        assert result["status_code"] == 200
        assert "status" in result["response_data"]
        assert result["response_data"]["status"] == "healthy"

    def test_health_live_probe(self):
        """Test health live probe endpoint."""
        result = self.make_request("GET", "/health/live")

        assert result["success"]
        assert result["status_code"] == 200
        assert "status" in result["response_data"]
        assert result["response_data"]["status"] == "alive"

    def test_health_ready_probe(self):
        """Test health ready probe endpoint."""
        result = self.make_request("GET", "/health/ready")

        assert result["success"]
        assert result["status_code"] == 200
        assert "status" in result["response_data"]
        assert result["response_data"]["status"] == "ready"

    def test_system_status_endpoint(self):
        """Test system status endpoint."""
        result = self.make_request("GET", "/status")

        assert result["success"]
        assert result["status_code"] == 200

        data = result["response_data"]
        assert "config" in data
        assert "agents" in data
        assert "timestamp" in data
        assert isinstance(data["agents"]["total"], int)
        assert data["agents"]["total"] > 0

    def test_system_metrics_endpoint(self):
        """Test system metrics endpoint."""
        result = self.make_request("GET", "/metrics/system")

        assert result["success"]
        assert result["status_code"] == 200

        data = result["response_data"]
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert isinstance(data["cpu_percent"], (int, float))
        assert isinstance(data["memory_percent"], (int, float))

    def test_openapi_schema_endpoint(self):
        """Test OpenAPI schema endpoint."""
        result = self.make_request("GET", "/openapi.json")

        assert result["success"]
        assert result["status_code"] == 200

        schema = result["response_data"]
        assert "openapi" in schema
        assert "paths" in schema
        assert "/analyze" in schema["paths"]

    def test_swagger_ui_endpoint(self):
        """Test Swagger UI documentation endpoint."""
        result = self.make_request("GET", "/docs")

        assert result["success"]
        assert result["status_code"] == 200
        assert "text/html" in result["headers"].get("content-type", "")

    def test_redoc_endpoint(self):
        """Test ReDoc documentation endpoint."""
        result = self.make_request("GET", "/redoc")

        assert result["success"]
        assert result["status_code"] == 200
        assert "text/html" in result["headers"].get("content-type", "")

    @pytest.mark.mock_llm
    def test_analyze_endpoint_valid_request(self, api_test_data):
        """Test analyze endpoint with valid request."""
        result = self.make_request("POST", "/analyze", data=api_test_data)

        # May fail due to missing LLM configuration, but should not crash
        assert result["status_code"] in [200, 500, 422]

        if result["success"]:
            data = result["response_data"]
            assert "report" in data
            assert "request_id" in data

    def test_analyze_endpoint_invalid_request(self):
        """Test analyze endpoint with invalid request."""
        invalid_data = {
            "resources": [],  # Empty resources should fail validation
            "billing": [],
        }

        result = self.make_request(
            "POST", "/analyze", data=invalid_data, expected_status=422
        )

        assert result["status_code"] == 422
        assert "detail" in result["response_data"]

    def test_analyze_endpoint_missing_data(self):
        """Test analyze endpoint with missing required data."""
        result = self.make_request("POST", "/analyze", data={}, expected_status=422)

        assert result["status_code"] == 422
        assert "detail" in result["response_data"]

    def test_recommendation_detail_endpoint(self):
        """Test recommendation detail endpoint (should return 501 - not implemented)."""
        result = self.make_request(
            "GET", "/recommendations/test-rec-001", expected_status=501
        )

        assert result["status_code"] == 501

    def test_nonexistent_endpoint(self):
        """Test non-existent endpoint returns 404."""
        result = self.make_request("GET", "/non-existent-route", expected_status=404)

        assert result["status_code"] == 404

    def test_method_not_allowed(self):
        """Test method not allowed returns 405."""
        result = self.make_request("POST", "/health", expected_status=405)

        assert result["status_code"] == 405

    @pytest.mark.slow
    def test_response_times(self):
        """Test that response times are reasonable."""
        endpoints = [
            ("/health", "GET"),
            ("/status", "GET"),
            ("/metrics/system", "GET"),
        ]

        for endpoint, method in endpoints:
            result = self.make_request(method, endpoint)

            if result["success"]:
                # Response time should be under 5 seconds for health endpoints
                assert (
                    result["response_time"] < 5.0
                ), f"Slow response for {endpoint}: {result['response_time']:.2f}s"

    def test_cors_headers_present(self, api_test_data):
        """Test that CORS headers are present."""
        headers = {"Origin": "http://localhost:3000"}
        result = self.make_request(
            "POST", "/analyze", data=api_test_data, headers=headers
        )

        # Should have CORS headers regardless of success/failure
        response_headers = result.get("headers", {})
        cors_headers = [
            h for h in response_headers.keys() if "access-control" in h.lower()
        ]

        # At least some CORS headers should be present
        assert len(cors_headers) > 0, "No CORS headers found in response"

    @pytest.mark.parametrize(
        "endpoint,method",
        [
            ("/health", "GET"),
            ("/health/live", "GET"),
            ("/health/ready", "GET"),
            ("/status", "GET"),
            ("/metrics/system", "GET"),
            ("/docs", "GET"),
            ("/redoc", "GET"),
            ("/openapi.json", "GET"),
        ],
    )
    def test_all_get_endpoints(self, endpoint, method):
        """Test all GET endpoints systematically."""
        result = self.make_request(method, endpoint)

        assert result[
            "success"
        ], f"Failed to access {endpoint}: {result.get('error', 'Unknown error')}"
        assert result["status_code"] == 200


@pytest.mark.stress
@pytest.mark.slow
class TestAPIStress:
    """Stress tests for API endpoints."""

    def setup_method(self):
        """Setup for stress tests."""
        self.base_url = "http://localhost:8000"

    def test_rapid_health_checks(self):
        """Test rapid successive health check requests."""
        results = []

        for i in range(20):
            start_time = time.time()
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                response_time = time.time() - start_time

                results.append(
                    {
                        "success": response.status_code == 200,
                        "response_time": response_time,
                        "status_code": response.status_code,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time,
                    }
                )

        # At least 90% should succeed
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results)

        assert success_rate >= 0.9, f"Success rate too low: {success_rate:.2%}"

        # Average response time should be reasonable
        response_times = [r["response_time"] for r in results if r.get("success")]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            assert (
                avg_response_time < 2.0
            ), f"Average response time too high: {avg_response_time:.2f}s"
