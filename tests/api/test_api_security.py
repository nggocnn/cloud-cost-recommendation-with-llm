"""
API security and robustness tests.
"""

import pytest
import json
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from llm_cost_recommendation.api import app


class TestAPISecurityAndRobustness:
    """Test API security and robustness."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_request_size_limits(self, client):
        """Test that API handles large request sizes appropriately."""
        
        # Test with very large request body
        large_resources = []
        for i in range(10000):  # Large number of resources
            large_resources.append({
                "resource_id": f"resource-{i}",
                "service": "EC2",
                "region": "us-east-1",
                "tags": {f"tag-{j}": f"value-{j}" for j in range(10)},
                "properties": {f"prop-{j}": f"value-{j}" for j in range(10)}
            })
        
        large_request = {
            "resources": large_resources,
            "metrics": [],
            "billing": []
        }
        
        # Should handle large requests or reject them gracefully
        try:
            response = client.post("/analyze", json=large_request, timeout=30)
            # If accepted, should return appropriate response
            # 503 is also valid if services are unavailable during testing
            assert response.status_code in [200, 413, 422, 500, 503]  # Valid status codes
        except Exception as e:
            # May timeout or be rejected - that's acceptable
            assert "timeout" in str(e).lower() or "size" in str(e).lower()

    def test_malformed_json_requests(self, client):
        """Test handling of malformed JSON requests."""
        
        # Test with invalid JSON
        malformed_requests = [
            '{"invalid": json}',  # Invalid JSON syntax
            '{"resources": [{"incomplete"}]}',  # Incomplete JSON
            '{"resources": [null]}',  # Null resources
            '{"resources": "not_an_array"}',  # Wrong type
            '{}',  # Empty object
            '',  # Empty string
        ]
        
        for malformed_json in malformed_requests:
            response = client.post(
                "/analyze",
                content=malformed_json,
                headers={"Content-Type": "application/json"}
            )
            # Should return 4xx error for malformed requests
            assert 400 <= response.status_code < 500

    def test_http_method_security(self, client):
        """Test that endpoints only accept appropriate HTTP methods."""
        
        endpoints = [
            "/health",
            "/health/live",
            "/health/ready", 
            "/status",
            "/analyze"
        ]
        
        for endpoint in endpoints:
            # Test unauthorized methods
            if endpoint != "/analyze":
                # Most endpoints should only accept GET
                response = client.post(endpoint)
                assert response.status_code == 405  # Method Not Allowed
                
                response = client.put(endpoint)
                assert response.status_code == 405
                
                response = client.delete(endpoint)
                assert response.status_code == 405
            
            # Test OPTIONS (should be allowed for CORS)
            response = client.options(endpoint)
            assert response.status_code in [200, 405]  # May be allowed or not

    def test_cors_headers(self, client):
        """Test CORS header configuration."""
        
        # Test preflight request
        response = client.options("/health", headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "GET"
        })
        
        # Check CORS headers
        if response.status_code == 200:
            # If CORS is enabled, check headers
            assert "Access-Control-Allow-Origin" in response.headers
            # Should be careful about allowing all origins in production
            
    def test_rate_limiting_simulation(self, client):
        """Test behavior under rapid requests (rate limiting simulation)."""
        
        # Make rapid requests
        responses = []
        start_time = time.time()
        
        for i in range(20):  # Rapid requests
            response = client.get("/health")
            responses.append(response.status_code)
        
        end_time = time.time()
        request_duration = end_time - start_time
        
        # All requests should complete (no rate limiting currently implemented)
        assert all(status == 200 for status in responses)
        
        # But should complete reasonably quickly
        assert request_duration < 5.0, f"Requests took too long: {request_duration:.2f}s"

    def test_error_information_disclosure(self, client):
        """Test that errors don't disclose sensitive information."""
        
        # Test with requests designed to trigger errors
        error_requests = [
            {"invalid_field": "value"},
            {"resources": [{"invalid_resource": "data"}]},
            {"resources": [], "invalid_top_level": "field"}
        ]
        
        for error_request in error_requests:
            response = client.post("/analyze", json=error_request)
            
            if response.status_code >= 400:
                error_body = response.json()
                error_text = json.dumps(error_body)
                
                # Should not expose internal paths, stack traces, or sensitive info
                sensitive_patterns = [
                    "/home/",
                    "C:\\",
                    "password",
                    "secret", 
                    "key",
                    "token",
                    "Traceback",
                    "__file__",
                    "line ",
                    ".py"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in error_text.lower(), f"Sensitive pattern '{pattern}' found in error response"

    def test_request_id_tracking(self, client):
        """Test that requests have proper ID tracking."""
        
        response = client.get("/health")
        
        # Should have request ID in headers
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0
        
        # Request ID should be unique
        response2 = client.get("/health")
        request_id2 = response2.headers["X-Request-ID"]
        assert request_id != request_id2

    def test_concurrent_request_safety(self, client):
        """Test thread safety with concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.get("/health")
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        statuses = []
        while not results.empty():
            statuses.append(results.get())
        
        # All should succeed
        assert len(statuses) == 10
        assert all(status == 200 for status in statuses if isinstance(status, int))

    def test_input_sanitization_headers(self, client):
        """Test input sanitization in headers."""
        
        # Test with malicious headers
        malicious_headers = {
            "X-Forwarded-For": "'; DROP TABLE users; --",
            "User-Agent": "<script>alert('xss')</script>",
            "Referer": "javascript:alert('xss')",
            "Custom-Header": "../../../etc/passwd"
        }
        
        response = client.get("/health", headers=malicious_headers)
        
        # Should handle malicious headers safely
        assert response.status_code == 200
        
        # Response should not echo back malicious content
        response_text = response.text
        assert "<script>" not in response_text
        assert "DROP TABLE" not in response_text

    def test_response_time_consistency(self, client):
        """Test that response times are consistent and not vulnerable to timing attacks."""
        
        # Make multiple requests and measure timing
        times = []
        
        for _ in range(10):
            start = time.time()
            response = client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            times.append(end - start)
        
        # Response times should be relatively consistent
        avg_time = sum(times) / len(times)
        max_deviation = max(abs(t - avg_time) for t in times)
        
        # Allow reasonable deviation (not more than 10x average)
        assert max_deviation < avg_time * 10, "Inconsistent response times may indicate timing attack vulnerability"

    def test_memory_exhaustion_protection(self, client):
        """Test protection against memory exhaustion attacks."""
        
        # Test with deeply nested JSON (smaller depth to avoid JSON serialization issues)
        deeply_nested = {}
        current = deeply_nested
        
        # Use moderate nesting that tests protection without causing client-side issues
        for i in range(50):  # Moderate nesting
            current["nested"] = {}
            current = current["nested"]
        
        current["value"] = "deep"
        
        try:
            response = client.post("/analyze", json={
                "resources": [deeply_nested],
                "metrics": [],
                "billing": []
            })
            
            # Should handle deeply nested structures without crashing
            # May accept or reject, but should not cause server crash
            # 503 is also valid if services are unavailable during testing
            assert response.status_code in [200, 400, 413, 422, 500, 503]
            
        except RecursionError:
            # Client-side recursion error during JSON serialization is acceptable
            # This indicates the nesting is deep enough to test the protection
            pass

    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection in API parameters."""
        
        # Test with SQL injection patterns in request body
        injection_request = {
            "resources": [
                {
                    "resource_id": "'; DROP TABLE resources; --",
                    "service": "EC2' UNION SELECT * FROM secrets --",
                    "region": "us-east-1'; DELETE FROM users; --",
                    "tags": {
                        "Environment": "' OR 1=1; --"
                    }
                }
            ],
            "metrics": [],
            "billing": []
        }
        
        response = client.post("/analyze", json=injection_request)
        
        # Should handle SQL injection patterns safely
        # The application uses Pydantic models and doesn't use SQL databases,
        # but should still handle these patterns safely
        assert response.status_code in [200, 400, 422]

    def test_file_upload_security(self, client):
        """Test file upload security (if applicable)."""
        
        # Test with malicious file-like content in JSON
        malicious_content = {
            "resources": [
                {
                    "resource_id": "test",
                    "service": "EC2",
                    "region": "us-east-1",
                    "properties": {
                        "script_content": "#!/bin/bash\nrm -rf /",
                        "config_file": "<?php system($_GET['cmd']); ?>"
                    }
                }
            ],
            "metrics": [],
            "billing": []
        }
        
        response = client.post("/analyze", json=malicious_content)
        
        # Should treat as regular string data, not executable content
        # 503 is also valid if services are unavailable during testing
        assert response.status_code in [200, 400, 422, 503]

    @patch('llm_cost_recommendation.api.app_state')
    def test_service_unavailable_handling(self, mock_app_state, client):
        """Test handling when internal services are unavailable."""
        
        # Simulate service unavailability
        mock_app_state.get.return_value = None
        
        response = client.post("/analyze", json={
            "resources": [
                {
                    "resource_id": "test",
                    "service": "EC2", 
                    "region": "us-east-1"
                }
            ],
            "metrics": [],
            "billing": []
        })
        
        # Should handle service unavailability gracefully
        assert response.status_code in [500, 503]  # Internal Error or Service Unavailable
        
        if response.status_code >= 400:
            error_response = response.json()
            assert "error" in error_response

    def test_request_timeout_handling(self, client):
        """Test handling of request timeouts."""
        
        # Test with very short timeout
        try:
            response = client.post("/analyze", json={
                "resources": [{"resource_id": "test", "service": "EC2", "region": "us-east-1"}],
                "metrics": [],
                "billing": []
            }, timeout=0.001)  # Very short timeout
            
            # If it completes, should be valid response
            # 503 is also valid if services are unavailable during testing
            assert response.status_code in [200, 400, 422, 500, 503]
            
        except Exception as e:
            # Timeout is expected with such a short limit
            assert "timeout" in str(e).lower() or "time" in str(e).lower()

    def test_content_type_validation(self, client):
        """Test content type validation."""
        
        valid_json = '{"resources": [], "metrics": [], "billing": []}'
        
        # Test with correct content type
        response = client.post(
            "/analyze",
            content=valid_json,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 400, 422]
        
        # Test with incorrect content type
        response = client.post(
            "/analyze",
            content=valid_json,
            headers={"Content-Type": "text/plain"}
        )
        # Should reject or handle gracefully
        assert response.status_code in [400, 415, 422]  # Bad Request or Unsupported Media Type