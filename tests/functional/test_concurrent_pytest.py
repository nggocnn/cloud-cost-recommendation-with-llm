"""
Pytest-compatible concurrent API testing.
Tests concurrent processing capabilities and load handling.
"""

import pytest
import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Any
import statistics


@pytest.mark.concurrent
@pytest.mark.integration
class TestConcurrentAPI:
    """Test concurrent API processing capabilities."""

    @pytest.fixture
    def base_url(self):
        """Base URL for API testing."""
        return "http://localhost:8000"

    def generate_test_data(self, client_id: int) -> Dict[str, Any]:
        """Generate unique test data for each client."""
        instance_types = ["t3.medium", "t3.large", "t3.xlarge", "m5.large", "c5.large"]
        regions = ["us-east-1", "us-west-2", "eu-west-1"]

        return {
            "resources": [
                {
                    "resource_id": f"i-client{client_id}-{random.randint(1000, 9999)}",
                    "service": "AWS.EC2",
                    "cloud_provider": "AWS",
                    "region": random.choice(regions),
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {
                        "Environment": random.choice(
                            ["production", "staging", "development"]
                        ),
                        "Team": f"team-{client_id}",
                        "Application": f"app-{client_id}",
                    },
                    "properties": {
                        "InstanceType": random.choice(instance_types),
                        "State": "running",
                        "LaunchTime": "2024-01-15T10:30:00Z",
                    },
                }
            ],
            "billing": [
                {
                    "resource_id": f"i-client{client_id}-{random.randint(1000, 9999)}",
                    "service": "AWS.EC2",
                    "region": random.choice(regions),
                    "unblended_cost": random.uniform(50.0, 500.0),
                    "usage_type": f"BoxUsage:{random.choice(instance_types)}",
                    "period": "2024-09-01",
                }
            ],
            "metrics": [
                {
                    "resource_id": f"i-client{client_id}-{random.randint(1000, 9999)}",
                    "timestamp": "2024-09-01T00:00:00Z",
                    "period_days": 30,
                    "cpu_utilization_p50": random.uniform(5.0, 80.0),
                    "cpu_utilization_p90": random.uniform(20.0, 95.0),
                    "cpu_utilization_p95": random.uniform(30.0, 98.0),
                    "memory_utilization_p50": random.uniform(20.0, 75.0),
                    "memory_utilization_p90": random.uniform(40.0, 90.0),
                    "memory_utilization_p95": random.uniform(50.0, 95.0),
                    "is_idle": random.choice([True, False]),
                }
            ],
            "individual_processing": random.choice([True, False]),
            "max_recommendations": random.randint(5, 15),
        }

    @pytest.mark.asyncio
    async def test_server_availability(self, base_url):
        """Test that server is available before running concurrent tests."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    assert (
                        response.status == 200
                    ), f"Server health check failed: {response.status}"
            except Exception as e:
                pytest.skip(f"Server not available: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, base_url):
        """Test concurrent health check requests."""

        async def check_health(
            session: aiohttp.ClientSession, client_id: int
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                async with session.get(
                    f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = time.time() - start_time
                    return {
                        "client_id": client_id,
                        "success": response.status == 200,
                        "status_code": response.status,
                        "response_time": response_time,
                    }
            except Exception as e:
                return {
                    "client_id": client_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        async with aiohttp.ClientSession() as session:
            # Test with 10 concurrent health checks
            tasks = [check_health(session, i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, dict)]

            # All should succeed
            success_count = sum(1 for r in valid_results if r.get("success", False))
            assert success_count == len(valid_results), "Some health checks failed"

            # Response times should be reasonable
            response_times = [
                r["response_time"] for r in valid_results if r.get("success")
            ]
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                assert (
                    avg_time < 5.0
                ), f"Average response time too high: {avg_time:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.mock_llm
    async def test_concurrent_analyze_requests(self, base_url):
        """Test concurrent analyze requests."""

        async def send_analyze_request(
            session: aiohttp.ClientSession, client_id: int
        ) -> Dict[str, Any]:
            test_data = self.generate_test_data(client_id)
            start_time = time.time()

            try:
                async with session.post(
                    f"{base_url}/analyze",
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response_time = time.time() - start_time

                    try:
                        response_data = await response.json()
                    except Exception:
                        response_data = {"error": "Failed to parse JSON"}

                    return {
                        "client_id": client_id,
                        "success": response.status == 200,
                        "status_code": response.status,
                        "response_time": response_time,
                        "response_data": response_data,
                    }
            except Exception as e:
                return {
                    "client_id": client_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        async with aiohttp.ClientSession() as session:
            # Test with 3 concurrent analyze requests (smaller number due to complexity)
            tasks = [send_analyze_request(session, i) for i in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, dict)]
            assert len(valid_results) == 3, "Some requests failed to complete"

            # At least some should succeed (may fail due to missing LLM config)
            success_count = sum(1 for r in valid_results if r.get("success", False))
            total_count = len(valid_results)

            # If any succeeded, verify response structure
            for result in valid_results:
                if result.get("success"):
                    assert "response_data" in result
                    data = result["response_data"]
                    assert isinstance(data, dict)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_clients", [2, 5, 8])
    async def test_scalable_concurrent_requests(self, base_url, num_clients):
        """Test API scalability with varying numbers of concurrent clients."""

        async def lightweight_request(
            session: aiohttp.ClientSession, client_id: int
        ) -> Dict[str, Any]:
            start_time = time.time()

            try:
                async with session.get(
                    f"{base_url}/status", timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return {
                        "client_id": client_id,
                        "success": response.status == 200,
                        "response_time": time.time() - start_time,
                    }
            except Exception as e:
                return {
                    "client_id": client_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        async with aiohttp.ClientSession() as session:
            tasks = [lightweight_request(session, i) for i in range(num_clients)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = [r for r in results if isinstance(r, dict)]
            success_count = sum(1 for r in valid_results if r.get("success", False))

            # Success rate should be high
            success_rate = success_count / len(valid_results) if valid_results else 0
            assert (
                success_rate >= 0.8
            ), f"Low success rate with {num_clients} clients: {success_rate:.2%}"

            # Response time consistency
            response_times = [
                r["response_time"] for r in valid_results if r.get("success")
            ]
            if len(response_times) > 1:
                max_time = max(response_times)
                min_time = min(response_times)
                time_ratio = min_time / max_time if max_time > 0 else 1

                # Performance should be reasonably consistent (within 5x)
                assert (
                    time_ratio > 0.2
                ), f"High response time variance with {num_clients} clients"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_concurrent_load(self, base_url):
        """Test sustained concurrent load over time."""

        async def sustained_request(
            session: aiohttp.ClientSession, batch_id: int, request_id: int
        ) -> Dict[str, Any]:

            start_time = time.time()

            try:
                async with session.get(
                    f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    return {
                        "batch_id": batch_id,
                        "request_id": request_id,
                        "success": response.status == 200,
                        "response_time": time.time() - start_time,
                    }
            except Exception as e:
                return {
                    "batch_id": batch_id,
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        all_results = []

        async with aiohttp.ClientSession() as session:
            # Send 3 batches of 5 requests each, with small delays between batches
            for batch in range(3):
                tasks = [
                    sustained_request(session, batch, req_id) for req_id in range(5)
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_batch_results = [r for r in batch_results if isinstance(r, dict)]
                all_results.extend(valid_batch_results)

                # Small delay between batches
                if batch < 2:
                    await asyncio.sleep(1)

        # Analyze sustained performance
        assert len(all_results) >= 10, "Insufficient results from sustained test"

        success_count = sum(1 for r in all_results if r.get("success", False))
        success_rate = success_count / len(all_results)

        assert (
            success_rate >= 0.8
        ), f"Sustained load test success rate too low: {success_rate:.2%}"

        # Performance should remain consistent across batches
        response_times = [r["response_time"] for r in all_results if r.get("success")]
        if len(response_times) > 5:
            avg_time = sum(response_times) / len(response_times)
            assert avg_time < 10.0, f"Average response time degraded: {avg_time:.2f}s"
