#!/usr/bin/env python3
"""
Concurrent Load Test for Analyze API
Tests multiple simultaneous requests to validate concurrent processing capabilities.
"""

import asyncio
import aiohttp
import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import statistics


class ConcurrentAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def generate_test_data(self, client_id: int) -> Dict[str, Any]:
        """Generate unique test data for each client"""

        # Different instance types for variety
        instance_types = ["t3.medium", "t3.large", "t3.xlarge", "m5.large", "c5.large"]
        regions = ["us-east-1", "us-west-2", "eu-west-1"]

        instance_type = random.choice(instance_types)
        region = random.choice(regions)

        return {
            "resources": [
                {
                    "resource_id": f"i-client{client_id}-web-{random.randint(1000, 9999)}",
                    "service": "AWS.EC2",
                    "cloud_provider": "AWS",
                    "region": region,
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {
                        "Environment": random.choice(
                            ["production", "staging", "development"]
                        ),
                        "Team": f"team-{client_id}",
                        "ClientID": str(client_id),
                    },
                    "properties": {
                        "InstanceType": instance_type,
                        "State": "running",
                        "LaunchTime": "2024-01-15T10:30:00Z",
                    },
                },
                {
                    "resource_id": f"vol-client{client_id}-{random.randint(1000, 9999)}",
                    "service": "AWS.EBS",
                    "cloud_provider": "AWS",
                    "region": region,
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {"Environment": "production", "ClientID": str(client_id)},
                    "properties": {
                        "VolumeType": "gp3",
                        "Size": str(random.randint(50, 500)),
                        "State": "in-use",
                    },
                },
            ],
            "billing": [
                {
                    "service": "AWS.EC2",
                    "resource_id": f"i-client{client_id}-web-{random.randint(1000, 9999)}",
                    "region": region,
                    "usage_type": f"BoxUsage:{instance_type}",
                    "usage_amount": 720.0,
                    "usage_unit": "Hrs",
                    "unblended_cost": random.uniform(30.0, 150.0),
                    "amortized_cost": random.uniform(30.0, 150.0),
                    "credit": 0.0,
                    "bill_period_start": "2024-09-01T00:00:00Z",
                    "bill_period_end": "2024-09-30T23:59:59Z",
                    "tags": {"ClientID": str(client_id)},
                },
                {
                    "service": "AWS.EBS",
                    "resource_id": f"vol-client{client_id}-{random.randint(1000, 9999)}",
                    "region": region,
                    "usage_type": "EBS:VolumeUsage.gp3",
                    "usage_amount": random.uniform(50.0, 500.0),
                    "usage_unit": "GB-Mo",
                    "unblended_cost": random.uniform(5.0, 50.0),
                    "amortized_cost": random.uniform(5.0, 50.0),
                    "credit": 0.0,
                    "bill_period_start": "2024-09-01T00:00:00Z",
                    "bill_period_end": "2024-09-30T23:59:59Z",
                    "tags": {"ClientID": str(client_id)},
                },
            ],
            "metrics": [
                {
                    "resource_id": f"i-client{client_id}-web-{random.randint(1000, 9999)}",
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

    async def send_analyze_request(self, client_id: int) -> Dict[str, Any]:
        """Send a single analyze request for a client"""

        test_data = self.generate_test_data(client_id)
        start_time = time.time()

        try:
            async with self.session.post(
                f"{self.base_url}/analyze",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                end_time = time.time()
                duration = end_time - start_time

                if response.status == 200:
                    result = await response.json()
                    return {
                        "client_id": client_id,
                        "status": "success",
                        "status_code": response.status,
                        "duration": duration,
                        "response_size": len(str(result)),
                        "recommendations": len(
                            result.get("report", {}).get("recommendations", [])
                        ),
                        "monthly_savings": result.get("report", {}).get(
                            "total_monthly_savings", 0
                        ),
                        "processing_mode": (
                            "individual"
                            if result.get("individual_processing")
                            else "batch"
                        ),
                        "resources_analyzed": result.get("resources_analyzed", 0),
                    }
                else:
                    error_text = await response.text()
                    return {
                        "client_id": client_id,
                        "status": "error",
                        "status_code": response.status,
                        "duration": duration,
                        "error": error_text[:200],
                    }

        except asyncio.TimeoutError:
            return {
                "client_id": client_id,
                "status": "timeout",
                "status_code": 408,
                "duration": time.time() - start_time,
            }
        except Exception as e:
            return {
                "client_id": client_id,
                "status": "exception",
                "status_code": 500,
                "duration": time.time() - start_time,
                "error": str(e),
            }

    async def run_concurrent_test(
        self, num_clients: int, delay_between_clients: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Run concurrent requests from multiple clients"""

        print(f"🚀 Starting concurrent test with {num_clients} clients")
        if delay_between_clients > 0:
            print(
                f"⏱️  Staggered start with {delay_between_clients}s delay between clients"
            )
        print()

        # Create tasks for all clients
        tasks = []
        for client_id in range(1, num_clients + 1):
            if delay_between_clients > 0:
                await asyncio.sleep(delay_between_clients)

            task = asyncio.create_task(self.send_analyze_request(client_id))
            tasks.append(task)
            print(f"📤 Client {client_id} request started")

        # Wait for all requests to complete
        print("\n⏳ Waiting for all requests to complete...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "client_id": i + 1,
                        "status": "exception",
                        "status_code": 500,
                        "duration": 0,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and display test results"""

        print("\n" + "=" * 60)
        print("📊 CONCURRENT LOAD TEST RESULTS")
        print("=" * 60)

        # Overall statistics
        total_requests = len(results)
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]

        print(f"📈 Total Requests: {total_requests}")
        print(
            f"✅ Successful: {len(successful)} ({len(successful)/total_requests*100:.1f}%)"
        )
        print(f"❌ Failed: {len(failed)} ({len(failed)/total_requests*100:.1f}%)")
        print()

        # Performance statistics
        if successful:
            durations = [r["duration"] for r in successful]
            response_sizes = [r.get("response_size", 0) for r in successful]

            print("⏱️  PERFORMANCE METRICS")
            print("-" * 25)
            print(f"Average Response Time: {statistics.mean(durations):.2f}s")
            print(f"Median Response Time: {statistics.median(durations):.2f}s")
            print(f"Min Response Time: {min(durations):.2f}s")
            print(f"Max Response Time: {max(durations):.2f}s")
            print(f"Response Time Std Dev: {statistics.stdev(durations):.2f}s")
            print(
                f"Average Response Size: {statistics.mean(response_sizes):,.0f} bytes"
            )
            print()

            # Throughput calculation
            total_duration = max(durations) if durations else 0
            throughput = len(successful) / total_duration if total_duration > 0 else 0
            print(f"🚀 Throughput: {throughput:.2f} requests/second")
            print()

        # Business metrics
        if successful:
            recommendations_count = [r.get("recommendations", 0) for r in successful]
            monthly_savings = [r.get("monthly_savings", 0) for r in successful]
            resources_analyzed = [r.get("resources_analyzed", 0) for r in successful]

            print("💰 BUSINESS METRICS")
            print("-" * 20)
            print(f"Total Recommendations Generated: {sum(recommendations_count)}")
            print(
                f"Average Recommendations per Request: {statistics.mean(recommendations_count):.1f}"
            )
            print(f"Total Monthly Savings Identified: ${sum(monthly_savings):,.2f}")
            print(
                f"Average Monthly Savings per Request: ${statistics.mean(monthly_savings):.2f}"
            )
            print(f"Total Resources Analyzed: {sum(resources_analyzed)}")
            print()

        # Processing mode breakdown
        if successful:
            batch_mode = len(
                [r for r in successful if r.get("processing_mode") == "batch"]
            )
            individual_mode = len(
                [r for r in successful if r.get("processing_mode") == "individual"]
            )

            print("🔄 PROCESSING MODE BREAKDOWN")
            print("-" * 30)
            print(
                f"Batch Processing: {batch_mode} requests ({batch_mode/len(successful)*100:.1f}%)"
            )
            print(
                f"Individual Processing: {individual_mode} requests ({individual_mode/len(successful)*100:.1f}%)"
            )
            print()

        # Error analysis
        if failed:
            print("❌ ERROR ANALYSIS")
            print("-" * 17)
            error_types = {}
            for error in failed:
                error_type = error.get("status", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                print(f"{error_type.title()}: {count} requests")

            # Show sample errors
            print("\n📝 Sample Errors:")
            for error in failed[:3]:
                client_id = error.get("client_id", "N/A")
                status = error.get("status", "unknown")
                error_msg = error.get("error", "No error message")[:100]
                print(f"   Client {client_id} ({status}): {error_msg}...")
            print()

        # Individual client results
        print("👥 CLIENT-BY-CLIENT RESULTS")
        print("-" * 30)
        for result in results[:10]:  # Show first 10 clients
            client_id = result.get("client_id", "N/A")
            status = result.get("status", "unknown")
            duration = result.get("duration", 0)

            if status == "success":
                recs = result.get("recommendations", 0)
                savings = result.get("monthly_savings", 0)
                mode = result.get("processing_mode", "unknown")
                print(
                    f"   Client {client_id:2d}: ✅ {duration:.2f}s | {recs} recs | ${savings:.2f}/mo | {mode}"
                )
            else:
                error = result.get("error", "Unknown error")[:50]
                print(
                    f"   Client {client_id:2d}: ❌ {duration:.2f}s | {status} | {error}..."
                )

        if len(results) > 10:
            print(f"   ... and {len(results) - 10} more clients")
        print()


async def main():
    """Main test execution"""

    print("🔥 LLM Cost Recommendation API - Concurrent Load Test")
    print("=" * 55)
    print()

    # Check server availability first
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "http://localhost:8000/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print("✅ Server is available and ready for testing")
                else:
                    print(f"❌ Server returned status {response.status}")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("Please make sure the API server is running on localhost:8000")
            return

    print()

    # Test scenarios
    test_scenarios = [
        {"clients": 3, "delay": 0.0, "name": "Burst Test - 3 simultaneous clients"},
        {
            "clients": 5,
            "delay": 0.5,
            "name": "Staggered Test - 5 clients with 0.5s delay",
        },
        {
            "clients": 8,
            "delay": 1.0,
            "name": "High Load Test - 8 clients with 1s delay",
        },
    ]

    async with ConcurrentAPITester() as tester:
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"🎯 Test Scenario {i}: {scenario['name']}")
            print("-" * 50)

            start_time = time.time()
            results = await tester.run_concurrent_test(
                num_clients=scenario["clients"], delay_between_clients=scenario["delay"]
            )
            total_time = time.time() - start_time

            print(f"\n⏱️  Total Test Duration: {total_time:.2f} seconds")
            tester.analyze_results(results)

            # Wait between scenarios
            if i < len(test_scenarios):
                print("⏳ Waiting 10 seconds before next test scenario...")
                await asyncio.sleep(10)
                print()


if __name__ == "__main__":
    asyncio.run(main())
