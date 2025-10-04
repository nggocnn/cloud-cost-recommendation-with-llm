#!/usr/bin/env python3
"""
Intensive Concurrent Load Test for Analyze API
Tests real analysis workload with multiple concurrent clients.
"""

import asyncio
import aiohttp
import time
import json
import random
from typing import List, Dict, Any
import statistics


class IntensiveLoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def create_realistic_analysis_request(self, client_id: int) -> Dict[str, Any]:
        """Create a realistic analysis request that will trigger full processing"""

        return {
            "resources": [
                {
                    "resource_id": f"i-prod-web-{client_id:03d}-{random.randint(1000, 9999)}",
                    "service": "AWS.EC2",
                    "cloud_provider": "AWS",
                    "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {
                        "Environment": "production",
                        "Team": f"team-{client_id}",
                        "Application": f"web-app-{client_id}",
                        "CostCenter": "engineering",
                    },
                    "properties": {
                        "InstanceType": random.choice(
                            ["t3.large", "t3.xlarge", "m5.large", "c5.xlarge"]
                        ),
                        "State": "running",
                        "LaunchTime": "2024-01-15T10:30:00Z",
                        "VpcId": f"vpc-{client_id:08x}",
                    },
                },
                {
                    "resource_id": f"vol-prod-{client_id:03d}-{random.randint(1000, 9999)}",
                    "service": "AWS.EBS",
                    "cloud_provider": "AWS",
                    "region": "us-east-1",
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {
                        "Environment": "production",
                        "AttachedInstance": f"i-prod-web-{client_id:03d}",
                    },
                    "properties": {
                        "VolumeType": "gp3",
                        "Size": str(random.randint(100, 1000)),
                        "State": "in-use",
                        "IOPS": str(random.randint(3000, 10000)),
                    },
                },
                {
                    "resource_id": f"bucket-logs-client-{client_id}",
                    "service": "AWS.S3",
                    "cloud_provider": "AWS",
                    "region": "us-east-1",
                    "account_id": f"{123456789000 + client_id:012d}",
                    "tags": {
                        "Environment": "production",
                        "Purpose": "application-logs",
                        "ClientID": str(client_id),
                    },
                    "properties": {
                        "StorageClass": "STANDARD",
                        "CreationDate": "2024-01-01T00:00:00Z",
                    },
                },
            ],
            "billing": [
                {
                    "service": "AWS.EC2",
                    "resource_id": f"i-prod-web-{client_id:03d}-{random.randint(1000, 9999)}",
                    "region": "us-east-1",
                    "usage_type": f"BoxUsage:t3.xlarge",
                    "usage_amount": 720.0,
                    "usage_unit": "Hrs",
                    "unblended_cost": random.uniform(80.0, 150.0),
                    "amortized_cost": random.uniform(80.0, 150.0),
                    "credit": 0.0,
                    "bill_period_start": "2024-09-01T00:00:00Z",
                    "bill_period_end": "2024-09-30T23:59:59Z",
                    "tags": {"Environment": "production", "ClientID": str(client_id)},
                },
                {
                    "service": "AWS.EBS",
                    "resource_id": f"vol-prod-{client_id:03d}-{random.randint(1000, 9999)}",
                    "region": "us-east-1",
                    "usage_type": "EBS:VolumeUsage.gp3",
                    "usage_amount": random.uniform(100.0, 1000.0),
                    "usage_unit": "GB-Mo",
                    "unblended_cost": random.uniform(10.0, 100.0),
                    "amortized_cost": random.uniform(10.0, 100.0),
                    "credit": 0.0,
                    "bill_period_start": "2024-09-01T00:00:00Z",
                    "bill_period_end": "2024-09-30T23:59:59Z",
                    "tags": {"Environment": "production", "ClientID": str(client_id)},
                },
                {
                    "service": "AWS.S3",
                    "resource_id": f"bucket-logs-client-{client_id}",
                    "region": "us-east-1",
                    "usage_type": "StorageUsage",
                    "usage_amount": random.uniform(100.0, 2000.0),
                    "usage_unit": "GB-Mo",
                    "unblended_cost": random.uniform(5.0, 50.0),
                    "amortized_cost": random.uniform(5.0, 50.0),
                    "credit": 0.0,
                    "bill_period_start": "2024-09-01T00:00:00Z",
                    "bill_period_end": "2024-09-30T23:59:59Z",
                    "tags": {"Environment": "production", "ClientID": str(client_id)},
                },
            ],
            "metrics": [
                {
                    "resource_id": f"i-prod-web-{client_id:03d}-{random.randint(1000, 9999)}",
                    "timestamp": "2024-09-01T00:00:00Z",
                    "period_days": 30,
                    # Low utilization to trigger rightsizing recommendations
                    "cpu_utilization_p50": random.uniform(5.0, 25.0),
                    "cpu_utilization_p90": random.uniform(15.0, 40.0),
                    "cpu_utilization_p95": random.uniform(25.0, 50.0),
                    "memory_utilization_p50": random.uniform(20.0, 45.0),
                    "memory_utilization_p90": random.uniform(35.0, 60.0),
                    "memory_utilization_p95": random.uniform(45.0, 70.0),
                    "is_idle": False,
                    "peak_usage_hours": [9, 10, 11, 14, 15, 16],
                },
                {
                    "resource_id": f"vol-prod-{client_id:03d}-{random.randint(1000, 9999)}",
                    "timestamp": "2024-09-01T00:00:00Z",
                    "period_days": 30,
                    "iops_read": random.uniform(50.0, 500.0),
                    "iops_write": random.uniform(25.0, 250.0),
                    "throughput_read": random.uniform(500.0, 5000.0),
                    "throughput_write": random.uniform(250.0, 2500.0),
                    "is_idle": False,
                },
            ],
            "individual_processing": False,  # Use batch for faster processing
            "max_recommendations": 10,
        }

    async def send_concurrent_requests(self, num_clients: int):
        """Send multiple concurrent analysis requests"""

        print(f"🚀 Launching {num_clients} concurrent analysis requests...")
        print("📊 Each request analyzes 3 resources (EC2, EBS, S3)")
        print()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=180)
        ) as session:

            # Create all tasks
            tasks = []
            start_times = {}

            for client_id in range(1, num_clients + 1):
                request_data = await self.create_realistic_analysis_request(client_id)
                start_times[client_id] = time.time()

                task = asyncio.create_task(
                    self.analyze_request(
                        session, client_id, request_data, start_times[client_id]
                    )
                )
                tasks.append(task)

                print(f"📤 Client {client_id:2d}: Request queued")

                # Small delay to prevent overwhelming
                if client_id % 3 == 0:
                    await asyncio.sleep(0.1)

            print(f"\n⏳ Waiting for all {num_clients} requests to complete...")
            print("   (This may take several minutes for full analysis)")

            # Execute all requests concurrently
            global_start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            global_end_time = time.time()

            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "client_id": i + 1,
                            "status": "exception",
                            "error": str(result),
                            "duration": 0,
                        }
                    )
                else:
                    processed_results.append(result)

            return processed_results, global_end_time - global_start_time

    async def analyze_request(
        self,
        session: aiohttp.ClientSession,
        client_id: int,
        request_data: dict,
        start_time: float,
    ):
        """Analyze a single request"""

        try:
            async with session.post(
                f"{self.base_url}/analyze",
                json=request_data,
                headers={"Content-Type": "application/json"},
            ) as response:

                end_time = time.time()
                duration = end_time - start_time

                if response.status == 200:
                    result = await response.json()
                    report = result.get("report", {})
                    recommendations = report.get("recommendations", [])

                    # Extract cost data from request for context
                    total_cost = sum(
                        b["unblended_cost"] for b in request_data["billing"]
                    )

                    return {
                        "client_id": client_id,
                        "status": "success",
                        "duration": duration,
                        "resources_analyzed": len(request_data["resources"]),
                        "recommendations_generated": len(recommendations),
                        "monthly_savings": report.get("total_monthly_savings", 0),
                        "annual_savings": report.get("total_annual_savings", 0),
                        "original_monthly_cost": total_cost,
                        "processing_mode": result.get("individual_processing", False),
                        "request_id": result.get("request_id", "N/A"),
                        "response_size": len(str(result)),
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
                "duration": time.time() - start_time,
                "error": "Request timed out after 180 seconds",
            }
        except Exception as e:
            return {
                "client_id": client_id,
                "status": "exception",
                "duration": time.time() - start_time,
                "error": str(e),
            }

    def print_detailed_results(self, results: List[Dict], total_duration: float):
        """Print comprehensive results analysis"""

        print("\n" + "=" * 70)
        print("🏆 INTENSIVE CONCURRENT LOAD TEST RESULTS")
        print("=" * 70)

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]

        # Overall metrics
        print(f"📊 OVERALL PERFORMANCE")
        print(f"   Total Clients: {len(results)}")
        print(
            f"   Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)"
        )
        print(f"   Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"   Total Test Duration: {total_duration:.2f} seconds")
        print()

        if successful:
            # Performance analysis
            durations = [r["duration"] for r in successful]
            response_sizes = [r.get("response_size", 0) for r in successful]

            print(f"⚡ RESPONSE TIME ANALYSIS")
            print(f"   Average: {statistics.mean(durations):.2f}s")
            print(f"   Median: {statistics.median(durations):.2f}s")
            print(f"   Min: {min(durations):.2f}s")
            print(f"   Max: {max(durations):.2f}s")
            print(
                f"   95th Percentile: {sorted(durations)[int(len(durations)*0.95)]:.2f}s"
            )
            if len(durations) > 1:
                print(f"   Standard Deviation: {statistics.stdev(durations):.2f}s")
            print()

            # Throughput metrics
            requests_per_second = len(successful) / total_duration
            print(f"🚀 THROUGHPUT METRICS")
            print(f"   Requests per Second: {requests_per_second:.2f} req/s")
            print(
                f"   Average Response Size: {statistics.mean(response_sizes):,.0f} bytes"
            )
            print(f"   Total Data Transferred: {sum(response_sizes):,.0f} bytes")
            print()

            # Business metrics
            total_resources = sum(r.get("resources_analyzed", 0) for r in successful)
            total_recommendations = sum(
                r.get("recommendations_generated", 0) for r in successful
            )
            total_monthly_savings = sum(r.get("monthly_savings", 0) for r in successful)
            total_annual_savings = sum(r.get("annual_savings", 0) for r in successful)
            total_original_cost = sum(
                r.get("original_monthly_cost", 0) for r in successful
            )

            print(f"💰 BUSINESS INTELLIGENCE")
            print(f"   Resources Analyzed: {total_resources}")
            print(f"   Recommendations Generated: {total_recommendations}")
            print(f"   Total Monthly Costs Analyzed: ${total_original_cost:,.2f}")
            print(f"   Total Monthly Savings Identified: ${total_monthly_savings:,.2f}")
            print(f"   Total Annual Savings Potential: ${total_annual_savings:,.2f}")
            if total_original_cost > 0:
                savings_percentage = (total_monthly_savings / total_original_cost) * 100
                print(f"   Average Savings Percentage: {savings_percentage:.1f}%")
            print(
                f"   Avg Recommendations per Request: {total_recommendations/len(successful):.1f}"
            )
            print()

            # Client-by-client breakdown
            print(f"👥 CLIENT PERFORMANCE BREAKDOWN")
            print(
                f"   {'Client':>6} {'Status':>8} {'Duration':>10} {'Recs':>5} {'Savings':>10} {'ROI':>8}"
            )
            print(f"   {'-'*6} {'-'*8} {'-'*10} {'-'*5} {'-'*10} {'-'*8}")

            for result in sorted(successful, key=lambda x: x["duration"]):
                client_id = result["client_id"]
                duration = result["duration"]
                recs = result.get("recommendations_generated", 0)
                savings = result.get("monthly_savings", 0)
                original = result.get("original_monthly_cost", 0)
                roi = (savings / original * 100) if original > 0 else 0

                print(
                    f"   {client_id:>6} {'✅':>8} {duration:>8.2f}s {recs:>5} ${savings:>8.2f} {roi:>6.1f}%"
                )
            print()

        # Error analysis
        if failed:
            print(f"❌ ERROR ANALYSIS")
            error_types = {}
            for error in failed:
                error_type = error.get("status", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            for error_type, count in error_types.items():
                print(f"   {error_type.title()}: {count} occurrences")

            print("\n   Sample Errors:")
            for error in failed[:3]:
                client_id = error.get("client_id", "N/A")
                error_msg = error.get("error", "No message")[:80]
                print(f"   Client {client_id}: {error_msg}...")
            print()

        # Concurrency insights
        if len(successful) > 1:
            print(f"🔄 CONCURRENCY ANALYSIS")
            concurrent_efficiency = (
                (min(durations) / max(durations)) * 100 if successful else 0
            )
            print(f"   Concurrency Efficiency: {concurrent_efficiency:.1f}%")
            print(f"   (Higher is better - shows consistent performance under load)")

            if statistics.stdev(durations) / statistics.mean(durations) < 0.2:
                print(f"   ✅ Performance is highly consistent across clients")
            else:
                print(f"   ⚠️  Performance varies significantly under concurrent load")
            print()


async def main():
    """Execute intensive load test"""

    print("🔥 INTENSIVE CONCURRENT ANALYSIS LOAD TEST")
    print("=" * 50)
    print("Testing real-world cost analysis workloads with multiple concurrent clients")
    print()

    # Verify server is available
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8000/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    print("❌ Server health check failed")
                    return
                print("✅ Server is healthy and ready")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Test configurations
    test_configs = [
        {"clients": 2, "name": "Light Load - 2 Concurrent Clients"},
        {"clients": 4, "name": "Medium Load - 4 Concurrent Clients"},
        {"clients": 6, "name": "Heavy Load - 6 Concurrent Clients"},
    ]

    tester = IntensiveLoadTester()

    for i, config in enumerate(test_configs, 1):
        print(f"\n🎯 Test {i}: {config['name']}")
        print("-" * 60)

        try:
            results, total_duration = await tester.send_concurrent_requests(
                config["clients"]
            )
            tester.print_detailed_results(results, total_duration)

            # Cool down between tests
            if i < len(test_configs):
                print("😴 Cooling down for 15 seconds before next test...")
                await asyncio.sleep(15)

        except Exception as e:
            print(f"❌ Test failed: {e}")

    print("\n🎉 All concurrent load tests completed!")
    print("The API has been thoroughly tested for concurrent performance.")


if __name__ == "__main__":
    asyncio.run(main())
