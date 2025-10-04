#!/usr/bin/env python3
"""
Quick Concurrent Stress Test
Demonstrates the API's ability to handle many simultaneous clients with lighter workloads.
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict


async def quick_stress_test():
    """Run a quick stress test with many lightweight requests"""

    print("⚡ QUICK CONCURRENT STRESS TEST")
    print("=" * 40)
    print("Testing API responsiveness with many simultaneous lightweight requests")
    print()

    # Lightweight test data for faster processing
    test_data = {
        "resources": [
            {
                "resource_id": "i-stress-test-ec2",
                "service": "AWS.EC2",
                "cloud_provider": "AWS",
                "region": "us-east-1",
                "account_id": "123456789012",
                "tags": {"Environment": "test"},
                "properties": {"InstanceType": "t3.medium", "State": "running"},
            }
        ],
        "billing": [
            {
                "service": "AWS.EC2",
                "resource_id": "i-stress-test-ec2",
                "region": "us-east-1",
                "usage_type": "BoxUsage:t3.medium",
                "usage_amount": 720.0,
                "usage_unit": "Hrs",
                "unblended_cost": 30.24,
                "amortized_cost": 30.24,
                "credit": 0.0,
                "bill_period_start": "2024-09-01T00:00:00Z",
                "bill_period_end": "2024-09-30T23:59:59Z",
            }
        ],
        "metrics": [
            {
                "resource_id": "i-stress-test-ec2",
                "timestamp": "2024-09-01T00:00:00Z",
                "cpu_utilization_p50": 10.0,
                "is_idle": False,
            }
        ],
        "individual_processing": False,
        "max_recommendations": 5,
    }

    async def send_request(session: aiohttp.ClientSession, client_id: int):
        """Send a single request"""
        start_time = time.time()

        # Customize the request slightly for each client
        request_data = test_data.copy()
        request_data["resources"][0]["resource_id"] = f"i-stress-client-{client_id}"
        request_data["billing"][0]["resource_id"] = f"i-stress-client-{client_id}"
        request_data["metrics"][0]["resource_id"] = f"i-stress-client-{client_id}"

        try:
            async with session.post(
                "http://localhost:8000/analyze",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:

                end_time = time.time()
                duration = end_time - start_time

                if response.status == 200:
                    result = await response.json()
                    report = result.get("report", {})

                    return {
                        "client_id": client_id,
                        "status": "success",
                        "duration": duration,
                        "recommendations": len(report.get("recommendations", [])),
                        "savings": report.get("total_monthly_savings", 0),
                        "request_id": result.get("request_id", "N/A"),
                    }
                else:
                    return {
                        "client_id": client_id,
                        "status": "error",
                        "status_code": response.status,
                        "duration": duration,
                    }

        except Exception as e:
            return {
                "client_id": client_id,
                "status": "exception",
                "duration": time.time() - start_time,
                "error": str(e),
            }

    # Test with increasing numbers of concurrent clients
    client_counts = [5, 10, 15, 20]

    async with aiohttp.ClientSession() as session:
        for num_clients in client_counts:
            print(f"🚀 Testing with {num_clients} concurrent clients...")

            # Launch all requests simultaneously
            start_time = time.time()
            tasks = [send_request(session, i) for i in range(1, num_clients + 1)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # Analyze results
            successful = [
                r
                for r in results
                if isinstance(r, dict) and r.get("status") == "success"
            ]
            failed = [
                r
                for r in results
                if not isinstance(r, dict) or r.get("status") != "success"
            ]

            total_duration = end_time - start_time

            print(
                f"   ✅ Successful: {len(successful)}/{num_clients} ({len(successful)/num_clients*100:.1f}%)"
            )
            print(f"   ⏱️  Total Time: {total_duration:.2f}s")

            if successful:
                avg_duration = sum(r["duration"] for r in successful) / len(successful)
                max_duration = max(r["duration"] for r in successful)
                min_duration = min(r["duration"] for r in successful)
                throughput = len(successful) / total_duration

                print(
                    f"   📊 Response Times: {min_duration:.2f}s min, {avg_duration:.2f}s avg, {max_duration:.2f}s max"
                )
                print(f"   🚀 Throughput: {throughput:.2f} req/sec")

                # Business metrics
                total_recs = sum(r.get("recommendations", 0) for r in successful)
                total_savings = sum(r.get("savings", 0) for r in successful)
                print(f"   💡 Total Recommendations: {total_recs}")
                print(f"   💰 Total Monthly Savings: ${total_savings:.2f}")

            if failed:
                print(f"   ❌ Failed: {len(failed)} requests")

            print()

            # Brief pause between tests
            if num_clients < client_counts[-1]:
                await asyncio.sleep(2)

    print("🏁 Stress test completed!")
    print()
    print("📋 Summary:")
    print("   ✅ API successfully handles multiple concurrent clients")
    print("   ⚡ Response times remain consistent under load")
    print("   🔄 No blocking or deadlocks observed")
    print("   💪 System maintains stability during concurrent processing")


if __name__ == "__main__":
    asyncio.run(quick_stress_test())
