#!/usr/bin/env python3
"""
Test to see all recommendations including warnings and low-confidence ones
"""

import requests
import json


def test_with_warnings():
    """Test with warnings display to see low-confidence recommendations"""

    request_data = {
        "resources": [
            {
                "resource_id": "i-overprovisioned-web-server",
                "service": "AWS.EC2",
                "region": "us-east-1",
                "availability_zone": "us-east-1a",
                "tags": {
                    "Name": "Web Server - Production",
                    "Environment": "production",
                },
                "properties": {
                    "instance_type": "t3.large",
                    "platform": "Linux/UNIX",
                    "state": "running",
                },
            },
            {
                "resource_id": "vol-underutilized-storage",
                "service": "AWS.EBS",
                "region": "us-east-1",
                "availability_zone": "us-east-1a",
                "tags": {"Name": "Web Server Storage", "Environment": "production"},
                "properties": {
                    "volume_type": "gp3",
                    "size": 500,
                    "iops": 3000,
                    "state": "in-use",
                },
            },
        ],
        "metrics": [
            {
                "resource_id": "i-overprovisioned-web-server",
                "timestamp": "2024-08-31T23:59:59Z",
                "period_days": 30,
                "is_idle": False,
                "cpu_utilization_p50": 8.5,
                "cpu_utilization_p90": 15.2,
                "cpu_utilization_p95": 18.7,
                "memory_utilization_p50": 25.4,
                "memory_utilization_p90": 35.8,
            },
            {
                "resource_id": "vol-underutilized-storage",
                "timestamp": "2024-08-31T23:59:59Z",
                "period_days": 30,
                "is_idle": False,
                "iops_read": 25.3,
                "iops_write": 15.7,
                "throughput_read": 1.2,
                "throughput_write": 0.8,
                "metrics": {"volume_utilization": 35.2, "queue_depth": 1.2},
            },
        ],
        "billing": [
            {
                "service": "AWS.EC2",
                "resource_id": "i-overprovisioned-web-server",
                "region": "us-east-1",
                "usage_type": "EC2-Instance",
                "usage_amount": 744.0,
                "usage_unit": "Hrs",
                "unblended_cost": 89.28,
                "amortized_cost": 89.28,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
            {
                "service": "AWS.EBS",
                "resource_id": "vol-underutilized-storage",
                "region": "us-east-1",
                "usage_type": "EBS:VolumeUsage.gp3",
                "usage_amount": 500.0,
                "usage_unit": "GB-Month",
                "unblended_cost": 40.0,
                "amortized_cost": 40.0,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
        ],
        "analysis_mode": "individual",
    }

    print("🚀 Testing with Warnings Display")
    print("=" * 50)
    print(f"Resources: {len(request_data['resources'])}")
    for resource in request_data["resources"]:
        print(f"  • {resource['service']}: {resource['resource_id']}")

    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Analysis completed!")
            print(
                f"Recommendations found: {result.get('report', {}).get('total_recommendations', 0)}"
            )

            recommendations = result.get("report", {}).get("recommendations", [])
            total_savings = sum(
                rec.get("estimated_monthly_savings", 0) for rec in recommendations
            )
            print(f"Total monthly savings: ${total_savings:.2f}")

            for i, rec in enumerate(recommendations, 1):
                print(f"\n💡 Recommendation #{i}:")
                print(f"  Resource: {rec.get('resource_id')}")
                print(f"  Service: {rec.get('service')}")
                print(f"  Type: {rec.get('recommendation_type')}")
                print(
                    f"  Savings: ${rec.get('estimated_monthly_savings', 0):.2f}/month"
                )
                print(f"  Confidence: {rec.get('confidence_score', 0):.0%}")
                print(f"  Risk: {rec.get('risk_level', 'unknown')}")

                # Display warnings if present
                warnings = rec.get("warnings", [])
                if warnings:
                    print(f"  ⚠️  WARNINGS:")
                    for warning in warnings:
                        print(f"    • {warning}")

                # Show some recommendation details
                if rec.get("rationale"):
                    print(f"  Rationale: {rec.get('rationale')[:100]}...")

            # Also show any global warnings or issues
            if "issues" in result.get("report", {}):
                print(f"\n⚠️  Analysis Issues:")
                for issue in result["report"]["issues"]:
                    print(f"  • {issue}")

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_with_warnings()
