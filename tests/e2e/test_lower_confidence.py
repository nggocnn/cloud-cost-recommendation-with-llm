#!/usr/bin/env python3
"""
Test with lower confidence thresholds to see all LLM recommendations
"""

import requests
import json


def test_with_lower_confidence():
    """Test with lower confidence threshold to see all recommendations"""

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
            {
                "resource_id": "bucket-large-storage-costs",
                "service": "AWS.S3",
                "region": "us-east-1",
                "tags": {"Name": "Legacy Data Archive", "Environment": "production"},
                "properties": {
                    "storage_class": "STANDARD",
                    "encryption": "AES256",
                    "versioning": "Enabled",
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
            {
                "resource_id": "bucket-large-storage-costs",
                "timestamp": "2024-08-31T23:59:59Z",
                "period_days": 30,
                "is_idle": False,
                "storage_size_gb": 15000.0,
                "request_count_get": 2500,
                "request_count_put": 150,
                "data_transfer_gb": 45.2,
                "access_frequency": 0.1,  # Low access frequency (numeric)
                "last_access_days": 45,
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
            {
                "service": "AWS.S3",
                "resource_id": "bucket-large-storage-costs",
                "region": "us-east-1",
                "usage_type": "S3:StandardStorage",
                "usage_amount": 15000.0,
                "usage_unit": "GB-Month",
                "unblended_cost": 345.0,
                "amortized_cost": 345.0,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
        ],
        "analysis_mode": "individual",
        # Try to override confidence threshold (though API might not support this)
        "config_overrides": {
            "confidence_threshold": 0.5  # Lower threshold to see more recommendations
        },
    }

    print("🚀 Testing with Lower Confidence Threshold")
    print("=" * 50)
    print(f"Resources: {len(request_data['resources'])}")
    for resource in request_data["resources"]:
        print(f"  • {resource['service']}: {resource['resource_id']}")
    print(f"Config Override: confidence_threshold = 0.5 (instead of default 0.7)")

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

                # Show some recommendation details
                if rec.get("rationale"):
                    print(f"  Rationale: {rec.get('rationale')[:100]}...")

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_with_lower_confidence()
