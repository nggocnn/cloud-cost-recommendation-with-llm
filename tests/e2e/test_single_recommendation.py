#!/usr/bin/env python3
"""
Simple test to get a single recommendation and see the full output
"""

import requests
import json


def test_single_recommendation():
    """Test with a single EC2 instance to see the recommendation output"""

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
                    "Team": "web-team",
                },
                "properties": {
                    "instance_type": "t3.large",
                    "platform": "Linux/UNIX",
                    "state": "running",
                },
            }
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
                "cpu_utilization_min": 2.1,
                "cpu_utilization_max": 22.3,
                "memory_utilization_p50": 25.4,
                "memory_utilization_p90": 35.8,
                "memory_utilization_p95": 42.1,
            }
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
            }
        ],
        "analysis_mode": "individual",
    }

    print("🚀 Testing Single Resource Recommendation Generation")
    print("=" * 60)
    print(f"Resource: {request_data['resources'][0]['resource_id']}")
    print(f"Service: {request_data['resources'][0]['service']}")
    print(
        f"Instance Type: {request_data['resources'][0]['properties']['instance_type']}"
    )
    print(f"CPU Utilization P50: {request_data['metrics'][0]['cpu_utilization_p50']}%")
    print(f"Monthly Cost: ${request_data['billing'][0]['unblended_cost']}")

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
            print(f"Raw response length: {len(response.text)} characters")

            # Pretty print the entire response
            print(f"\n📄 FULL API RESPONSE:")
            print("=" * 60)
            print(json.dumps(result, indent=2))

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_single_recommendation()
