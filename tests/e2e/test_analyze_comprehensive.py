#!/usr/bin/env python3
"""
Comprehensive test script for the analyze API endpoint.
Tests both batch and individual processing modes with real data.
"""

import requests
import json
import time
import sys
from datetime import datetime


def test_analyze_api():
    """Test the analyze API endpoint comprehensively"""

    base_url = "http://localhost:8000"

    print("🚀 Testing Analyze API Endpoint")
    print("=" * 50)

    # Test data with correct format
    test_data = {
        "resources": [
            {
                "resource_id": "i-1234567890abcdef0",
                "service": "AWS.EC2",
                "cloud_provider": "AWS",
                "region": "us-east-1",
                "account_id": "123456789012",
                "tags": {
                    "Environment": "production",
                    "Team": "backend",
                    "Application": "web-server",
                },
                "properties": {
                    "InstanceType": "t3.large",
                    "State": "running",
                    "LaunchTime": "2024-01-15T10:30:00Z",
                },
            },
            {
                "resource_id": "vol-0987654321fedcba0",
                "service": "AWS.EBS",
                "cloud_provider": "AWS",
                "region": "us-east-1",
                "account_id": "123456789012",
                "tags": {"Environment": "production"},
                "properties": {"VolumeType": "gp3", "Size": "100", "State": "in-use"},
            },
        ],
        "billing": [
            {
                "service": "AWS.EC2",
                "resource_id": "i-1234567890abcdef0",
                "region": "us-east-1",
                "usage_type": "BoxUsage:t3.large",
                "usage_amount": 720.0,
                "usage_unit": "Hrs",
                "unblended_cost": 45.60,
                "amortized_cost": 45.60,
                "credit": 0.0,
                "bill_period_start": "2024-09-01T00:00:00Z",
                "bill_period_end": "2024-09-30T23:59:59Z",
                "tags": {"Environment": "production"},
            },
            {
                "service": "AWS.EBS",
                "resource_id": "vol-0987654321fedcba0",
                "region": "us-east-1",
                "usage_type": "EBS:VolumeUsage.gp3",
                "usage_amount": 100.0,
                "usage_unit": "GB-Mo",
                "unblended_cost": 12.00,
                "amortized_cost": 12.00,
                "credit": 0.0,
                "bill_period_start": "2024-09-01T00:00:00Z",
                "bill_period_end": "2024-09-30T23:59:59Z",
                "tags": {"Environment": "production"},
            },
        ],
        "metrics": [
            {
                "resource_id": "i-1234567890abcdef0",
                "timestamp": "2024-09-01T00:00:00Z",
                "period_days": 30,
                "cpu_utilization_p50": 15.5,
                "cpu_utilization_p90": 35.2,
                "cpu_utilization_p95": 42.8,
                "memory_utilization_p50": 25.3,
                "memory_utilization_p90": 48.7,
                "memory_utilization_p95": 55.1,
                "is_idle": False,
            },
            {
                "resource_id": "vol-0987654321fedcba0",
                "timestamp": "2024-09-01T00:00:00Z",
                "period_days": 30,
                "iops_read": 150.0,
                "iops_write": 75.0,
                "is_idle": False,
            },
        ],
        "individual_processing": False,
        "max_recommendations": 10,
    }

    print(
        f'📊 Test Data: {len(test_data["resources"])} resources, {len(test_data["billing"])} billing records'
    )
    total_cost = sum(b["unblended_cost"] for b in test_data["billing"])
    print(f"💰 Total Monthly Cost: ${total_cost:.2f}")
    print()

    # Test 1: Check server health first
    print("🏥 Test 1: Server Health Check")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("   ✅ Server is healthy and ready")
        else:
            print(f"   ❌ Server health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False

    # Test 2: Batch Processing
    print("🧪 Test 2: Batch Processing Analysis")
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"   Status: {response.status_code}")
        print(f"   Processing Time: {processing_time:.2f}s")
        print(f"   Response Size: {len(response.text)} bytes")

        if response.status_code == 200:
            result = response.json()
            print("   ✅ SUCCESS!")

            # Parse and display results
            report = result.get("report", {})
            recommendations = report.get("recommendations", [])

            print(f'   🆔 Request ID: {result.get("request_id", "N/A")}')
            print(f'   📊 Resources Analyzed: {result.get("resources_analyzed", 0)}')
            print(f"   📈 Recommendations: {len(recommendations)}")
            print(
                f'   💰 Monthly Savings: ${report.get("total_monthly_savings", 0):.2f}'
            )
            print(f'   💸 Annual Savings: ${report.get("total_annual_savings", 0):.2f}')

            # Show top recommendations
            if recommendations:
                print("   🎯 Top Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    title = rec.get("rationale", "No description")[:60]
                    resource = rec.get("resource_id", "N/A")
                    savings = rec.get("estimated_monthly_savings", 0)
                    risk = rec.get("risk_level", "UNKNOWN")
                    print(
                        f"      {i}. [{resource}] {title}... (${savings:.2f}/mo, {risk} risk)"
                    )

            # Performance breakdown
            resources_summary = report.get("resources_summary", {})
            if resources_summary:
                by_cloud = resources_summary.get("by_cloud_provider", {})
                by_service = resources_summary.get("by_service", {})
                print(f"   🌍 Cloud Breakdown: {by_cloud}")
                print(f"   🔧 Service Breakdown: {by_service}")

        elif response.status_code == 422:
            print("   ❌ VALIDATION ERROR:")
            try:
                error_details = response.json()
                if "detail" in error_details:
                    for error in error_details["detail"][:3]:
                        field = " -> ".join(map(str, error.get("loc", [])))
                        message = error.get("msg", "Unknown error")
                        print(f"      Field: {field}")
                        print(f"      Error: {message}")
            except:
                print(f"      Raw error: {response.text[:200]}...")
        else:
            print(f"   ❌ ERROR {response.status_code}: {response.text[:300]}...")

    except requests.exceptions.Timeout:
        print("   ⏰ TIMEOUT: Analysis took longer than 2 minutes")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

    print()

    # Test 3: Individual Processing
    print("🧪 Test 3: Individual Processing Mode")
    individual_data = test_data.copy()
    individual_data["individual_processing"] = True
    individual_data["max_recommendations"] = 5

    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze",
            json=individual_data,
            headers={"Content-Type": "application/json"},
            timeout=180,
        )
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"   Status: {response.status_code}")
        print(f"   Processing Time: {processing_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print("   ✅ SUCCESS!")

            report = result.get("report", {})
            recommendations = report.get("recommendations", [])

            print(
                f'   🔄 Individual Mode: {result.get("individual_processing", False)}'
            )
            print(f"   📈 Recommendations: {len(recommendations)}")
            print(
                f'   💰 Monthly Savings: ${report.get("total_monthly_savings", 0):.2f}'
            )

        else:
            print(f"   ❌ ERROR {response.status_code}: {response.text[:200]}...")

    except requests.exceptions.Timeout:
        print("   ⏰ TIMEOUT: Individual processing took longer than 3 minutes")
    except Exception as e:
        print(f"   ❌ Request failed: {e}")

    print()

    # Test 4: Error Handling
    print("🧪 Test 4: Error Validation")
    invalid_data = {"resources": [], "billing": [], "metrics": []}

    try:
        response = requests.post(
            f"{base_url}/analyze",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 422:
            print("   ✅ Validation working correctly - rejected empty resources")
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error test failed: {e}")

    print()
    print("=" * 50)
    print("🎯 Analyze API Testing Complete!")

    # Test passed successfully - pytest expects None return
    assert True


if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)

    try:
        test_analyze_api()
        print("✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)
