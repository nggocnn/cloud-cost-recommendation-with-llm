#!/usr/bin/env python3
"""
Test script to generate actual recommendations with realistic data
This will show the full recommendation output including LLM analysis
"""

import requests
import json
import time
from datetime import datetime, timezone


# Test data designed to trigger recommendations
def create_test_request_with_recommendations():
    """Create test data that will definitely generate recommendations"""

    return {
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
                    "CostCenter": "engineering",
                },
                "properties": {
                    "instance_type": "t3.large",
                    "platform": "Linux/UNIX",
                    "state": "running",
                    "launch_time": "2024-01-15T10:30:00Z",
                    "vpc_id": "vpc-12345678",
                    "subnet_id": "subnet-87654321",
                },
            },
            {
                "resource_id": "vol-underutilized-storage",
                "service": "AWS.EBS",
                "region": "us-east-1",
                "availability_zone": "us-east-1a",
                "tags": {
                    "Name": "Web Server Storage",
                    "Environment": "production",
                    "AttachedTo": "i-overprovisioned-web-server",
                },
                "properties": {
                    "volume_type": "gp3",
                    "size": 500,
                    "iops": 3000,
                    "throughput": 125,
                    "state": "in-use",
                    "encrypted": True,
                },
            },
            {
                "resource_id": "bucket-lifecycle-candidate",
                "service": "AWS.S3",
                "region": "us-east-1",
                "tags": {
                    "Name": "Application Logs Bucket",
                    "Environment": "production",
                    "DataClassification": "logs",
                },
                "properties": {
                    "bucket_name": "company-app-logs-bucket",
                    "versioning": "Enabled",
                    "encryption": "AES256",
                    "lifecycle_policy": None,
                    "storage_class": "STANDARD",
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
                "cpu_utilization_min": 2.1,
                "cpu_utilization_max": 22.3,
                "memory_utilization_p50": 25.4,
                "memory_utilization_p90": 35.8,
                "memory_utilization_p95": 42.1,
                "iops_read": 45.2,
                "iops_write": 23.8,
                "throughput_read": 2.1,
                "throughput_write": 1.3,
                "network_in": 125.4,
                "network_out": 89.7,
                "metrics": {
                    "disk_utilization": 35.2,
                    "connection_count": 15,
                    "request_count_per_minute": 245,
                },
            },
            {
                "resource_id": "vol-underutilized-storage",
                "timestamp": "2024-08-31T23:59:59Z",
                "period_days": 30,
                "is_idle": False,
                "cpu_utilization_p50": None,
                "cpu_utilization_p90": None,
                "cpu_utilization_p95": None,
                "memory_utilization_p50": None,
                "memory_utilization_p90": None,
                "memory_utilization_p95": None,
                "iops_read": 25.3,
                "iops_write": 15.7,
                "throughput_read": 1.2,
                "throughput_write": 0.8,
                "metrics": {"volume_utilization": 35.2, "queue_depth": 1.2},
            },
            {
                "resource_id": "bucket-lifecycle-candidate",
                "timestamp": "2024-08-31T23:59:59Z",
                "period_days": 30,
                "is_idle": False,
                "metrics": {
                    "total_objects": 2500000,
                    "total_size_gb": 1250.5,
                    "get_requests_per_day": 150,
                    "put_requests_per_day": 8500,
                    "objects_older_than_90_days": 1800000,
                    "objects_older_than_365_days": 900000,
                    "average_object_access_frequency": 0.02,
                },
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
                "service": "AWS.EBS",
                "resource_id": "vol-underutilized-storage",
                "region": "us-east-1",
                "usage_type": "EBS:VolumeIOPS.gp3",
                "usage_amount": 3000.0,
                "usage_unit": "IOPS-Month",
                "unblended_cost": 18.0,
                "amortized_cost": 18.0,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
            {
                "service": "AWS.S3",
                "resource_id": "bucket-lifecycle-candidate",
                "region": "us-east-1",
                "usage_type": "S3-Storage-Class",
                "usage_amount": 1250.5,
                "usage_unit": "GB-Month",
                "unblended_cost": 28.76,
                "amortized_cost": 28.76,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
            {
                "service": "AWS.S3",
                "resource_id": "bucket-lifecycle-candidate",
                "region": "us-east-1",
                "usage_type": "S3-Requests-PUT",
                "usage_amount": 263500.0,
                "usage_unit": "Requests",
                "unblended_cost": 1.32,
                "amortized_cost": 1.32,
                "bill_period_start": "2024-08-01T00:00:00Z",
                "bill_period_end": "2024-08-31T23:59:59Z",
            },
        ],
        "analysis_mode": "individual",
    }


def format_recommendation_output(recommendations, title=""):
    """Format recommendations for nice display"""
    if not recommendations:
        return f"\n🔍 {title}\n{'='*60}\n❌ No recommendations generated\n"

    output = f"\n🎯 {title}\n{'='*60}\n"
    output += f"📊 Total Recommendations: {len(recommendations)}\n\n"

    total_monthly_savings = 0
    for i, rec in enumerate(recommendations, 1):
        total_monthly_savings += rec.get("estimated_monthly_savings", 0)

        output += f"💡 Recommendation #{i}: {rec.get('recommendation_type', 'Unknown').title()}\n"
        output += f"🎯 Resource: {rec.get('resource_id', 'N/A')} ({rec.get('service', 'N/A')})\n"
        output += (
            f"💰 Monthly Savings: ${rec.get('estimated_monthly_savings', 0):.2f}\n"
        )
        output += f"📈 Confidence: {rec.get('confidence_score', 0):.1%}\n"
        output += f"⚠️  Risk Level: {rec.get('risk_level', 'unknown').title()}\n"

        # Current vs Recommended config
        current = rec.get("current_config", {})
        recommended = rec.get("recommended_config", {})

        if current or recommended:
            output += f"\n🔧 Configuration Changes:\n"
            if current:
                output += f"   Current: {current}\n"
            if recommended:
                output += f"   Recommended: {recommended}\n"

        # Implementation details
        impact = rec.get("impact_description", "")
        if impact:
            output += f"\n📋 Impact: {impact}\n"

        rationale = rec.get("rationale", "")
        if rationale:
            output += f"🧠 Rationale: {rationale}\n"

        # Implementation steps
        steps = rec.get("implementation_steps", [])
        if steps:
            output += f"\n✅ Implementation Steps:\n"
            for step in steps[:3]:  # Show first 3 steps
                output += f"   • {step}\n"

        # Evidence
        evidence = rec.get("evidence", {})
        if evidence:
            output += f"\n📊 Supporting Evidence:\n"
            for key, value in evidence.items():
                if isinstance(value, str) and len(value) < 100:
                    output += f"   • {key.replace('_', ' ').title()}: {value}\n"

        # Warnings
        warnings = rec.get("evidence_warnings", [])
        if warnings:
            output += f"\n⚠️  Warnings: {', '.join(warnings)}\n"

        output += f"\n{'-'*60}\n"

    output += f"\n💰 TOTAL MONTHLY SAVINGS POTENTIAL: ${total_monthly_savings:.2f}\n"
    output += f"💰 TOTAL ANNUAL SAVINGS POTENTIAL: ${total_monthly_savings * 12:.2f}\n"

    return output


def test_api_with_recommendations():
    """Test the API and display detailed recommendation output"""

    print("🚀 LLM Cost Recommendation API - Recommendation Generation Test")
    print("=" * 70)

    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is available and ready")
        else:
            print("❌ Server returned error status:", health_response.status_code)
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not available: {e}")
        print("💡 Please start the server first with:")
        print("   python -m llm_cost_recommendation serve --host 0.0.0.0 --port 8000")
        return

    # Create request with recommendation-friendly data
    request_data = create_test_request_with_recommendations()

    print(f"\n📊 Test Data Summary:")
    print(f"   Resources: {len(request_data['resources'])}")
    print(f"   Metrics: {len(request_data['metrics'])}")
    print(f"   Billing Records: {len(request_data['billing'])}")
    print(f"   Analysis Mode: {request_data['analysis_mode']}")

    print(f"\n🎯 Resources to Analyze:")
    for resource in request_data["resources"]:
        service = resource["service"]
        resource_id = resource["resource_id"]

        # Get associated metrics and billing
        has_metrics = any(
            m["resource_id"] == resource_id for m in request_data["metrics"]
        )
        has_billing = any(
            b["resource_id"] == resource_id for b in request_data["billing"]
        )
        total_cost = sum(
            b["unblended_cost"]
            for b in request_data["billing"]
            if b["resource_id"] == resource_id
        )

        print(f"   • {service}: {resource_id}")
        print(
            f"     Cost: ${total_cost:.2f}/month | Metrics: {'✅' if has_metrics else '❌'} | Billing: {'✅' if has_billing else '❌'}"
        )

    print(f"\n🚀 Sending analysis request...")
    start_time = time.time()

    try:
        # Send request to API
        response = requests.post(
            "http://localhost:8000/analyze",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120,  # Allow 2 minutes for analysis
        )

        response_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            print(f"✅ Analysis completed successfully!")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            print(f"📦 Response size: {len(response.text):,} bytes")

            # Extract and display recommendations
            recommendations = result.get("recommendations", [])
            summary = result.get("summary", {})
            coverage = result.get("coverage", {})

            # Display summary
            print(f"\n📋 Analysis Summary:")
            print(
                f"   Total Recommendations: {summary.get('total_recommendations', 0)}"
            )
            print(
                f"   High Impact Recommendations: {summary.get('high_impact_recommendations', 0)}"
            )
            print(
                f"   Total Monthly Savings: ${summary.get('total_monthly_savings', 0):.2f}"
            )
            print(
                f"   Total Annual Savings: ${summary.get('total_annual_savings', 0):.2f}"
            )
            print(f"   Analysis Time: {summary.get('analysis_time_seconds', 0):.2f}s")

            # Display coverage
            print(f"\n🎯 Analysis Coverage:")
            print(f"   Resources Analyzed: {coverage.get('resources_analyzed', 0)}")
            print(
                f"   Services with Specific Agents: {coverage.get('services_with_specific_agents', 0)}"
            )
            print(
                f"   Resources with Recommendations: {coverage.get('resources_with_recommendations', 0)}"
            )

            # Display detailed recommendations
            print(
                format_recommendation_output(
                    recommendations, "DETAILED RECOMMENDATIONS"
                )
            )

            # Show any resources without recommendations
            resources_without_recs = result.get("resources_without_recommendations", [])
            if resources_without_recs:
                print(f"\n⚠️  Resources Without Recommendations:")
                for resource in resources_without_recs:
                    print(
                        f"   • {resource.get('resource_id', 'N/A')} ({resource.get('service', 'N/A')})"
                    )
                    print(f"     Reason: {resource.get('reason', 'Unknown')}")

        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.Timeout:
        print("⏰ Request timed out - analysis took too long")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")
        print(f"Raw response: {response.text}")


if __name__ == "__main__":
    test_api_with_recommendations()
