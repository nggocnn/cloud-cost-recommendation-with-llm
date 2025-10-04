#!/usr/bin/env python3
"""
Comprehensive API route testing script.
Tests all available API endpoints with proper error handling and reporting.
"""

import json
import requests
import time
import sys
from typing import Dict, Any
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"


class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = {}

    def test_route(
        self,
        name: str,
        method: str,
        endpoint: str,
        data: Dict[Any, Any] = None,
        headers: Dict[str, str] = None,
        expected_status: int = 200,
    ) -> Dict[str, Any]:
        """Test a single API route"""
        print(f"\n🧪 Testing {name}")
        print(f"   {method} {endpoint}")

        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Parse response
            try:
                response_json = response.json()
            except:
                response_json = {"raw_response": response.text}

            result = {
                "success": response.status_code == expected_status,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": response.elapsed.total_seconds(),
                "response": response_json,
                "headers": dict(response.headers),
            }

            # Log result
            status_emoji = "✅" if result["success"] else "❌"
            print(
                f"   {status_emoji} Status: {response.status_code} (expected {expected_status})"
            )
            print(f"   ⏱️  Response time: {result['response_time']:.3f}s")

            if not result["success"]:
                print(f"   📄 Response: {json.dumps(response_json, indent=2)[:200]}...")

            return result

        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "status_code": None,
                "response_time": None,
            }
            print(f"   ❌ Error: {e}")
            return result

    def test_health_endpoints(self):
        """Test health and status endpoints"""
        print("\n" + "=" * 60)
        print("🏥 HEALTH & STATUS ENDPOINTS")
        print("=" * 60)

        # Health check
        self.results["health"] = self.test_route("Health Check", "GET", "/health")

        # Health live probe
        self.results["health_live"] = self.test_route(
            "Health Live Probe", "GET", "/health/live"
        )

        # Health ready probe
        self.results["health_ready"] = self.test_route(
            "Health Ready Probe", "GET", "/health/ready"
        )

        # System status
        self.results["status"] = self.test_route("System Status", "GET", "/status")

    def test_documentation_endpoints(self):
        """Test API documentation endpoints"""
        print("\n" + "=" * 60)
        print("📚 DOCUMENTATION ENDPOINTS")
        print("=" * 60)

        # OpenAPI schema
        self.results["openapi"] = self.test_route(
            "OpenAPI Schema", "GET", "/openapi.json"
        )

        # Swagger UI
        self.results["docs"] = self.test_route(
            "Swagger UI Documentation", "GET", "/docs"
        )

        # ReDoc
        self.results["redoc"] = self.test_route("ReDoc Documentation", "GET", "/redoc")

    def test_system_endpoints(self):
        """Test system monitoring endpoints"""
        print("\n" + "=" * 60)
        print("🖥️  SYSTEM MONITORING ENDPOINTS")
        print("=" * 60)

        # System metrics
        self.results["system_metrics"] = self.test_route(
            "System Metrics", "GET", "/metrics/system"
        )

    def test_analysis_endpoints(self):
        """Test analysis endpoints"""
        print("\n" + "=" * 60)
        print("🔍 ANALYSIS ENDPOINTS")
        print("=" * 60)

        # Test with minimal valid request
        sample_request = {
            "resources": [
                {
                    "resource_id": "test-ec2-001",
                    "provider": "AWS",
                    "service": "AWS.EC2",
                    "region": "us-east-1",
                    "tags": {"Environment": "test"},
                    "configuration": {"InstanceType": "t3.large", "State": "running"},
                }
            ],
            "billing": [],
            "metrics": [],
            "individual_processing": False,
            "max_recommendations": 5,
        }

        # Test analysis endpoint
        self.results["analyze"] = self.test_route(
            "Cost Analysis", "POST", "/analyze", data=sample_request
        )

        # Test analysis with invalid data (should return 422)
        invalid_request = {
            "resources": [],  # Empty resources should fail validation
            "billing": [],
        }

        self.results["analyze_invalid"] = self.test_route(
            "Cost Analysis (Invalid Data)",
            "POST",
            "/analyze",
            data=invalid_request,
            expected_status=422,
        )

    def test_recommendation_endpoints(self):
        """Test recommendation detail endpoints"""
        print("\n" + "=" * 60)
        print("💡 RECOMMENDATION ENDPOINTS")
        print("=" * 60)

        # Test recommendation detail (should return 501 - not implemented)
        self.results["recommendation_detail"] = self.test_route(
            "Recommendation Detail",
            "GET",
            "/recommendations/test-rec-001",
            expected_status=501,
        )

    def test_error_endpoints(self):
        """Test error handling"""
        print("\n" + "=" * 60)
        print("🚨 ERROR HANDLING")
        print("=" * 60)

        # Test 404 - non-existent endpoint
        self.results["not_found"] = self.test_route(
            "Non-existent Endpoint", "GET", "/non-existent-route", expected_status=404
        )

        # Test method not allowed
        self.results["method_not_allowed"] = self.test_route(
            "Method Not Allowed", "POST", "/health", expected_status=405
        )

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Comprehensive API Route Testing")
        print(f"🎯 Target: {self.base_url}")

        start_time = time.time()

        # Test server connectivity first
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            print(f"✅ Server is reachable (status: {response.status_code})")
        except Exception as e:
            print(f"❌ Server is not reachable: {e}")
            return False

        # Run all test suites
        self.test_health_endpoints()
        self.test_documentation_endpoints()
        self.test_system_endpoints()
        self.test_analysis_endpoints()
        self.test_recommendation_endpoints()
        self.test_error_endpoints()

        total_time = time.time() - start_time

        # Generate summary report
        self.generate_summary_report(total_time)

        return True

    def generate_summary_report(self, total_time: float):
        """Generate a summary report of all tests"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY REPORT")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tests = total_tests - passed_tests

        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        print(f"\n📋 Detailed Results:")
        for test_name, result in self.results.items():
            status_emoji = "✅" if result.get("success", False) else "❌"
            status_code = result.get("status_code", "N/A")
            response_time = result.get("response_time")
            time_str = f"{response_time:.3f}s" if response_time else "N/A"

            print(f"   {status_emoji} {test_name}: {status_code} ({time_str})")

            if not result.get("success", False) and "error" in result:
                print(f"      Error: {result['error']}")

        # Performance summary
        response_times = [
            r.get("response_time", 0)
            for r in self.results.values()
            if r.get("response_time") is not None
        ]

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)

            print(f"\n⚡ Performance Summary:")
            print(f"   Average Response Time: {avg_time:.3f}s")
            print(f"   Fastest Response: {min_time:.3f}s")
            print(f"   Slowest Response: {max_time:.3f}s")

        # Save detailed results to file
        results_file = Path("api_test_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")


def main():
    """Main function to run API tests"""
    tester = APITester()
    success = tester.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
