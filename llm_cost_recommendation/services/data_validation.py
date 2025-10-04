"""
Enhanced data validation service to ensure data quality before LLM analysis.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta


from ..models import Resource, Metrics, BillingData
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataQualityValidator:
    """Validates data quality before sending to LLM for analysis"""

    def __init__(self):
        self.validation_results = {}

    def validate_resource_data(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive validation of resource data for LLM analysis"""

        validation_result = {
            "resource_id": resource.resource_id,
            "overall_quality": "unknown",
            "quality_score": 0.0,
            "data_availability": {
                "inventory": False,
                "billing": False,
                "metrics": False,
                "historical": False,
            },
            "data_quality_issues": [],
            "missing_critical_data": [],
            "recommendations_possible": False,
            "confidence_ceiling": 0.0,
            "required_data_collection": [],
        }

        # Validate inventory data
        inventory_score = self._validate_inventory_data(resource, validation_result)

        # Validate billing data
        billing_score = self._validate_billing_data(billing_data, validation_result)

        # Validate metrics data
        metrics_score = self._validate_metrics_data(metrics, validation_result)

        # Calculate overall quality score
        total_score = inventory_score + billing_score + metrics_score
        max_score = 3.0
        validation_result["quality_score"] = total_score / max_score

        # Determine overall quality level
        if validation_result["quality_score"] >= 0.8:
            validation_result["overall_quality"] = "excellent"
            validation_result["confidence_ceiling"] = 0.95
        elif validation_result["quality_score"] >= 0.6:
            validation_result["overall_quality"] = "good"
            validation_result["confidence_ceiling"] = 0.85
        elif validation_result["quality_score"] >= 0.4:
            validation_result["overall_quality"] = "fair"
            validation_result["confidence_ceiling"] = 0.70
        else:
            validation_result["overall_quality"] = "poor"
            validation_result["confidence_ceiling"] = 0.50

        # Determine if recommendations are possible
        validation_result["recommendations_possible"] = validation_result[
            "data_availability"
        ]["inventory"] and (
            validation_result["data_availability"]["billing"]
            or validation_result["data_availability"]["metrics"]
        )

        self.validation_results[resource.resource_id] = validation_result

        logger.info(
            "Data quality validation completed",
            resource_id=resource.resource_id,
            quality_score=validation_result["quality_score"],
            overall_quality=validation_result["overall_quality"],
            recommendations_possible=validation_result["recommendations_possible"],
        )

        return validation_result

    def _validate_inventory_data(
        self, resource: Resource, validation_result: Dict
    ) -> float:
        """Validate inventory/configuration data quality"""
        score = 0.0

        if resource and resource.resource_id:
            validation_result["data_availability"]["inventory"] = True
            score += 0.5

            # Check for essential properties
            if resource.properties and len(resource.properties) > 0:
                score += 0.3
            else:
                validation_result["data_quality_issues"].append(
                    "Missing resource properties/configuration details"
                )

            # Check for tags
            if resource.tags and len(resource.tags) > 0:
                score += 0.2
            else:
                validation_result["data_quality_issues"].append(
                    "Missing resource tags for context"
                )
        else:
            validation_result["data_availability"]["inventory"] = False
            validation_result["missing_critical_data"].append("Resource inventory data")
            validation_result["required_data_collection"].append(
                "Collect basic resource configuration and properties"
            )

        return min(score, 1.0)

    def _validate_billing_data(
        self, billing_data: Optional[List[BillingData]], validation_result: Dict
    ) -> float:
        """Validate billing/cost data quality"""
        score = 0.0

        if billing_data and len(billing_data) > 0:
            validation_result["data_availability"]["billing"] = True
            score += 0.5

            # Check for actual costs vs estimates
            has_real_costs = any(bd.unblended_cost > 0 for bd in billing_data)
            if has_real_costs:
                score += 0.3
            else:
                validation_result["data_quality_issues"].append(
                    "Billing costs appear to be estimated, not actual AWS costs"
                )

            # Check for usage data
            has_usage_data = any(bd.usage_amount > 0 for bd in billing_data)
            if has_usage_data:
                score += 0.2
            else:
                validation_result["data_quality_issues"].append(
                    "Missing usage amount data in billing records"
                )
        else:
            validation_result["data_availability"]["billing"] = False
            validation_result["missing_critical_data"].append("Billing/cost data")
            validation_result["required_data_collection"].extend(
                [
                    "Enable AWS Cost and Usage Reports",
                    "Collect actual billing data from AWS",
                ]
            )

        return min(score, 1.0)

    def _validate_metrics_data(
        self, metrics: Optional[Metrics], validation_result: Dict
    ) -> float:
        """Validate performance metrics data quality"""
        score = 0.0

        if metrics and metrics.resource_id:
            validation_result["data_availability"]["metrics"] = True
            score += 0.3

            # Check for CPU metrics
            if any(
                getattr(metrics, attr, None) is not None
                for attr in [
                    "cpu_utilization_p50",
                    "cpu_utilization_p90",
                    "cpu_utilization_p95",
                ]
            ):
                score += 0.3
            else:
                validation_result["data_quality_issues"].append(
                    "Missing CPU utilization metrics"
                )
                validation_result["required_data_collection"].append(
                    "Collect CloudWatch CPU utilization metrics"
                )

            # Check for memory metrics
            if any(
                getattr(metrics, attr, None) is not None
                for attr in [
                    "memory_utilization_p50",
                    "memory_utilization_p90",
                    "memory_utilization_p95",
                ]
            ):
                score += 0.2
            else:
                validation_result["data_quality_issues"].append(
                    "Missing memory utilization metrics"
                )
                validation_result["required_data_collection"].append(
                    "Collect memory utilization metrics"
                )

            # Check for network metrics
            if any(
                getattr(metrics, attr, None) is not None
                for attr in ["network_in", "network_out"]
            ):
                score += 0.2
            else:
                validation_result["data_quality_issues"].append(
                    "Missing network utilization metrics"
                )
                validation_result["required_data_collection"].append(
                    "Collect network traffic metrics"
                )
        else:
            validation_result["data_availability"]["metrics"] = False
            validation_result["missing_critical_data"].append("Performance metrics")
            validation_result["required_data_collection"].extend(
                [
                    "Enable CloudWatch detailed monitoring",
                    "Collect performance metrics for utilization analysis",
                ]
            )

        return min(score, 1.0)

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality across all validated resources"""
        if not self.validation_results:
            return {
                "total_resources": 0,
                "quality_distribution": {},
                "common_issues": [],
                "recommendations": [],
            }

        total = len(self.validation_results)
        quality_counts = {}
        all_issues = []
        all_recommendations = []

        for result in self.validation_results.values():
            quality = result["overall_quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            all_issues.extend(result["data_quality_issues"])
            all_recommendations.extend(result["required_data_collection"])

        # Find most common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        # Find most common recommendations
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        common_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_resources": total,
            "quality_distribution": {
                quality: f"{count}/{total} ({count/total*100:.1f}%)"
                for quality, count in quality_counts.items()
            },
            "common_issues": [
                f"{issue} ({count} resources)" for issue, count in common_issues
            ],
            "recommendations": [
                f"{rec} ({count} resources)" for rec, count in common_recs
            ],
            "average_quality_score": sum(
                r["quality_score"] for r in self.validation_results.values()
            )
            / total,
        }

    def create_data_quality_report(self, output_file: str = "data_quality_report.json"):
        """Create detailed data quality report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_data_quality_summary(),
            "resource_details": self.validation_results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Data quality report created", output_file=output_file)
        return report


class EvidenceValidator:
    """Validates that LLM recommendations are properly supported by evidence"""

    def __init__(self):
        pass

    def validate_recommendation_evidence(
        self, recommendation: Dict[str, Any], available_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that a recommendation is properly supported by available data"""

        validation = {
            "recommendation_id": recommendation.get("resource_id", "unknown"),
            "evidence_quality": "unknown",
            "evidence_issues": [],
            "unsupported_claims": [],
            "data_gaps": [],
            "evidence_score": 0.0,
        }

        evidence = recommendation.get("evidence", {})

        # Check metrics claims
        metrics_claim = evidence.get("metrics_analysis", "")
        self._validate_metrics_claims(metrics_claim, available_data, validation)

        # Check cost claims
        cost_claim = evidence.get("cost_breakdown", "")
        self._validate_cost_claims(cost_claim, available_data, validation)

        # Check performance impact claims
        performance_claim = evidence.get("performance_impact", "")
        self._validate_performance_claims(performance_claim, available_data, validation)

        # Check confidence score alignment with evidence quality
        confidence_score = recommendation.get("confidence_score", 0.0)
        self._validate_confidence_alignment(
            confidence_score, evidence, available_data, validation
        )

        # Calculate evidence score
        validation["evidence_score"] = self._calculate_evidence_score(
            validation, available_data
        )

        # Determine overall evidence quality
        if validation["evidence_score"] >= 0.8:
            validation["evidence_quality"] = "excellent"
        elif validation["evidence_score"] >= 0.6:
            validation["evidence_quality"] = "good"
        elif validation["evidence_score"] >= 0.4:
            validation["evidence_quality"] = "fair"
        else:
            validation["evidence_quality"] = "poor"

        return validation

    def _validate_metrics_claims(
        self, metrics_claim: str, available_data: Dict, validation: Dict
    ):
        """Validate metrics-related claims in evidence"""
        if not metrics_claim:
            return

        # Detect fabricated percentage claims
        percentage_indicators = [
            "% of requests",
            "% of traffic",
            "% utilization",
            "traffic shows",
            "analysis shows",
            "observed traffic",
            "requests are served",
            "traffic analysis over",
            "performance data indicates",
        ]

        has_percentage_claim = any(
            indicator in metrics_claim.lower() for indicator in percentage_indicators
        )
        has_specific_percentage = any(
            char.isdigit() and "%" in metrics_claim[i : i + 5]
            for i, char in enumerate(metrics_claim)
            if char.isdigit()
        )

        if has_percentage_claim or has_specific_percentage:
            if not available_data.get("metrics") or not available_data.get(
                "metrics", {}
            ).get("datapoints"):
                validation["unsupported_claims"].append(
                    f"Traffic/Performance claim: '{metrics_claim[:100]}...' - but no performance data available"
                )
                validation["evidence_issues"].append(
                    "Fabricated traffic percentage without actual metrics"
                )

    def _validate_cost_claims(
        self, cost_claim: str, available_data: Dict, validation: Dict
    ):
        """Validate cost-related claims in evidence"""
        if not cost_claim:
            return

        # Check for specific cost numbers
        if "$" in cost_claim and (
            "current" in cost_claim.lower() or "monthly" in cost_claim.lower()
        ):
            if not available_data.get("billing_data"):
                if (
                    "estimated" not in cost_claim.lower()
                    and "approximate" not in cost_claim.lower()
                ):
                    validation["evidence_issues"].append(
                        "Cost claim doesn't indicate if data is estimated vs actual"
                    )

        # Check for contradiction: saying "no cost data" but providing specific amounts
        no_cost_data_indicators = [
            "no actual cost data",
            "no cost data available",
            "cost data not available",
            "no billing data",
            "billing data not available",
        ]

        if any(
            indicator in cost_claim.lower() for indicator in no_cost_data_indicators
        ):
            validation["unsupported_claims"].append(
                f"Cost contradiction: Evidence claims '{cost_claim}' but specific dollar amounts provided in recommendation"
            )
            validation["evidence_issues"].append(
                "Fabricated cost estimates without actual billing data"
            )

    def _validate_performance_claims(
        self, performance_claim: str, available_data: Dict, validation: Dict
    ):
        """Validate performance impact claims"""
        if not performance_claim:
            return

        performance_indicators = [
            "performance",
            "latency",
            "response time",
            "throughput",
            "performance remains",
            "expected performance",
            "impact",
        ]

        has_performance_claim = any(
            indicator in performance_claim.lower()
            for indicator in performance_indicators
        )

        if has_performance_claim:
            if not available_data.get("metrics"):
                validation["unsupported_claims"].append(
                    f"Performance claim: '{performance_claim}' - but no performance data available"
                )

    def _validate_confidence_alignment(
        self,
        confidence_score: float,
        evidence: Dict,
        available_data: Dict,
        validation: Dict,
    ):
        """Validate that confidence score aligns with available evidence"""

        # Check for high confidence with poor data
        data_availability_score = 0.0
        if available_data.get("metrics"):
            data_availability_score += 0.4
        if available_data.get("billing_data"):
            data_availability_score += 0.4
        if available_data.get("resource"):
            data_availability_score += 0.2

        # Check evidence statements
        evidence_statements = " ".join(
            [
                evidence.get("metrics_analysis", ""),
                evidence.get("cost_breakdown", ""),
                evidence.get("performance_impact", ""),
            ]
        ).lower()

        no_data_indicators = [
            "no performance metrics",
            "no actual cost data",
            "no cost data available",
            "metrics not available",
            "data not available",
            "insufficient data",
        ]

        has_data_limitations = any(
            indicator in evidence_statements for indicator in no_data_indicators
        )

        # Flag high confidence with poor data availability
        if confidence_score > 0.6 and (
            data_availability_score < 0.4 or has_data_limitations
        ):
            validation["evidence_issues"].append(
                f"High confidence score ({confidence_score}) despite limited data availability"
            )
            validation["unsupported_claims"].append(
                f"Confidence score {confidence_score} not justified by evidence quality"
            )

        # Add specific warnings for low confidence scores
        if confidence_score < 0.5:
            validation["evidence_warnings"] = validation.get("evidence_warnings", [])
            validation["evidence_warnings"].append(
                f"LOW CONFIDENCE ({confidence_score:.0%}): This recommendation may be unreliable. "
                "Additional performance monitoring and data collection recommended before implementation."
            )
        elif confidence_score < 0.7:
            validation["evidence_warnings"] = validation.get("evidence_warnings", [])
            validation["evidence_warnings"].append(
                f"MODERATE CONFIDENCE ({confidence_score:.0%}): Consider validating this recommendation "
                "with additional monitoring or testing in a non-production environment first."
            )

        # Add data quality context warnings
        if data_availability_score < 0.6:
            validation["evidence_warnings"] = validation.get("evidence_warnings", [])
            missing_data = []
            if not available_data.get("metrics"):
                missing_data.append("performance metrics")
            if not available_data.get("billing_data"):
                missing_data.append("billing data")

            if missing_data:
                validation["evidence_warnings"].append(
                    f"DATA LIMITATION: Missing {' and '.join(missing_data)}. "
                    "Recommendation based on limited information may not reflect actual optimization potential."
                )

    def _calculate_evidence_score(
        self, validation: Dict, available_data: Dict
    ) -> float:
        """Calculate evidence quality score"""
        score = 1.0

        # Deduct for each unsupported claim
        score -= len(validation["unsupported_claims"]) * 0.3

        # Deduct for evidence issues
        score -= len(validation["evidence_issues"]) * 0.2

        # Boost if data is actually available
        if available_data.get("metrics"):
            score += 0.1
        if available_data.get("billing_data"):
            score += 0.1

        return max(0.0, min(1.0, score))
