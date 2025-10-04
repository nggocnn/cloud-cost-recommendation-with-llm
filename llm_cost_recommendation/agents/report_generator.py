"""
Report Generator - Handles comprehensive report generation.
Extracted from CoordinatorAgent to focus on metrics calculation and report formatting.
"""

from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime, timezone

from ..models import Resource, Recommendation, RecommendationReport, RiskLevel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Handles comprehensive recommendation report generation."""
    
    def generate_report(
        self,
        recommendations: List[Recommendation],
        resources: List[Resource],
        start_time: datetime,
        resources_without_recommendations: List[Dict[str, Any]] = None,
    ) -> RecommendationReport:
        """Generate comprehensive recommendation report."""
        logger.info(
            "Generating recommendation report",
            recommendations_count=len(recommendations),
            resources_count=len(resources)
        )
        
        # Calculate summary metrics
        metrics = self._calculate_summary_metrics(recommendations)
        
        # Calculate risk distribution
        risk_distribution = self._calculate_risk_distribution(recommendations)
        
        # Calculate savings breakdown by service
        savings_by_service = self._calculate_savings_by_service(recommendations)
        
        # Categorize recommendations by implementation timeline
        timeline_categories = self._categorize_by_timeline(recommendations)
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(resources, recommendations, resources_without_recommendations)
        
        # Generate analysis metadata
        analysis_metadata = self._generate_analysis_metadata(start_time, resources, resources_without_recommendations)
        
        # Extract data quality issues
        data_quality_issues = self._extract_data_quality_issues(recommendations)
        
        # Create comprehensive report
        report = RecommendationReport(
            id=f"report_{int(datetime.now().timestamp())}",
            generated_at=datetime.now(),
            total_monthly_savings=metrics["total_monthly_savings"],
            total_annual_savings=metrics["total_annual_savings"],
            total_recommendations=len(recommendations),
            recommendations=recommendations,
            low_risk_count=risk_distribution["low"],
            medium_risk_count=risk_distribution["medium"], 
            high_risk_count=risk_distribution["high"],
            savings_by_service=savings_by_service,
            quick_wins=timeline_categories["quick_wins"],
            medium_term=timeline_categories["medium_term"],
            long_term=timeline_categories["long_term"],
            coverage=coverage_metrics,
            analysis_metadata=analysis_metadata,
            data_quality_issues=data_quality_issues
        )
        
        logger.info(
            "Report generation completed",
            report_id=report.id,
            total_savings=report.total_monthly_savings,
            recommendations=report.total_recommendations
        )
        
        return report
    
    def _calculate_summary_metrics(self, recommendations: List[Recommendation]) -> Dict[str, float]:
        """Calculate summary financial metrics."""
        if not recommendations:
            return {"total_monthly_savings": 0.0, "total_annual_savings": 0.0}
        
        total_monthly = sum(rec.estimated_monthly_savings for rec in recommendations)
        total_annual = sum(rec.annual_savings for rec in recommendations)
        
        logger.debug(
            "Summary metrics calculated",
            monthly_savings=total_monthly,
            annual_savings=total_annual
        )
        
        return {
            "total_monthly_savings": total_monthly,
            "total_annual_savings": total_annual,
        }
    
    def _calculate_risk_distribution(self, recommendations: List[Recommendation]) -> Dict[str, int]:
        """Calculate distribution of recommendations by risk level."""
        risk_counts = defaultdict(int)
        
        for rec in recommendations:
            risk_level = rec.risk_level
            if hasattr(risk_level, 'value'):
                risk_key = risk_level.value.lower()
            else:
                risk_key = str(risk_level).lower()
            
            risk_counts[risk_key] += 1
        
        # Ensure all risk levels are represented
        distribution = {
            "low": risk_counts.get("low", 0),
            "medium": risk_counts.get("medium", 0),
            "high": risk_counts.get("high", 0)
        }
        
        logger.debug("Risk distribution calculated", **distribution)
        
        return distribution
    
    def _calculate_savings_by_service(self, recommendations: List[Recommendation]) -> Dict[str, float]:
        """Calculate savings breakdown by service type."""
        savings_by_service = defaultdict(float)
        
        for rec in recommendations:
            service_name = rec.service if isinstance(rec.service, str) else rec.service.value
            savings_by_service[service_name] += rec.estimated_monthly_savings
        
        result = dict(savings_by_service)
        
        logger.debug(
            "Savings by service calculated",
            services=len(result),
            breakdown=result
        )
        
        return result
    
    def _categorize_by_timeline(self, recommendations: List[Recommendation]) -> Dict[str, List[str]]:
        """Categorize recommendations by implementation timeline."""
        timeline_categories = {
            "quick_wins": [],
            "medium_term": [],
            "long_term": []
        }
        
        for rec in recommendations:
            category = self._determine_timeline_category(rec)
            timeline_categories[category].append(rec.id)
        
        logger.debug(
            "Timeline categorization completed",
            quick_wins=len(timeline_categories["quick_wins"]),
            medium_term=len(timeline_categories["medium_term"]),
            long_term=len(timeline_categories["long_term"])
        )
        
        return timeline_categories
    
    def _determine_timeline_category(self, rec: Recommendation) -> str:
        """Determine timeline category for a recommendation."""
        # Quick wins: Low risk AND simple implementation (≤2 steps)
        if (rec.risk_level == RiskLevel.LOW and 
            len(rec.implementation_steps or []) <= 2):
            return "quick_wins"
        
        # Long term: High risk OR complex implementation (>5 steps)
        elif (rec.risk_level == RiskLevel.HIGH or 
              len(rec.implementation_steps or []) > 5):
            return "long_term"
        
        # Medium term: Everything else
        else:
            return "medium_term"
    
    def _calculate_coverage_metrics(
        self, 
        resources: List[Resource], 
        recommendations: List[Recommendation],
        resources_without_recommendations: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate analysis coverage metrics."""
        if not resources:
            return {}
        
        # Basic coverage stats
        total_resources = len(resources)
        resources_with_recommendations = len(set(rec.resource_id for rec in recommendations))
        resources_without_recs = len(resources_without_recommendations or [])
        
        # Service-level coverage
        analyzed_services = set(resource.service for resource in resources)
        services_with_recommendations = set(rec.service for rec in recommendations)
        
        coverage_metrics = {
            "total_resources_analyzed": total_resources,
            "resources_with_recommendations": resources_with_recommendations,
            "resources_without_recommendations": resources_without_recs,
            "coverage_percentage": (resources_with_recommendations / total_resources * 100) if total_resources > 0 else 0,
            "services_analyzed": len(analyzed_services),
            "services_with_recommendations": len(services_with_recommendations),
            "analyzed_services_list": [s.value if hasattr(s, 'value') else str(s) for s in analyzed_services],
        }
        
        logger.debug("Coverage metrics calculated", **coverage_metrics)
        
        return coverage_metrics
    
    def _generate_analysis_metadata(
        self, 
        start_time: datetime, 
        resources: List[Resource],
        resources_without_recommendations: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate metadata about the analysis process."""
        end_time = datetime.now(timezone.utc)
        analysis_duration = (end_time - start_time).total_seconds()
        
        metadata = {
            "analysis_start_time": start_time.isoformat(),
            "analysis_end_time": end_time.isoformat(),
            "analysis_time_seconds": analysis_duration,
            "resources_analyzed": len(resources),
            "resources_without_recommendations": len(resources_without_recommendations or []),
            "analysis_efficiency": {
                "resources_per_second": len(resources) / analysis_duration if analysis_duration > 0 else 0,
                "analysis_duration_formatted": self._format_duration(analysis_duration)
            }
        }
        
        logger.debug(
            "Analysis metadata generated",
            duration=analysis_duration,
            efficiency=metadata["analysis_efficiency"]["resources_per_second"]
        )
        
        return metadata
    
    def _extract_data_quality_issues(self, recommendations: List[Recommendation]) -> List[str]:
        """Extract data quality issues from recommendations."""
        data_quality_issues = []
        seen_issues = set()
        
        for rec in recommendations:
            if rec.warnings:
                for warning in rec.warnings:
                    # Look for data quality related warnings
                    if any(keyword in warning for keyword in [
                        "DEGRADED",
                        "CONVERSION ERROR", 
                        "LOW CONFIDENCE",
                        "LOW IMPACT",
                        "Evidence Quality",
                        "DATA QUALITY ISSUE"
                    ]):
                        issue_description = f"Resource {rec.resource_id}: {warning}"
                        if issue_description not in seen_issues:
                            data_quality_issues.append(issue_description)
                            seen_issues.add(issue_description)
        
        logger.debug(
            "Data quality issues extracted",
            total_issues=len(data_quality_issues)
        )
        
        return data_quality_issues
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def get_report_statistics(self, report: RecommendationReport) -> Dict[str, Any]:
        """Get statistics about the generated report."""
        return {
            "report_id": report.id,
            "generation_timestamp": report.generated_at,
            "financial_impact": {
                "monthly_savings": report.total_monthly_savings,
                "annual_savings": report.total_annual_savings,
                "total_recommendations": report.total_recommendations
            },
            "risk_profile": {
                "low_risk": report.low_risk_count,
                "medium_risk": report.medium_risk_count,
                "high_risk": report.high_risk_count
            },
            "implementation_timeline": {
                "quick_wins": len(report.quick_wins),
                "medium_term": len(report.medium_term),
                "long_term": len(report.long_term)
            },
            "data_quality": {
                "issues_found": len(report.data_quality_issues),
                "coverage_percentage": report.coverage.get("coverage_percentage", 0)
            }
        }