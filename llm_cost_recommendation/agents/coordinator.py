"""
Coordinator agent that orchestrates service agents and consolidates recommendations.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timezone

from ..models import (
    Resource,
    Metrics,
    BillingData,
    Recommendation,
    RecommendationReport,
    ServiceType,
    RiskLevel,
)
from ..services.llm import LLMService
from ..services.config import ConfigManager
from ..utils.logging import get_logger
from .base import ServiceAgent

logger = get_logger(__name__)


class CoordinatorAgent:
    """Coordinator agent that orchestrates service agents"""

    def __init__(self, config_manager: ConfigManager, llm_service: LLMService):
        self.config_manager = config_manager
        self.llm_service = llm_service
        self.config = config_manager.global_config

        # Initialize service agents
        self.service_agents: Dict[ServiceType, ServiceAgent] = {}
        self._initialize_agents()

        # Verify default agent is available (critical for system functionality)
        if ServiceType.DEFAULT not in self.service_agents:
            logger.critical("CRITICAL: Default agent not available - system will have coverage gaps")
            raise RuntimeError("Failed to initialize default agent - system cannot guarantee coverage")

        logger.info(
            "Coordinator agent initialized", 
            enabled_services=len(self.service_agents),
            agents=[agent.value for agent in self.service_agents.keys()]
        )

    def _initialize_agents(self):
        """Initialize service agents based on configuration"""
        # Always initialize default agent
        default_config = self.config_manager.get_service_config(ServiceType.DEFAULT)
        if default_config:
            try:
                default_agent = ServiceAgent(
                    default_config, self.llm_service, self.config
                )
                self.service_agents[ServiceType.DEFAULT] = default_agent
                logger.info(
                    "Default agent initialized",
                    agent_id=default_agent.agent_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize default agent",
                    error=str(e),
                )

        # Initialize service-specific agents
        for service in self.config.enabled_services:
            # Skip DEFAULT as it's already initialized
            if service == ServiceType.DEFAULT:
                continue

            service_config = self.config_manager.get_service_config(service)

            if service_config and service_config.enabled:
                try:
                    agent = ServiceAgent(service_config, self.llm_service, self.config)
                    self.service_agents[service] = agent

                    logger.info(
                        "Service agent initialized",
                        service=service.value,
                        agent_id=agent.agent_id,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to initialize service agent",
                        service=service.value,
                        error=str(e),
                    )

    async def analyze_resources_and_generate_report(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
        batch_mode: bool = True,
    ) -> RecommendationReport:
        """Analyze resources and generate comprehensive cost optimization report"""
        logger.info(
            "Starting resource analysis",
            total_resources=len(resources),
        )

        start_time = datetime.now(timezone.utc)

        # Group resources by service type for targeted analysis
        resources_by_service = self._group_resources_by_service(resources)

        # Analyze resources using appropriate service agents
        all_recommendations = []
        resources_without_recommendations = []

        if batch_mode:
            # Batch processing: Analyze services in parallel for efficiency
            # Each service agent will use intelligent batching internally
            analysis_tasks = []

            for service, service_resources in resources_by_service.items():
                if service in self.service_agents:
                    agent = self.service_agents[service]
                # Fall back to default agent
                elif ServiceType.DEFAULT in self.service_agents:
                    agent = self.service_agents[ServiceType.DEFAULT]
                    logger.info(
                        "Using default agent for unsupported service",
                        service=service.value,
                        resource_count=len(service_resources),
                    )
                else:
                    logger.warning(
                        "No agent available for service (no default agent configured)",
                        service=service.value,
                        resource_count=len(service_resources),
                    )
                    continue

                # Create parallel analysis task for this service
                task = self._analyze_service_resources_with_tracking(
                    agent, service_resources, metrics_data, billing_data
                )
                analysis_tasks.append(task)

            # Execute all service analyses in parallel
            if analysis_tasks:
                service_results = await asyncio.gather(
                    *analysis_tasks, return_exceptions=True
                )

                # Collect results from all services
                for result in service_results:
                    if isinstance(result, Exception):
                        logger.error("Service analysis failed", error=str(result))
                    else:
                        service_recommendations, service_no_recommendations = result
                        all_recommendations.extend(service_recommendations)
                        resources_without_recommendations.extend(service_no_recommendations)
        else:
            # Individual processing: Analyze each resource one by one for maximum precision
            # Useful for debugging or when dealing with problematic resources
            total_resources = len(resources)
            for i, resource in enumerate(resources, 1):
                if resource.service in self.service_agents:
                    agent = self.service_agents[resource.service]
                # Fall back to default agent
                elif ServiceType.DEFAULT in self.service_agents:
                    agent = self.service_agents[ServiceType.DEFAULT]
                    logger.info(
                        f"Using default agent for resource {i}/{total_resources}",
                        resource_id=resource.resource_id,
                        service=resource.service.value,
                    )
                else:
                    logger.warning(
                        f"No agent available for resource {i}/{total_resources}",
                        resource_id=resource.resource_id,
                        service=resource.service.value,
                    )
                    continue

                logger.info(
                    f"Analyzing resource {i}/{total_resources}",
                    resource_id=resource.resource_id,
                    service=resource.service.value,
                )

                # Get relevant metrics and billing data for this specific resource
                resource_metrics = (
                    metrics_data.get(resource.resource_id) if metrics_data else None
                )
                resource_billing = (
                    billing_data.get(resource.resource_id) if billing_data else None
                )

                try:
                    recommendations = await agent.analyze_single_resource(
                        resource, resource_metrics, resource_billing
                    )
                    if recommendations:
                        all_recommendations.extend(recommendations)
                    else:
                        # Track resources that didn't get recommendations
                        reason = await self._determine_no_recommendation_reason(
                            resource, resource_metrics, resource_billing, agent
                        )
                        resources_without_recommendations.append({
                            "resource_id": resource.resource_id,
                            "service": resource.service.value,
                            "reason": reason,
                            "agent_used": agent.agent_id
                        })
                except Exception as e:
                    logger.error(
                        "Failed to analyze individual resource",
                        resource_id=resource.resource_id,
                        service=resource.service.value,
                        error=str(e),
                    )

        # Post-process recommendations (deduplicate, filter, rank)
        processed_recommendations = self._post_process_recommendations(
            all_recommendations
        )

        # Generate comprehensive report with metrics and insights
        report = self._generate_report(processed_recommendations, resources, start_time, resources_without_recommendations)

        logger.info(
            "Resource analysis completed successfully",
            total_recommendations=len(processed_recommendations),
            total_monthly_savings=report.total_monthly_savings,
            total_annual_savings=report.total_annual_savings,
            analysis_time_seconds=(
                datetime.now(timezone.utc) - start_time
            ).total_seconds(),
        )

        # Log detailed coverage summary
        logger.info(
            "Analysis Summary",
            total_resources=len(resources),
            resources_with_specific_agents=report.coverage["resources_with_specific_agents"],
            resources_falling_back_to_default=report.coverage["resources_falling_back_to_default"],
            services_with_specific_agents=report.coverage["services_with_specific_agents"],
            services_falling_back_to_default=report.coverage["services_falling_back_to_default"],
            fallback_service_types=report.coverage["services_falling_back_to_default_list"] if report.coverage["services_falling_back_to_default_list"] else "None",
        )

        return report

    def _group_resources_by_service(
        self, resources: List[Resource]
    ) -> Dict[ServiceType, List[Resource]]:
        """Group resources by service type"""
        grouped = defaultdict(list)

        for resource in resources:
            grouped[resource.service].append(resource)

        return dict(grouped)

    async def _analyze_service_resources(
        self,
        agent: ServiceAgent,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze resources for a specific service"""
        try:
            recommendations = await agent.analyze_multiple_resources(
                resources, metrics_data, billing_data
            )

            # Apply service-specific limits
            max_recommendations = self.config.max_recommendations_per_service
            if len(recommendations) > max_recommendations:
                # Sort by savings and take top recommendations
                recommendations.sort(
                    key=lambda r: r.estimated_monthly_savings, reverse=True
                )
                recommendations = recommendations[:max_recommendations]

                logger.info(
                    "Applied recommendation limit",
                    service=agent.service.value,
                    original_count=len(recommendations),
                    limited_count=max_recommendations,
                )

            return recommendations

        except Exception as e:
            logger.error(
                "Failed to analyze service resources",
                service=agent.service.value,
                error=str(e),
            )
            return []

    async def _analyze_service_resources_with_tracking(
        self,
        agent: ServiceAgent,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> Tuple[List[Recommendation], List[Dict[str, Any]]]:
        """Analyze resources for a specific service and track those without recommendations"""
        try:
            recommendations = await agent.analyze_multiple_resources(
                resources, metrics_data, billing_data
            )

            # Apply service-specific limits
            max_recommendations = self.config.max_recommendations_per_service
            if len(recommendations) > max_recommendations:
                # Sort by savings and take top recommendations
                recommendations.sort(
                    key=lambda r: r.estimated_monthly_savings, reverse=True
                )
                recommendations = recommendations[:max_recommendations]

                logger.info(
                    "Applied recommendation limit",
                    service=agent.service.value,
                    original_count=len(recommendations),
                    limited_count=max_recommendations,
                )

            # Find resources that didn't get recommendations
            recommended_resource_ids = {rec.resource_id for rec in recommendations}
            resources_without_recommendations = []
            
            for resource in resources:
                if resource.resource_id not in recommended_resource_ids:
                    # Get relevant metrics and billing data for this resource
                    resource_metrics = (
                        metrics_data.get(resource.resource_id) if metrics_data else None
                    )
                    resource_billing = (
                        billing_data.get(resource.resource_id) if billing_data else None
                    )
                    
                    reason = await self._determine_no_recommendation_reason(
                        resource, resource_metrics, resource_billing, agent
                    )
                    resources_without_recommendations.append({
                        "resource_id": resource.resource_id,
                        "service": resource.service.value,
                        "reason": reason,
                        "agent_used": agent.agent_id
                    })

            return recommendations, resources_without_recommendations

        except Exception as e:
            logger.error(
                "Failed to analyze service resources",
                service=agent.service.value,
                error=str(e),
            )
            return [], []

    async def _determine_no_recommendation_reason(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
        agent: ServiceAgent = None,
    ) -> str:
        """Determine why a resource didn't get recommendations"""
        try:
            # Check data quality first
            if hasattr(agent, 'data_validator'):
                data_quality = agent.data_validator.validate_resource_data(
                    resource, metrics, billing_data
                )
                
                if not data_quality["recommendations_possible"]:
                    missing_data = data_quality.get("missing_critical_data", [])
                    if missing_data:
                        return f"Insufficient data: missing {', '.join(missing_data)}"
                    else:
                        return "Insufficient data quality for analysis"
            
            # Check if resource is already optimized
            if metrics:
                if hasattr(metrics, 'is_idle') and metrics.is_idle:
                    return "Resource appears to be idle/unused"
                
                # Check utilization patterns
                cpu_low = (metrics.cpu_utilization_p95 or 0) < 10
                memory_low = (metrics.memory_utilization_p95 or 0) < 30
                
                if cpu_low and memory_low:
                    return "Resource utilization is very low - may need further investigation"
                
                # Check if utilization is optimal
                cpu_optimal = 30 <= (metrics.cpu_utilization_p90 or 0) <= 80
                memory_optimal = 40 <= (metrics.memory_utilization_p90 or 0) <= 85
                
                if cpu_optimal and memory_optimal:
                    return "Resource appears to be optimally utilized"
            
            # Check cost implications
            if billing_data:
                monthly_cost = sum(bd.unblended_cost for bd in billing_data)
                if monthly_cost < 5.0:  # Very low cost
                    return f"Low monthly cost (${monthly_cost:.2f}) - optimization may not be cost-effective"
            
            # Check resource state
            state = resource.properties.get('state', '').lower()
            if state in ['stopped', 'terminated', 'deleted']:
                return f"Resource is {state} - no active optimization needed"
            
            # Default reason for unknown cases
            return "No optimization opportunities identified with current configuration"
            
        except Exception as e:
            logger.warning(
                "Failed to determine no-recommendation reason",
                resource_id=resource.resource_id,
                error=str(e)
            )
            return "Unable to determine reason for no recommendations"

    def _post_process_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Post-process recommendations to deduplicate and rank"""
        if not recommendations:
            return []

        # Remove low-impact recommendations if configured
        if not self.config.include_low_impact:
            recommendations = [
                rec for rec in recommendations if rec.estimated_monthly_savings >= 10.0
            ]

        # Deduplicate similar recommendations
        deduplicated = self._deduplicate_recommendations(recommendations)

        # Rank recommendations
        ranked = self._rank_recommendations(deduplicated)

        return ranked

    def _deduplicate_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Remove duplicate or very similar recommendations"""
        # Group by resource_id
        by_resource = defaultdict(list)
        for rec in recommendations:
            by_resource[rec.resource_id].append(rec)

        deduplicated = []

        for resource_id, resource_recs in by_resource.items():
            if len(resource_recs) == 1:
                deduplicated.extend(resource_recs)
            else:
                # Keep the recommendation with highest savings
                best_rec = max(resource_recs, key=lambda r: r.estimated_monthly_savings)
                deduplicated.append(best_rec)

                logger.debug(
                    "Deduplicated recommendations",
                    resource_id=resource_id,
                    original_count=len(resource_recs),
                    kept_recommendation=best_rec.id,
                )

        return deduplicated

    def _rank_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Rank recommendations by composite score"""

        def calculate_score(rec: Recommendation) -> float:
            # Normalize factors
            savings_score = min(
                rec.estimated_monthly_savings / 100.0, 1.0
            )  # Normalize to 0-1

            risk_scores = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.6,
                RiskLevel.HIGH: 0.2,
            }
            risk_score = risk_scores.get(rec.risk_level, 0.5)

            confidence_score = rec.confidence_score

            # Implementation ease (inverse of steps count)
            impl_ease_score = max(1.0 - len(rec.implementation_steps) / 10.0, 0.1)

            # Calculate weighted score
            score = (
                savings_score * self.config.savings_weight
                + risk_score * self.config.risk_weight
                + confidence_score * self.config.confidence_weight
                + impl_ease_score * self.config.implementation_ease_weight
            )

            return score

        # Sort by score (descending)
        recommendations.sort(key=calculate_score, reverse=True)

        return recommendations

    def _generate_report(
        self,
        recommendations: List[Recommendation],
        resources: List[Resource],
        start_time: datetime,
        resources_without_recommendations: List[Dict[str, Any]] = None,
    ) -> RecommendationReport:
        """Generate comprehensive recommendation report"""

        # Calculate summary metrics
        total_monthly_savings = sum(
            rec.estimated_monthly_savings for rec in recommendations
        )
        total_annual_savings = sum(rec.annual_savings for rec in recommendations)

        # Risk distribution
        risk_counts = defaultdict(int)
        for rec in recommendations:
            risk_counts[rec.risk_level] += 1

        # Savings by service
        savings_by_service = defaultdict(float)
        for rec in recommendations:
            savings_by_service[rec.service] += rec.estimated_monthly_savings

        # Implementation timeline
        quick_wins = []
        medium_term = []
        long_term = []

        for rec in recommendations:
            if rec.risk_level == RiskLevel.LOW and len(rec.implementation_steps) <= 2:
                quick_wins.append(rec.id)
            elif (
                rec.risk_level == RiskLevel.MEDIUM or len(rec.implementation_steps) <= 5
            ):
                medium_term.append(rec.id)
            else:
                long_term.append(rec.id)

        # Coverage metrics - default agent fallback
        analyzed_services = set(resource.service for resource in resources)
        available_agents = set(self.service_agents.keys())
        
        # Services with specific agents
        services_with_specific_agents = analyzed_services.intersection(available_agents - {ServiceType.DEFAULT})
        
        # Services that fall back to default agent
        services_falling_back_to_default = analyzed_services - (available_agents - {ServiceType.DEFAULT})
        
        # Track resources by coverage type with detailed information
        resources_with_specific_agents = []
        resources_falling_back_to_default = []
        fallback_services_list = []
        
        for resource in resources:
            if resource.service in available_agents and resource.service != ServiceType.DEFAULT:
                # Resource has a specific agent
                resources_with_specific_agents.append(resource.resource_id)
            else:
                # Resource falls back to default agent
                resources_falling_back_to_default.append(resource.resource_id)
                # Track which services are falling back
                if resource.service.value not in fallback_services_list:
                    fallback_services_list.append(resource.service.value)

        coverage = {
            "total_resources": len(resources),
            "resources_with_specific_agents": len(resources_with_specific_agents),
            "services_with_specific_agents": len(services_with_specific_agents),
            "resources_with_specific_agents_list": resources_with_specific_agents,
            "services_with_specific_agents_list": [s.value for s in services_with_specific_agents],
            "resources_falling_back_to_default": len(resources_falling_back_to_default),
            "services_falling_back_to_default": len(services_falling_back_to_default),
            "resources_falling_back_to_default_list": resources_falling_back_to_default,
            "services_falling_back_to_default_list": fallback_services_list,
            "analysis_time_seconds": (
                datetime.now(timezone.utc) - start_time
            ).total_seconds(),
        }

        report_id = f"report_{start_time.strftime('%Y%m%d_%H%M%S')}"

        return RecommendationReport(
            id=report_id,
            generated_at=datetime.now(timezone.utc),
            total_monthly_savings=total_monthly_savings,
            total_annual_savings=total_annual_savings,
            total_recommendations=len(recommendations),
            recommendations=recommendations,
            low_risk_count=risk_counts[RiskLevel.LOW],
            medium_risk_count=risk_counts[RiskLevel.MEDIUM],
            high_risk_count=risk_counts[RiskLevel.HIGH],
            savings_by_service=dict(savings_by_service),
            quick_wins=quick_wins,
            medium_term=medium_term,
            long_term=long_term,
            coverage=coverage,
            resources_without_recommendations=resources_without_recommendations or [],
        )

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all service agents"""
        status = {
            "coordinator": {
                "enabled_services": [s.value for s in self.config.enabled_services],
                "active_agents": len(self.service_agents),
                "config": {
                    "similarity_threshold": self.config.similarity_threshold,
                    "max_recommendations_per_service": self.config.max_recommendations_per_service,
                    "include_low_impact": self.config.include_low_impact,
                },
            },
            "agents": {},
        }

        for service, agent in self.service_agents.items():
            status["agents"][service.value] = agent.get_capabilities()

        return status
