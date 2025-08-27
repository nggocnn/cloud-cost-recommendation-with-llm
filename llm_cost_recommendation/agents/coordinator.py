"""
Coordinator agent that orchestrates service agents and consolidates recommendations.
"""

import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timezone

from ..models import (
    Resource,
    Metrics,
    BillingData,
    Recommendation,
    RecommendationReport,
    ServiceType,
    ServiceAgentConfig,
    RiskLevel,
)
from ..services.llm import LLMService
from ..services.config import ConfigManager
from ..services.logging import get_logger
from .base import BaseAgent

logger = get_logger(__name__)


class ServiceAgentFactory:
    """Factory for creating service agents"""

    @staticmethod
    def create_agent(config: ServiceAgentConfig, llm_service: LLMService) -> BaseAgent:
        """Create appropriate agent for service"""
        
        # All services now use the unified LLM-based agent with custom rules
        return LLMServiceAgent(config, llm_service)


class LLMServiceAgent(BaseAgent):
    """Generic LLM-based service agent"""

    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze resource using LLM with custom rules support"""
        recommendations = []

        if not self._validate_resource_data(resource):
            return recommendations

        try:
            # Calculate estimated monthly cost for rule evaluation
            estimated_cost = None
            if billing_data:
                estimated_cost = sum(bd.unblended_cost for bd in billing_data)

            # Apply custom rules to get configuration overrides
            rule_results = self._apply_custom_rules(
                resource, metrics, billing_data, estimated_cost
            )

            # Merge thresholds with rule overrides
            context_data = self._prepare_context_data(resource, metrics, billing_data)
            original_thresholds = context_data.get("thresholds", {})
            merged_thresholds = self._merge_thresholds(
                original_thresholds, rule_results.get("threshold_overrides", {})
            )
            context_data["thresholds"] = merged_thresholds

            # Generate recommendations using LLM with rule-modified context
            llm_recommendations = await self._generate_recommendations_from_llm(
                context_data, rule_results
            )

            # Convert to recommendation models
            for llm_rec in llm_recommendations:
                rec = self._convert_llm_recommendation_to_model(llm_rec, resource)
                if rec:
                    recommendations.append(rec)

            logger.info(
                "Resource analysis completed",
                agent_id=self.agent_id,
                resource_id=resource.resource_id,
                recommendations_count=len(recommendations),
                rules_applied=len([r for r in self.config.custom_rules if r.enabled]),
                threshold_overrides=rule_results.get("threshold_overrides", {}),
            )

        except Exception as e:
            logger.error(
                "Failed to analyze resource with LLM",
                agent_id=self.agent_id,
                resource_id=resource.resource_id,
                error=str(e),
            )

        return recommendations


class CoordinatorAgent:
    """Coordinator agent that orchestrates service agents"""

    def __init__(self, config_manager: ConfigManager, llm_service: LLMService):
        self.config_manager = config_manager
        self.llm_service = llm_service
        self.config = config_manager.coordinator_config

        # Initialize service agents
        self.service_agents: Dict[ServiceType, BaseAgent] = {}
        self._initialize_agents()

        logger.info(
            "Coordinator agent initialized", enabled_services=len(self.service_agents)
        )

    def _initialize_agents(self):
        """Initialize service agents based on configuration"""
        for service in self.config.enabled_services:
            service_config = self.config_manager.get_service_config(service)

            if service_config and service_config.enabled:
                try:
                    agent = ServiceAgentFactory.create_agent(
                        service_config, self.llm_service
                    )
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

    async def analyze_account(
        self,
        account_id: str,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
        batch_mode: bool = True,
    ) -> RecommendationReport:
        """Analyze entire account and generate comprehensive report"""
        logger.info(
            "Starting account analysis",
            account_id=account_id,
            total_resources=len(resources),
        )

        start_time = datetime.now(timezone.utc)

        # Group resources by service
        resources_by_service = self._group_resources_by_service(resources)

        # Analyze each service
        all_recommendations = []
        
        if batch_mode:
            # Batch processing: analyze services in parallel
            analysis_tasks = []

            for service, service_resources in resources_by_service.items():
                if service in self.service_agents:
                    agent = self.service_agents[service]

                    # Create analysis task
                    task = self._analyze_service_resources(
                        agent, service_resources, metrics_data, billing_data
                    )
                    analysis_tasks.append(task)

            # Execute analysis tasks
            if analysis_tasks:
                service_recommendations = await asyncio.gather(
                    *analysis_tasks, return_exceptions=True
                )

                for result in service_recommendations:
                    if isinstance(result, Exception):
                        logger.error("Service analysis failed", error=str(result))
                    else:
                        all_recommendations.extend(result)
        else:
            # Individual processing: analyze resources one by one
            total_resources = len(resources)
            for i, resource in enumerate(resources, 1):
                if resource.service in self.service_agents:
                    logger.info(
                        f"Analyzing resource {i}/{total_resources}",
                        resource_id=resource.resource_id,
                        service=resource.service.value,
                    )
                    
                    # Get relevant metrics and billing data for this resource
                    resource_metrics = metrics_data.get(resource.resource_id) if metrics_data else None
                    resource_billing = billing_data.get(resource.resource_id) if billing_data else None
                    
                    try:
                        recommendations = await self.analyze_resource(
                            resource, resource_metrics, resource_billing
                        )
                        all_recommendations.extend(recommendations)
                    except Exception as e:
                        logger.error(
                            "Failed to analyze individual resource",
                            resource_id=resource.resource_id,
                            service=resource.service.value,
                            error=str(e),
                        )

        # Post-process recommendations
        processed_recommendations = self._post_process_recommendations(
            all_recommendations
        )

        # Generate report
        report = self._generate_report(
            account_id, processed_recommendations, resources, start_time
        )

        logger.info(
            "Account analysis completed",
            account_id=account_id,
            total_recommendations=len(processed_recommendations),
            total_savings=report.total_monthly_savings,
            analysis_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
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
        agent: BaseAgent,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze resources for a specific service"""
        try:
            recommendations = await agent.analyze_resources(
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
        account_id: str,
        recommendations: List[Recommendation],
        resources: List[Resource],
        start_time: datetime,
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

        # Coverage metrics
        analyzed_services = set(resource.service for resource in resources)
        available_agents = set(self.service_agents.keys())

        coverage = {
            "total_resources": len(resources),
            "total_services": len(analyzed_services),
            "covered_services": len(analyzed_services.intersection(available_agents)),
            "uncovered_services": list(analyzed_services - available_agents),
            "analysis_time_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
        }

        report_id = f"report_{account_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        return RecommendationReport(
            id=report_id,
            account_id=account_id,
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

    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze a single resource"""
        if resource.service not in self.service_agents:
            logger.warning(
                "No agent available for service",
                service=resource.service.value,
                resource_id=resource.resource_id,
            )
            return []

        agent = self.service_agents[resource.service]
        return await agent.analyze_resource(resource, metrics, billing_data)
