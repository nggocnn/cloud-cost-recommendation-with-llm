"""
Base agent class for service-specific cost optimization agents.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models import (
    Resource,
    Metrics,
    BillingData,
    Recommendation,
    RecommendationType,
    RiskLevel,
    ServiceAgentConfig,
)
from ..services.llm import LLMService, PromptTemplates
from ..services.logging import get_logger
from ..services.conditions import RuleProcessor

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for service-specific cost optimization agents"""

    def __init__(self, config: ServiceAgentConfig, llm_service: LLMService):
        self.config = config
        self.llm_service = llm_service
        self.service = config.service
        self.agent_id = config.agent_id
        self.rule_processor = RuleProcessor()

        logger.info("Agent initialized", agent_id=self.agent_id, service=self.service)

    @abstractmethod
    async def analyze_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze a single resource and generate recommendations"""
        pass

    async def analyze_resources(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze multiple resources"""
        recommendations = []

        for resource in resources:
            if resource.service != self.service:
                continue

            try:
                resource_metrics = (
                    metrics_data.get(resource.resource_id) if metrics_data else None
                )
                resource_billing = (
                    billing_data.get(resource.resource_id) if billing_data else None
                )

                resource_recommendations = await self.analyze_resource(
                    resource, resource_metrics, resource_billing
                )

                recommendations.extend(resource_recommendations)

            except Exception as e:
                logger.error(
                    "Failed to analyze resource",
                    resource_id=resource.resource_id,
                    agent_id=self.agent_id,
                    error=str(e),
                )
                continue

        logger.info(
            "Resource analysis completed",
            agent_id=self.agent_id,
            total_resources=len(resources),
            total_recommendations=len(recommendations),
        )

        return recommendations

    def _prepare_context_data(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> Dict[str, Any]:
        """Prepare context data for LLM analysis"""
        context = {
            "resource": {
                "resource_id": resource.resource_id,
                "service": resource.service.value,
                "region": resource.region,
                "availability_zone": resource.availability_zone,
                "account_id": resource.account_id,
                "tags": resource.tags,
                "properties": resource.properties,
                "extensions": resource.extensions,
            }
        }

        if metrics:
            context["metrics"] = {
                "period_days": metrics.period_days,
                "is_idle": metrics.is_idle,
                "cpu_utilization_p50": metrics.cpu_utilization_p50,
                "cpu_utilization_p90": metrics.cpu_utilization_p90,
                "cpu_utilization_p95": metrics.cpu_utilization_p95,
                "memory_utilization_p50": metrics.memory_utilization_p50,
                "memory_utilization_p90": metrics.memory_utilization_p90,
                "memory_utilization_p95": metrics.memory_utilization_p95,
                "iops_read": metrics.iops_read,
                "iops_write": metrics.iops_write,
                "throughput_read": metrics.throughput_read,
                "throughput_write": metrics.throughput_write,
                "network_in": metrics.network_in,
                "network_out": metrics.network_out,
                "other_metrics": metrics.metrics,
            }

        if billing_data:
            context["billing"] = {
                "total_monthly_cost": sum(bd.unblended_cost for bd in billing_data),
                "usage_patterns": [
                    {
                        "usage_type": bd.usage_type,
                        "usage_amount": bd.usage_amount,
                        "usage_unit": bd.usage_unit,
                        "cost": bd.unblended_cost,
                    }
                    for bd in billing_data
                ],
            }

        # Add service-specific thresholds
        context["thresholds"] = self.config.capability.thresholds
        context["analysis_window_days"] = self.config.capability.analysis_window_days

        return context

    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        return f"{PromptTemplates.BASE_SYSTEM_PROMPT}\n\n{self.config.base_prompt}\n\n{self.config.service_specific_prompt}"

    def _create_user_prompt(self, context_data: Dict[str, Any]) -> str:
        """Create user prompt with context data"""
        prompt_parts = [
            f"Analyze the following {self.service.value} resource for cost optimization opportunities:",
            "",
            "Resource Information:",
            json.dumps(context_data.get("resource", {}), indent=2),
            "",
        ]

        if "metrics" in context_data:
            prompt_parts.extend(
                [
                    "Performance Metrics:",
                    json.dumps(context_data["metrics"], indent=2),
                    "",
                ]
            )

        if "billing" in context_data:
            prompt_parts.extend(
                [
                    "Billing Information:",
                    json.dumps(context_data["billing"], indent=2),
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "Configuration:",
                f"- Analysis window: {context_data.get('analysis_window_days', 30)} days",
                f"- Service thresholds: {json.dumps(context_data.get('thresholds', {}), indent=2)}",
                "",
                "Please provide cost optimization recommendations in the specified JSON format.",
            ]
        )

        return "\n".join(prompt_parts)

    def _apply_custom_rules(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
        computed_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply custom conditional rules to modify agent behavior"""
        if not self.config.custom_rules:
            return {
                "threshold_overrides": {},
                "skip_recommendation_types": [],
                "force_recommendation_types": [],
                "custom_prompts": [],
                "risk_adjustments": [],
                "actions": {},
            }

        return self.rule_processor.apply_rules(
            rules=self.config.custom_rules,
            resource=resource,
            metrics=metrics,
            billing_data=billing_data,
            computed_cost=computed_cost,
            base_thresholds=self.config.capability.thresholds,
        )

    def _merge_thresholds(
        self, base_thresholds: Dict[str, float], overrides: Dict[str, float]
    ) -> Dict[str, float]:
        """Merge base thresholds with rule-based overrides"""
        merged = base_thresholds.copy()
        merged.update(overrides)

        logger.debug(
            "Thresholds merged",
            agent_id=self.agent_id,
            base_thresholds=base_thresholds,
            overrides=overrides,
            merged_thresholds=merged,
        )

        return merged

    async def _generate_recommendations_from_llm(
        self,
        context_data: Dict[str, Any],
        rule_results: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using LLM"""
        try:
            # Modify system prompt with custom prompts if available
            system_prompt = self._create_system_prompt()
            if rule_results and rule_results.get("custom_prompts"):
                custom_prompt_section = "\n\nADDITIONAL CUSTOM RULES:\n" + "\n".join(
                    rule_results["custom_prompts"]
                )
                system_prompt += custom_prompt_section

            # Add rule-based restrictions to user prompt
            user_prompt = self._create_user_prompt(context_data)
            if rule_results:
                rule_instructions = []

                if rule_results.get("skip_recommendation_types"):
                    skip_types = [
                        rt.value for rt in rule_results["skip_recommendation_types"]
                    ]
                    rule_instructions.append(
                        f"IMPORTANT: Do NOT recommend these types: {', '.join(skip_types)}"
                    )

                if rule_results.get("force_recommendation_types"):
                    force_types = [
                        rt.value for rt in rule_results["force_recommendation_types"]
                    ]
                    rule_instructions.append(
                        f"IMPORTANT: Always consider these recommendation types: {', '.join(force_types)}"
                    )

                if rule_instructions:
                    user_prompt += "\n\nCUSTOM RULE RESTRICTIONS:\n" + "\n".join(
                        rule_instructions
                    )

            response = await self.llm_service.generate_recommendation(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_data=context_data,
            )

            # Parse JSON response
            try:
                response_data = json.loads(response.content)
                recommendations = response_data.get("recommendations", [])

                # Apply rule-based filtering
                if rule_results:
                    recommendations = self._filter_recommendations_by_rules(
                        recommendations, rule_results
                    )

                logger.info(
                    "LLM recommendations generated",
                    agent_id=self.agent_id,
                    recommendations_count=len(recommendations),
                    response_time_ms=response.response_time_ms,
                )

                return recommendations

            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse LLM response as JSON",
                    agent_id=self.agent_id,
                    response_content=response.content[:500],
                    error=str(e),
                )
                return []

        except Exception as e:
            logger.error(
                "Failed to generate recommendations from LLM",
                agent_id=self.agent_id,
                error=str(e),
            )
            return []

    def _filter_recommendations_by_rules(
        self, recommendations: List[Dict[str, Any]], rule_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter and modify recommendations based on custom rules"""
        filtered_recommendations = []

        skip_types = [
            rt.value for rt in rule_results.get("skip_recommendation_types", [])
        ]
        force_types = [
            rt.value for rt in rule_results.get("force_recommendation_types", [])
        ]
        risk_adjustments = rule_results.get("risk_adjustments", [])

        for rec in recommendations:
            rec_type = rec.get("recommendation_type", "")

            # Skip if recommendation type is blacklisted
            if rec_type in skip_types:
                logger.debug(
                    "Recommendation filtered out by custom rule",
                    agent_id=self.agent_id,
                    recommendation_type=rec_type,
                    resource_id=rec.get("resource_id"),
                )
                continue

            # Apply risk level adjustments
            if risk_adjustments:
                current_risk = rec.get("risk_level", "medium")
                if "increase" in risk_adjustments:
                    if current_risk == "low":
                        rec["risk_level"] = "medium"
                    elif current_risk == "medium":
                        rec["risk_level"] = "high"
                elif "decrease" in risk_adjustments:
                    if current_risk == "high":
                        rec["risk_level"] = "medium"
                    elif current_risk == "medium":
                        rec["risk_level"] = "low"

                logger.debug(
                    "Risk level adjusted by custom rule",
                    agent_id=self.agent_id,
                    recommendation_type=rec_type,
                    original_risk=current_risk,
                    adjusted_risk=rec.get("risk_level"),
                )

            filtered_recommendations.append(rec)

        logger.info(
            "Recommendations filtered by custom rules",
            agent_id=self.agent_id,
            original_count=len(recommendations),
            filtered_count=len(filtered_recommendations),
            skip_types=skip_types,
            force_types=force_types,
        )

        return filtered_recommendations

    def _convert_llm_recommendation_to_model(
        self, llm_recommendation: Dict[str, Any], resource: Resource
    ) -> Optional[Recommendation]:
        """Convert LLM recommendation dict to Recommendation model"""
        try:
            # Generate unique ID
            rec_id = f"{self.agent_id}_{resource.resource_id}_{datetime.utcnow().isoformat()}"

            # Parse recommendation type
            rec_type_str = llm_recommendation.get("recommendation_type", "rightsizing")
            try:
                rec_type = RecommendationType(rec_type_str)
            except ValueError:
                rec_type = RecommendationType.RIGHTSIZING

            # Parse risk level
            risk_str = llm_recommendation.get("risk_level", "medium")
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM

            # Calculate savings
            current_cost = float(llm_recommendation.get("current_monthly_cost", 0))
            estimated_cost = float(llm_recommendation.get("estimated_monthly_cost", 0))
            monthly_savings = current_cost - estimated_cost
            annual_savings = monthly_savings * 12

            # Apply minimum cost threshold
            if monthly_savings < self.config.min_cost_threshold:
                logger.debug(
                    "Recommendation below minimum cost threshold",
                    resource_id=resource.resource_id,
                    monthly_savings=monthly_savings,
                    threshold=self.config.min_cost_threshold,
                )
                return None

            # Apply confidence threshold
            confidence = float(llm_recommendation.get("confidence_score", 0.5))
            if confidence < self.config.confidence_threshold:
                logger.debug(
                    "Recommendation below confidence threshold",
                    resource_id=resource.resource_id,
                    confidence=confidence,
                    threshold=self.config.confidence_threshold,
                )
                return None

            recommendation = Recommendation(
                id=rec_id,
                resource_id=resource.resource_id,
                service=self.service,
                recommendation_type=rec_type,
                current_config=llm_recommendation.get("current_config", {}),
                recommended_config=llm_recommendation.get("recommended_config", {}),
                current_monthly_cost=current_cost,
                estimated_monthly_cost=estimated_cost,
                estimated_monthly_savings=monthly_savings,
                annual_savings=annual_savings,
                risk_level=risk_level,
                impact_description=llm_recommendation.get("impact_description", ""),
                rollback_plan=llm_recommendation.get("rollback_plan", ""),
                rationale=llm_recommendation.get("rationale", ""),
                implementation_steps=llm_recommendation.get("implementation_steps", []),
                confidence_score=confidence,
                agent_id=self.agent_id,
            )

            return recommendation

        except Exception as e:
            logger.error(
                "Failed to convert LLM recommendation to model",
                agent_id=self.agent_id,
                resource_id=resource.resource_id,
                error=str(e),
            )
            return None

    def _validate_resource_data(self, resource: Resource) -> bool:
        """Validate if resource has required data for analysis"""
        if not resource.resource_id:
            return False

        if resource.service != self.service:
            return False

        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        return {
            "agent_id": self.agent_id,
            "service": self.service.value,
            "supported_recommendation_types": [
                rt.value for rt in self.config.capability.supported_recommendation_types
            ],
            "required_metrics": self.config.capability.required_metrics,
            "optional_metrics": self.config.capability.optional_metrics,
            "thresholds": self.config.capability.thresholds,
            "enabled": self.config.enabled,
        }
