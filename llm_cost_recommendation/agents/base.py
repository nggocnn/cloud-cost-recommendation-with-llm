"""
Base agent class for service-specific cost optimization agents.
"""

import asyncio
import json
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
    GlobalConfig,
)
from ..services.llm import LLMService, PromptTemplates
from ..utils.logging import get_logger
from ..services.conditions import RuleProcessor

logger = get_logger(__name__)


class ServiceAgent:
    """Service-specific cost optimization agent"""

    def __init__(
        self,
        agent_config: ServiceAgentConfig,
        llm_service: LLMService,
        global_config: GlobalConfig,
    ):
        self.agent_config = agent_config
        self.llm_service = llm_service
        self.global_config = global_config
        self.service = agent_config.service
        self.agent_id = agent_config.agent_id
        self.rule_processor = RuleProcessor()

        logger.debug("Agent initialized", agent_id=self.agent_id, service=self.service)

    async def analyze_single_resource(
        self,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze a single resource and generate recommendations"""
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
                rec = await self._convert_llm_recommendation_to_model(llm_rec, resource)
                if rec:
                    recommendations.append(rec)

            logger.info(
                "Resource analysis completed",
                agent_id=self.agent_id,
                resource_id=resource.resource_id,
                recommendations_count=len(recommendations),
                rules_applied=len(
                    [r for r in self.agent_config.custom_rules if r.enabled]
                ),
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

    async def analyze_multiple_resources(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze multiple resources using intelligent batching strategy"""
        recommendations = []

        # Filter resources for this service
        service_resources = [r for r in resources if r.service == self.service]

        if not service_resources:
            return recommendations

        # Use intelligent batching for efficiency
        batches = self._create_intelligent_batches(
            service_resources, metrics_data, billing_data
        )

        logger.info(
            "Starting intelligent batch analysis",
            agent_id=self.agent_id,
            total_resources=len(service_resources),
            total_batches=len(batches),
        )

        # Process batches in parallel for different groups
        batch_tasks = []
        for batch_info in batches:
            task = self._route_batch_analysis(batch_info, metrics_data, billing_data)
            batch_tasks.append(task)

        # Execute batches in parallel
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Collect recommendations from all batches
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(
                    "Batch processing failed",
                    agent_id=self.agent_id,
                    error=str(result),
                )
                continue
            recommendations.extend(result)

        logger.info(
            "Resource analysis completed",
            agent_id=self.agent_id,
            total_resources=len(service_resources),
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
        context["thresholds"] = self.agent_config.capability.thresholds
        context["analysis_window_days"] = (
            self.agent_config.capability.analysis_window_days
        )

        return context

    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        return f"{PromptTemplates.BASE_SYSTEM_PROMPT}\n\n{self.agent_config.base_prompt}\n\n{self.agent_config.service_specific_prompt}"

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
        if not self.agent_config.custom_rules:
            return {
                "threshold_overrides": {},
                "skip_recommendation_types": [],
                "force_recommendation_types": [],
                "custom_prompts": [],
                "risk_adjustments": [],
                "actions": {},
            }

        return self.rule_processor.apply_rules(
            rules=self.agent_config.custom_rules,
            resource=resource,
            metrics=metrics,
            billing_data=billing_data,
            computed_cost=computed_cost,
            base_thresholds=self.agent_config.capability.thresholds,
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

        logger.debug(
            "Recommendations filtered by custom rules",
            agent_id=self.agent_id,
            original_count=len(recommendations),
            filtered_count=len(filtered_recommendations),
            skip_types=skip_types,
            force_types=force_types,
        )

        return filtered_recommendations

    async def _convert_llm_recommendation_to_model(
        self, llm_recommendation: Dict[str, Any], resource: Resource
    ) -> Optional[Recommendation]:
        """Convert LLM recommendation dict to Recommendation model"""
        try:
            # Validate required fields - but be flexible for now
            required_fields = [
                "recommendation_type",
                "current_monthly_cost",
                "estimated_monthly_cost",
                "confidence_score",
            ]
            missing_fields = [
                field for field in required_fields if field not in llm_recommendation
            ]

            if missing_fields:
                logger.warning(
                    "LLM response missing critical fields",
                    agent_id=self.agent_id,
                    resource_id=resource.resource_id,
                    missing_fields=missing_fields,
                )
                return None

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

            # Calculate costs using LLM estimates
            current_cost = float(llm_recommendation.get("current_monthly_cost", 0))
            estimated_cost = float(llm_recommendation.get("estimated_monthly_cost", 0))

            monthly_savings = current_cost - estimated_cost
            annual_savings = monthly_savings * 12

            # Apply minimum cost threshold
            if monthly_savings < self.agent_config.min_cost_threshold:
                logger.debug(
                    "Recommendation below minimum cost threshold",
                    resource_id=resource.resource_id,
                    monthly_savings=monthly_savings,
                    threshold=self.agent_config.min_cost_threshold,
                )
                return None

            # Apply confidence threshold
            confidence = float(llm_recommendation.get("confidence_score", 0.5))
            if confidence < self.agent_config.confidence_threshold:
                logger.debug(
                    "Recommendation below confidence threshold",
                    resource_id=resource.resource_id,
                    confidence=confidence,
                    threshold=self.agent_config.confidence_threshold,
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
                impact_description=llm_recommendation["impact_description"],
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
                rt.value
                for rt in self.agent_config.capability.supported_recommendation_types
            ],
            "required_metrics": self.agent_config.capability.required_metrics,
            "optional_metrics": self.agent_config.capability.optional_metrics,
            "thresholds": self.agent_config.capability.thresholds,
            "enabled": self.agent_config.enabled,
        }

    def _create_intelligent_batches(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Dict[str, Any]]:
        """Create intelligent batches based on resource similarity and complexity"""
        batches = []

        # Group resources by similarity criteria
        resource_groups = self._group_resources_by_similarity(
            resources, metrics_data, billing_data
        )

        for group_key, group_resources in resource_groups.items():
            # Determine optimal batch size for this group
            batch_size = self._calculate_optimal_batch_size(group_resources, group_key)

            # Create batches within the group
            for i in range(0, len(group_resources), batch_size):
                batch_resources = group_resources[i : i + batch_size]

                batches.append(
                    {
                        "group_key": group_key,
                        "resources": batch_resources,
                        "batch_size": len(batch_resources),
                        "batch_index": i // batch_size,
                        "total_batches_in_group": (
                            len(group_resources) + batch_size - 1
                        )
                        // batch_size,
                    }
                )

        return batches

    def _group_resources_by_similarity(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> Dict[str, List[Resource]]:
        """Group resources by similarity for efficient batching"""
        groups = {}

        for resource in resources:
            # Create grouping key based on resource characteristics
            group_key = self._create_resource_group_key(
                resource, metrics_data, billing_data
            )

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(resource)

        return groups

    def _create_resource_group_key(
        self,
        resource: Resource,
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> str:
        """Create a grouping key for similar resources"""
        key_parts = [
            resource.service.value,
            resource.region or "unknown",
        ]

        # Add resource type/configuration similarity from properties
        if "instance_type" in resource.properties:
            key_parts.append(resource.properties["instance_type"] or "unknown")
        elif "storage_class" in resource.properties:
            key_parts.append(resource.properties["storage_class"] or "unknown")
        elif "engine" in resource.properties:
            key_parts.append(resource.properties["engine"] or "unknown")
        else:
            key_parts.append("unknown")

        # Add cost tier for batch sizing
        cost_tier = self._get_resource_cost_tier(resource, billing_data)
        key_parts.append(cost_tier)

        # Add complexity tier based on available metrics
        complexity_tier = self._get_resource_complexity_tier(resource, metrics_data)
        key_parts.append(complexity_tier)

        return "|".join(key_parts)

    def _get_resource_cost_tier(
        self,
        resource: Resource,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> str:
        """Determine cost tier for batching strategy"""
        if not billing_data or resource.resource_id not in billing_data:
            return "unknown_cost"

        monthly_cost = sum(
            bd.unblended_cost for bd in billing_data[resource.resource_id]
        )
        cost_tiers = self.global_config.cost_tiers

        # Find the appropriate cost tier based on monthly cost
        for tier_name, tier_config in cost_tiers.items():
            min_cost = tier_config.get("min", 0)
            max_cost = tier_config.get("max", float("inf"))

            if min_cost <= monthly_cost < max_cost:
                return tier_name

        # Fallback to unknown if no tier matches
        return "unknown_cost"

    def _should_analyze_individually(
        self,
        resource: Resource,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> bool:
        """Determine if a resource should be analyzed individually based on cost threshold"""
        if not billing_data or resource.resource_id not in billing_data:
            return False

        monthly_cost = sum(
            bd.unblended_cost for bd in billing_data[resource.resource_id]
        )
        threshold = self.global_config.batch_config.get(
            "single_resource_threshold_cost", 5000
        )

        return monthly_cost >= threshold

    def _get_resource_complexity_tier(
        self,
        resource: Resource,
        metrics_data: Dict[str, Metrics] = None,
    ) -> str:
        """Determine complexity tier for batching strategy"""
        if not metrics_data or resource.resource_id not in metrics_data:
            return "simple"

        metrics = metrics_data[resource.resource_id]
        metric_count = len([v for v in metrics.metrics.values() if v is not None])
        complexity_tiers = self.global_config.complexity_tiers

        # Find the appropriate complexity tier based on metric count
        # Sort tiers by metric_threshold to process from lowest to highest
        def get_threshold(item):
            threshold = item[1].get("metric_threshold", float("inf"))
            # Handle 'inf' string from YAML
            if isinstance(threshold, str) and threshold.lower() == "inf":
                return float("inf")
            return float(threshold)

        sorted_tiers = sorted(complexity_tiers.items(), key=get_threshold)

        for tier_name, tier_config in sorted_tiers:
            threshold = tier_config.get("metric_threshold", float("inf"))
            # Handle 'inf' string from YAML
            if isinstance(threshold, str) and threshold.lower() == "inf":
                threshold = float("inf")
            else:
                threshold = float(threshold)

            if metric_count <= threshold:
                return tier_name

        # Fallback to the last tier if metric count exceeds all thresholds
        return sorted_tiers[-1][0] if sorted_tiers else "simple"

    def _calculate_optimal_batch_size(
        self,
        resources: List[Resource],
        group_key: str,
    ) -> int:
        """Calculate optimal batch size"""
        # Parse group key to understand resource characteristics
        key_parts = group_key.split("|")
        cost_tier = key_parts[-2] if len(key_parts) >= 2 else "unknown_cost"
        complexity_tier = key_parts[-1] if len(key_parts) >= 1 else "simple"

        # Get batch configuration
        batch_config = self.global_config.batch_config
        complexity_tiers = self.global_config.complexity_tiers

        # Get base batch size from complexity tier configuration
        if complexity_tier in complexity_tiers:
            base_size = complexity_tiers[complexity_tier].get(
                "base_batch_size", batch_config.get("default_batch_size", 4)
            )
        else:
            base_size = batch_config.get("default_batch_size", 4)

        # Adjust based on cost tier using configuration
        cost_tier_adjustment = self._get_cost_tier_batch_adjustment(cost_tier)
        adjusted_size = base_size + cost_tier_adjustment

        # Apply global batch size constraints
        min_batch_size = batch_config.get("min_batch_size", 1)
        max_batch_size = batch_config.get("max_batch_size", 10)

        # Ensure batch size is within configured limits
        final_size = max(
            min_batch_size, min(adjusted_size, max_batch_size, len(resources))
        )

        return final_size

    def _get_cost_tier_batch_adjustment(self, cost_tier: str) -> int:
        """Get batch size adjustment based on dynamic cost tier configuration"""
        # Get cost tier configuration from global config
        cost_tiers = self.global_config.cost_tiers

        # Return configured batch adjustment if available
        if cost_tier in cost_tiers:
            tier_config = cost_tiers[cost_tier]
            return int(tier_config.get("batch_adjustment", 0))

        # Fallback: no adjustment for unknown tiers
        return 0

    async def _route_batch_analysis(
        self,
        batch_info: Dict[str, Any],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Route batch to appropriate analysis method (individual vs batch processing)"""
        recommendations = []
        batch_resources = batch_info["resources"]

        logger.info(
            "Routing batch analysis",
            agent_id=self.agent_id,
            group_key=batch_info["group_key"],
            batch_size=batch_info["batch_size"],
            batch_index=batch_info["batch_index"],
        )

        try:
            # Check if any resource in the batch should be analyzed individually
            individual_analysis_resources = []
            batch_analysis_resources = []

            for resource in batch_resources:
                resource_billing = (
                    billing_data.get(resource.resource_id) if billing_data else None
                )
                if self._should_analyze_individually(resource, billing_data):
                    individual_analysis_resources.append(resource)
                else:
                    batch_analysis_resources.append(resource)

            # Process individual analysis resources
            for resource in individual_analysis_resources:
                resource_metrics = (
                    metrics_data.get(resource.resource_id) if metrics_data else None
                )
                resource_billing = (
                    billing_data.get(resource.resource_id) if billing_data else None
                )

                individual_recs = await self.analyze_single_resource(
                    resource, resource_metrics, resource_billing
                )
                recommendations.extend(individual_recs)

                logger.info(
                    "High-cost resource analyzed individually",
                    agent_id=self.agent_id,
                    resource_id=resource.resource_id,
                    cost=(
                        sum(bd.unblended_cost for bd in resource_billing)
                        if resource_billing
                        else 0
                    ),
                )

            # Process batch analysis resources
            if len(batch_analysis_resources) == 1:
                # Single resource - use individual analysis for high precision
                resource = batch_analysis_resources[0]
                resource_metrics = (
                    metrics_data.get(resource.resource_id) if metrics_data else None
                )
                resource_billing = (
                    billing_data.get(resource.resource_id) if billing_data else None
                )

                individual_recs = await self.analyze_single_resource(
                    resource, resource_metrics, resource_billing
                )
                recommendations.extend(individual_recs)
            elif len(batch_analysis_resources) > 1:
                # Multiple resources - use batch analysis
                batch_recs = await self._analyze_batch(
                    batch_analysis_resources, metrics_data, billing_data
                )
                recommendations.extend(batch_recs)

            logger.debug(
                "Batch routing completed",
                agent_id=self.agent_id,
                group_key=batch_info["group_key"],
                batch_size=batch_info["batch_size"],
                recommendations_count=len(recommendations),
            )

        except Exception as e:
            logger.error(
                "Failed to route batch analysis",
                agent_id=self.agent_id,
                group_key=batch_info["group_key"],
                batch_size=batch_info["batch_size"],
                error=str(e),
            )

        return recommendations

    async def _analyze_batch(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
    ) -> List[Recommendation]:
        """Analyze multiple similar resources together using LLM batch processing"""
        recommendations = []

        # Prepare batch context data with custom rules applied
        batch_context = []
        batch_rule_results = []

        for resource in resources:
            resource_metrics = (
                metrics_data.get(resource.resource_id) if metrics_data else None
            )
            resource_billing = (
                billing_data.get(resource.resource_id) if billing_data else None
            )

            # Calculate estimated monthly cost for rule evaluation
            estimated_cost = None
            if resource_billing:
                estimated_cost = sum(bd.unblended_cost for bd in resource_billing)

            # Apply custom rules to get configuration overrides for this resource
            rule_results = self._apply_custom_rules(
                resource, resource_metrics, resource_billing, estimated_cost
            )
            batch_rule_results.append(rule_results)

            # Prepare context data with rule-modified thresholds
            context_data = self._prepare_context_data(
                resource, resource_metrics, resource_billing
            )
            original_thresholds = context_data.get("thresholds", {})
            merged_thresholds = self._merge_thresholds(
                original_thresholds, rule_results.get("threshold_overrides", {})
            )
            context_data["thresholds"] = merged_thresholds
            context_data["rule_results"] = (
                rule_results  # Include rule results in context
            )

            batch_context.append(context_data)

        try:
            # Create batch system prompt with custom rules
            system_prompt = self._create_batch_system_prompt(
                len(resources), batch_rule_results
            )

            # Use LLM service for batch recommendation generation
            batch_responses = await self.llm_service.generate_batch_recommendations(
                system_prompt=system_prompt,
                resources_data=batch_context,
                batch_size=len(resources),
            )

            # Process batch responses and apply custom rule filtering
            for response in batch_responses:
                try:
                    response_data = json.loads(response.content)
                    batch_recommendations = response_data.get("recommendations", [])

                    # Convert batch recommendations to individual recommendations
                    for rec_data in batch_recommendations:
                        if "resource_id" in rec_data:
                            # Find the corresponding resource and its rule results
                            resource = next(
                                (
                                    r
                                    for r in resources
                                    if r.resource_id == rec_data["resource_id"]
                                ),
                                None,
                            )
                            resource_rule_results = next(
                                (
                                    rr
                                    for i, rr in enumerate(batch_rule_results)
                                    if resources[i].resource_id
                                    == rec_data["resource_id"]
                                ),
                                {},
                            )

                            if resource:
                                # Apply rule-based filtering to the recommendation
                                filtered_recs = self._filter_recommendations_by_rules(
                                    [rec_data], resource_rule_results
                                )

                                for filtered_rec in filtered_recs:
                                    rec = (
                                        await self._convert_llm_recommendation_to_model(
                                            filtered_rec, resource
                                        )
                                    )
                                    if rec:
                                        recommendations.append(rec)

                except json.JSONDecodeError as e:
                    logger.error(
                        "Failed to parse batch LLM response",
                        agent_id=self.agent_id,
                        error=str(e),
                    )
                    continue

        except Exception as e:
            logger.error(
                "Failed to analyze resource batch with LLM",
                agent_id=self.agent_id,
                batch_size=len(resources),
                error=str(e),
            )

            # Fallback to individual analysis
            logger.info(
                "Falling back to individual resource analysis",
                agent_id=self.agent_id,
                batch_size=len(resources),
            )

            for resource in resources:
                try:
                    resource_metrics = (
                        metrics_data.get(resource.resource_id) if metrics_data else None
                    )
                    resource_billing = (
                        billing_data.get(resource.resource_id) if billing_data else None
                    )

                    individual_recs = await self.analyze_single_resource(
                        resource, resource_metrics, resource_billing
                    )
                    recommendations.extend(individual_recs)

                except Exception as individual_error:
                    logger.error(
                        "Failed individual resource analysis in fallback",
                        resource_id=resource.resource_id,
                        agent_id=self.agent_id,
                        error=str(individual_error),
                    )
                    continue

        return recommendations

    def _create_batch_system_prompt(
        self, batch_size: int, batch_rule_results: List[Dict[str, Any]]
    ) -> str:
        """Create system prompt optimized for batch analysis with custom rules"""
        base_prompt = self._create_system_prompt()

        # Collect all custom prompts from rule results
        all_custom_prompts = []
        all_skip_types = set()
        all_force_types = set()

        for rule_results in batch_rule_results:
            if rule_results.get("custom_prompts"):
                all_custom_prompts.extend(rule_results["custom_prompts"])

            if rule_results.get("skip_recommendation_types"):
                all_skip_types.update(
                    rt.value for rt in rule_results["skip_recommendation_types"]
                )

            if rule_results.get("force_recommendation_types"):
                all_force_types.update(
                    rt.value for rt in rule_results["force_recommendation_types"]
                )

        # Add custom prompts to system prompt
        if all_custom_prompts:
            custom_prompt_section = "\n\nADDITIONAL CUSTOM RULES:\n" + "\n".join(
                all_custom_prompts
            )
            base_prompt += custom_prompt_section

        batch_instructions = f"""

BATCH ANALYSIS MODE: You are analyzing {batch_size} resources simultaneously.

IMPORTANT BATCH GUIDELINES:
1. Analyze each resource individually but consider relationships between them
2. Look for patterns across similar resources in the batch
3. Provide recommendations for each resource with their specific resource_id
4. Consider bulk optimization opportunities when applicable
5. Return results in the standard JSON format with an array of recommendations"""

        # Add rule-based restrictions
        if all_skip_types:
            batch_instructions += f"\n6. IMPORTANT: Do NOT recommend these types: {', '.join(all_skip_types)}"

        if all_force_types:
            batch_instructions += f"\n7. IMPORTANT: Always consider these recommendation types: {', '.join(all_force_types)}"

        batch_instructions += """

RESPONSE FORMAT:
{
  "recommendations": [
    {
      "resource_id": "specific-resource-id-1",
      "recommendation_type": "...",
      "description": "...",
      // ... other fields
    },
    {
      "resource_id": "specific-resource-id-2",
      "recommendation_type": "...",
      "description": "...",
      // ... other fields
    }
  ]
}

Ensure each recommendation includes the exact resource_id it applies to.
"""

        return base_prompt + batch_instructions
