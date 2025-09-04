"""
Condition evaluation service for custom agent rules.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import (
    CustomCondition,
    ConditionalRule,
    ConditionOperator,
    Resource,
    Metrics,
    BillingData,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConditionEvaluator:
    """Evaluates custom conditions against resource data"""

    def __init__(self):
        self.operator_map = {
            ConditionOperator.EQUALS: self._equals,
            ConditionOperator.NOT_EQUALS: self._not_equals,
            ConditionOperator.GREATER_THAN: self._greater_than,
            ConditionOperator.LESS_THAN: self._less_than,
            ConditionOperator.GREATER_EQUAL: self._greater_equal,
            ConditionOperator.LESS_EQUAL: self._less_equal,
            ConditionOperator.CONTAINS: self._contains,
            ConditionOperator.NOT_CONTAINS: self._not_contains,
            ConditionOperator.IN: self._in,
            ConditionOperator.NOT_IN: self._not_in,
            ConditionOperator.REGEX: self._regex,
            ConditionOperator.EXISTS: self._exists,
            ConditionOperator.NOT_EXISTS: self._not_exists,
        }

    def evaluate_rule(
        self,
        rule: ConditionalRule,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
        computed_cost: Optional[float] = None,
    ) -> bool:
        """Evaluate a complete rule against resource data"""
        if not rule.enabled:
            return False

        results = []
        for condition in rule.conditions:
            result = self.evaluate_condition(
                condition, resource, metrics, billing_data, computed_cost
            )
            results.append(result)

            logger.debug(
                "Evaluating condition",
                rule_name=rule.name,
                resource_id=resource.resource_id,
                condition_field=condition.field,
                condition_operator=condition.operator,
                condition_value=condition.value,
                result=result,
            )

        # Apply logic (AND/OR)
        if rule.logic == "AND":
            final_result = all(results)
        else:  # OR
            final_result = any(results)

        logger.debug(
            "Rule evaluated",
            rule_name=rule.name,
            logic=rule.logic,
            conditions_count=len(rule.conditions),
            result=final_result,
            resource_id=resource.resource_id,
        )

        return final_result

    def evaluate_condition(
        self,
        condition: CustomCondition,
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
        computed_cost: Optional[float] = None,
    ) -> bool:
        """Evaluate a single condition"""
        try:
            field_value = self._get_field_value(
                condition.field, resource, metrics, billing_data, computed_cost
            )

            operator_func = self.operator_map.get(condition.operator)
            if not operator_func:
                logger.warning(
                    "Unknown operator",
                    operator=condition.operator,
                    field=condition.field,
                )
                return False

            return operator_func(field_value, condition.value)

        except Exception as e:
            logger.error(
                "Error evaluating condition",
                field=condition.field,
                operator=condition.operator.value,
                error=str(e),
                resource_id=resource.resource_id,
            )
            return False

    def _get_field_value(
        self,
        field: str,
        resource: Resource,
        metrics: Optional[Metrics],
        billing_data: Optional[List[BillingData]],
        computed_cost: Optional[float],
    ) -> Any:
        """Extract field value from resource data"""

        # Handle tag fields (tag.key_name)
        if field.startswith("tag."):
            tag_key = field[4:]  # Remove "tag." prefix
            return resource.tags.get(tag_key)

        # Handle property fields (property.key_name)
        if field.startswith("property."):
            prop_key = field[9:]  # Remove "property." prefix
            return resource.properties.get(prop_key)

        # Handle extension fields (extension.key_name)
        if field.startswith("extension."):
            ext_key = field[10:]  # Remove "extension." prefix
            return resource.extensions.get(ext_key)

        # Handle resource fields
        resource_fields = {
            "resource_id": resource.resource_id,
            "service": resource.service.value,
            "region": resource.region,
            "availability_zone": resource.availability_zone,
        }

        if field in resource_fields:
            return resource_fields[field]

        # Handle computed cost fields
        if field in ["monthly_cost", "daily_cost"] and computed_cost is not None:
            if field == "monthly_cost":
                return computed_cost
            elif field == "daily_cost":
                return computed_cost / 30.0

        # Handle time-based fields
        if field == "created_at":
            return resource.created_at
        elif field == "age_days":
            if resource.created_at:
                return (datetime.utcnow() - resource.created_at).days
            return None

        # Handle metrics fields
        if metrics and hasattr(metrics, field):
            return getattr(metrics, field)

        # Handle custom metrics
        if metrics and field in metrics.metrics:
            return metrics.metrics[field]

        # Handle boolean fields
        if field == "is_idle" and metrics:
            return metrics.is_idle

        # Field not found
        logger.debug("Field not found", field=field, resource_id=resource.resource_id)
        return None

    # Operator implementations
    def _equals(self, field_value: Any, condition_value: Any) -> bool:
        return field_value == condition_value

    def _not_equals(self, field_value: Any, condition_value: Any) -> bool:
        return field_value != condition_value

    def _greater_than(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return False
        try:
            return float(field_value) > float(condition_value)
        except (ValueError, TypeError):
            return False

    def _less_than(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return False
        try:
            return float(field_value) < float(condition_value)
        except (ValueError, TypeError):
            return False

    def _greater_equal(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return False
        try:
            return float(field_value) >= float(condition_value)
        except (ValueError, TypeError):
            return False

    def _less_equal(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return False
        try:
            return float(field_value) <= float(condition_value)
        except (ValueError, TypeError):
            return False

    def _contains(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return False
        return str(condition_value) in str(field_value)

    def _not_contains(self, field_value: Any, condition_value: Any) -> bool:
        if field_value is None:
            return True
        return str(condition_value) not in str(field_value)

    def _in(self, field_value: Any, condition_value: List[Any]) -> bool:
        if field_value is None:
            return False
        if not isinstance(condition_value, list):
            return False
        return field_value in condition_value

    def _not_in(self, field_value: Any, condition_value: List[Any]) -> bool:
        if field_value is None:
            return True
        if not isinstance(condition_value, list):
            return True
        return field_value not in condition_value

    def _regex(self, field_value: Any, condition_value: str) -> bool:
        if field_value is None:
            return False
        try:
            pattern = re.compile(str(condition_value))
            return bool(pattern.search(str(field_value)))
        except re.error:
            logger.warning(
                "Invalid regex pattern",
                pattern=condition_value,
                field_value=field_value,
            )
            return False

    def _exists(self, field_value: Any, condition_value: Any) -> bool:
        return field_value is not None

    def _not_exists(self, field_value: Any, condition_value: Any) -> bool:
        return field_value is None


class RuleProcessor:
    """Processes and applies conditional rules to agent configurations"""

    def __init__(self):
        self.evaluator = ConditionEvaluator()

    def apply_rules(
        self,
        rules: List[ConditionalRule],
        resource: Resource,
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None,
        computed_cost: Optional[float] = None,
        base_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Apply rules and return configuration overrides"""

        # Sort rules by priority (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        applied_rules = []
        result = {
            "threshold_overrides": {},
            "skip_recommendation_types": [],
            "force_recommendation_types": [],
            "custom_prompts": [],
            "risk_adjustments": [],
            "actions": {},
        }

        for rule in sorted_rules:
            if self.evaluator.evaluate_rule(
                rule, resource, metrics, billing_data, computed_cost
            ):
                applied_rules.append(rule.name)

                # Apply threshold overrides
                result["threshold_overrides"].update(rule.threshold_overrides)

                # Accumulate recommendation type filters
                result["skip_recommendation_types"].extend(
                    rule.skip_recommendation_types
                )
                result["force_recommendation_types"].extend(
                    rule.force_recommendation_types
                )

                # Collect custom prompts
                if rule.custom_prompt:
                    result["custom_prompts"].append(rule.custom_prompt)

                # Collect risk adjustments
                if rule.risk_adjustment:
                    result["risk_adjustments"].append(rule.risk_adjustment)

                # Merge actions
                result["actions"].update(rule.actions)

        # Remove duplicates
        result["skip_recommendation_types"] = list(
            set(result["skip_recommendation_types"])
        )
        result["force_recommendation_types"] = list(
            set(result["force_recommendation_types"])
        )

        logger.debug(
            "Rules processing completed",
            resource_id=resource.resource_id,
            total_rules=len(rules),
            applied_rules=applied_rules,
            threshold_overrides=result["threshold_overrides"],
            skip_types=result["skip_recommendation_types"],
            force_types=result["force_recommendation_types"],
        )

        return result
