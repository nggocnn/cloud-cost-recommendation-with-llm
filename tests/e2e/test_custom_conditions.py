#!/usr/bin/env python3
"""
Test script to demonstrate how custom conditions are applied in practice.
This script shows the complete workflow from custom condition evaluation to
recommendation filtering.
"""

import sys
import os

# Add the parent directory to Python path to import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from datetime import datetime

from llm_cost_recommendation.services.conditions import (
    ConditionEvaluator,
    RuleProcessor,
)
from llm_cost_recommendation.models import (
    Resource,
    ServiceType,
    CustomCondition,
    ConditionalRule,
    ConditionOperator,
    RecommendationType,
    Metrics,
    BillingData,
)


async def demonstrate_custom_conditions():
    """Demonstrate how custom conditions work step by step"""

    print("=== Custom Conditions Demonstration ===\n")

    # 1. Create test resources with different characteristics
    production_resource = Resource(
        resource_id="i-prod-web-server",
        service=ServiceType.AWS.EC2,
        region="us-east-1",
        tags={
            "Environment": "production",
            "Criticality": "high",
            "Application": "payment-gateway",
            "Team": "payment-team",
        },
        properties={"instance_type": "t3.large"},
    )

    dev_resource = Resource(
        resource_id="i-dev-test-server",
        service=ServiceType.AWS.EC2,
        region="us-east-1",
        tags={"Environment": "dev", "Team": "dev-team"},
        properties={"instance_type": "t3.medium"},
    )

    # 2. Create metrics for resources
    prod_metrics = Metrics(
        resource_id="i-prod-web-server",
        timestamp=datetime.utcnow().isoformat() + "Z",
        period_days=30,
        cpu_utilization_p50=45.0,
        cpu_utilization_p95=75.0,  # High CPU usage
        memory_utilization_p50=60.0,
    )

    dev_metrics = Metrics(
        resource_id="i-dev-test-server",
        timestamp=datetime.utcnow().isoformat() + "Z",
        period_days=30,
        cpu_utilization_p50=8.0,
        cpu_utilization_p95=15.0,  # Low CPU usage
        memory_utilization_p50=20.0,
    )

    # 3. Define custom conditional rules

    # Rule 1: Production CPU Buffer - Higher thresholds for production
    prod_buffer_rule = ConditionalRule(
        name="production_cpu_buffer",
        description="Production instances need higher CPU buffer",
        enabled=True,
        priority=100,
        logic="AND",
        conditions=[
            CustomCondition(
                field="tag.Environment",
                operator=ConditionOperator.EQUALS,
                value="production",
            ),
            CustomCondition(
                field="cpu_utilization_p95",
                operator=ConditionOperator.GREATER_THAN,
                value=60,
            ),
        ],
        threshold_overrides={
            "cpu_low_threshold": 50.0  # Higher threshold for production
        },
        custom_prompt="For production workloads, maintain at least 50% CPU headroom.",
    )

    # Rule 2: Critical Applications - No rightsizing
    critical_rule = ConditionalRule(
        name="critical_no_rightsizing",
        description="Critical applications avoid rightsizing",
        enabled=True,
        priority=90,
        logic="OR",
        conditions=[
            CustomCondition(
                field="tag.Criticality", operator=ConditionOperator.EQUALS, value="high"
            ),
            CustomCondition(
                field="tag.Application",
                operator=ConditionOperator.CONTAINS,
                value="payment",
            ),
        ],
        skip_recommendation_types=[RecommendationType.RIGHTSIZING],
        risk_adjustment="increase",
        custom_prompt="Critical system - focus on purchasing optimizations only.",
    )

    # Rule 3: Development Aggressive Optimization
    dev_rule = ConditionalRule(
        name="dev_aggressive_optimization",
        description="Development can be optimized aggressively",
        enabled=True,
        priority=70,
        logic="OR",
        conditions=[
            CustomCondition(
                field="tag.Environment",
                operator=ConditionOperator.IN,
                value=["dev", "test", "staging"],
            )
        ],
        threshold_overrides={"cpu_low_threshold": 10.0, "memory_low_threshold": 15.0},
        force_recommendation_types=[RecommendationType.RIGHTSIZING],
        risk_adjustment="decrease",
        custom_prompt="Development environment - apply aggressive optimization.",
    )

    # 4. Test rule evaluation
    print("1. INDIVIDUAL CONDITION TESTING")
    print("-" * 40)

    evaluator = ConditionEvaluator()

    # Test environment condition
    env_condition = CustomCondition(
        field="tag.Environment", operator=ConditionOperator.EQUALS, value="production"
    )

    prod_env_result = evaluator.evaluate_condition(env_condition, production_resource)
    dev_env_result = evaluator.evaluate_condition(env_condition, dev_resource)

    print(f"Environment='production' condition:")
    print(f"  Production resource: {prod_env_result}")
    print(f"  Dev resource: {dev_env_result}")

    # Test CPU condition with metrics
    cpu_condition = CustomCondition(
        field="cpu_utilization_p95", operator=ConditionOperator.GREATER_THAN, value=60
    )

    prod_cpu_result = evaluator.evaluate_condition(
        cpu_condition, production_resource, prod_metrics
    )
    dev_cpu_result = evaluator.evaluate_condition(
        cpu_condition, dev_resource, dev_metrics
    )

    print(f"\\nCPU > 60% condition:")
    print(
        f"  Production resource: {prod_cpu_result} (CPU: {prod_metrics.cpu_utilization_p95}%)"
    )
    print(f"  Dev resource: {dev_cpu_result} (CPU: {dev_metrics.cpu_utilization_p95}%)")

    # 5. Test complete rules
    print("\\n2. COMPLETE RULE EVALUATION")
    print("-" * 40)

    processor = RuleProcessor()
    rules = [prod_buffer_rule, critical_rule, dev_rule]

    # Test production resource
    prod_results = processor.apply_rules(rules, production_resource, prod_metrics)
    print(f"\\nProduction resource rule results:")
    print(f"  Threshold overrides: {prod_results['threshold_overrides']}")
    print(f"  Skip types: {[t for t in prod_results['skip_recommendation_types']]}")
    print(f"  Force types: {[t for t in prod_results['force_recommendation_types']]}")
    print(f"  Custom prompts: {prod_results['custom_prompts']}")
    print(f"  Risk adjustments: {prod_results['risk_adjustments']}")

    # Test dev resource
    dev_results = processor.apply_rules(rules, dev_resource, dev_metrics)
    print(f"\\nDevelopment resource rule results:")
    print(f"  Threshold overrides: {dev_results['threshold_overrides']}")
    print(f"  Skip types: {[t for t in dev_results['skip_recommendation_types']]}")
    print(f"  Force types: {[t for t in dev_results['force_recommendation_types']]}")
    print(f"  Custom prompts: {dev_results['custom_prompts']}")
    print(f"  Risk adjustments: {dev_results['risk_adjustments']}")

    # 6. Demonstrate how these results affect recommendations
    print("\\n3. RECOMMENDATION FILTERING SIMULATION")
    print("-" * 45)

    # Simulate some sample recommendations
    sample_recommendations = [
        {
            "recommendation_type": "rightsizing",
            "resource_id": "i-prod-web-server",
            "risk_level": "medium",
            "description": "Downsize to t3.medium",
        },
        {
            "recommendation_type": "purchasing_option",
            "resource_id": "i-prod-web-server",
            "risk_level": "low",
            "description": "Switch to Reserved Instance",
        },
        {
            "recommendation_type": "rightsizing",
            "resource_id": "i-dev-test-server",
            "risk_level": "medium",
            "description": "Downsize to t3.small",
        },
    ]

    print("Original recommendations:")
    for i, rec in enumerate(sample_recommendations, 1):
        print(
            f"  {i}. {rec['recommendation_type']} - {rec['description']} (Risk: {rec['risk_level']})"
        )

    # Apply production rules filtering
    prod_filtered = []
    for rec in sample_recommendations:
        if rec["resource_id"] == production_resource.resource_id:
            # Skip rightsizing for production (critical rule matched)
            if rec["recommendation_type"] in [
                t for t in prod_results["skip_recommendation_types"]
            ]:
                print(
                    f"\\n❌ Filtered out: {rec['recommendation_type']} (blocked by critical rule)"
                )
                continue

            # Adjust risk level
            if "increase" in prod_results["risk_adjustments"]:
                if rec["risk_level"] == "low":
                    rec["risk_level"] = "medium"
                elif rec["risk_level"] == "medium":
                    rec["risk_level"] = "high"
                print(
                    f"🔺 Risk adjusted: {rec['recommendation_type']} risk increased to {rec['risk_level']}"
                )

            prod_filtered.append(rec)

    # Apply dev rules filtering
    dev_filtered = []
    for rec in sample_recommendations:
        if rec["resource_id"] == dev_resource.resource_id:
            # Force rightsizing for dev environments
            if rec["recommendation_type"] in [
                t for t in dev_results["force_recommendation_types"]
            ]:
                print(
                    f"\\n✅ Prioritized: {rec['recommendation_type']} (forced by dev rule)"
                )

            # Adjust risk level
            if "decrease" in dev_results["risk_adjustments"]:
                if rec["risk_level"] == "high":
                    rec["risk_level"] = "medium"
                elif rec["risk_level"] == "medium":
                    rec["risk_level"] = "low"
                print(
                    f"🔻 Risk adjusted: {rec['recommendation_type']} risk decreased to {rec['risk_level']}"
                )

            dev_filtered.append(rec)

    print(f"\\nFinal recommendations after custom rules:")
    all_filtered = prod_filtered + dev_filtered
    for i, rec in enumerate(all_filtered, 1):
        print(
            f"  {i}. {rec['recommendation_type']} - {rec['description']} (Risk: {rec['risk_level']})"
        )

    # 7. Show how thresholds are modified
    print("\\n4. THRESHOLD MODIFICATIONS")
    print("-" * 30)

    base_thresholds = {"cpu_low_threshold": 20.0, "memory_low_threshold": 30.0}

    print(f"Base thresholds: {base_thresholds}")

    # Production thresholds
    prod_thresholds = base_thresholds.copy()
    prod_thresholds.update(prod_results["threshold_overrides"])
    print(f"Production thresholds: {prod_thresholds}")
    print(
        f"  → CPU threshold raised from 20% to {prod_thresholds['cpu_low_threshold']}% for production safety"
    )

    # Dev thresholds
    dev_thresholds = base_thresholds.copy()
    dev_thresholds.update(dev_results["threshold_overrides"])
    print(f"Development thresholds: {dev_thresholds}")
    print(
        f"  → CPU threshold lowered to {dev_thresholds['cpu_low_threshold']}% for aggressive optimization"
    )
    print(
        f"  → Memory threshold lowered to {dev_thresholds['memory_low_threshold']}% for aggressive optimization"
    )

    print("\\n=== Custom Conditions Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(demonstrate_custom_conditions())
