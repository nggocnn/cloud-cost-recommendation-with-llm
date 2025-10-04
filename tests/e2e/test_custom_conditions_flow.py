#!/usr/bin/env python3
"""
Comprehensive test showing where and how custom conditions are applied throughout the system.
This test demonstrates the complete flow from rule evaluation to final recommendation output.
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


def show_where_custom_conditions_are_applied():
    """Demonstrate the complete flow of custom condition application"""

    print("=" * 60)
    print("CUSTOM CONDITIONS APPLICATION FLOW DEMONSTRATION")
    print("=" * 60)

    # Create test resource
    resource = Resource(
        resource_id="i-prod-critical-db",
        service=ServiceType.AWS.EC2,
        region="us-east-1",
        tags={
            "Environment": "production",
            "Criticality": "critical",
            "Application": "payment-database",
            "Team": "dba-team",
        },
        properties={"instance_type": "m5.2xlarge"},
    )

    # Create metrics showing high utilization
    metrics = Metrics(
        resource_id="i-prod-critical-db",
        timestamp=datetime.utcnow().isoformat() + "Z",
        period_days=30,
        cpu_utilization_p50=65.0,
        cpu_utilization_p95=85.0,  # High CPU
        memory_utilization_p50=70.0,
        memory_utilization_p95=88.0,  # High memory
    )

    # Define comprehensive custom rules
    rules = [
        # Rule 1: Production Safety Buffer
        ConditionalRule(
            name="production_safety_buffer",
            description="Production needs higher safety margins",
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
                    value=70,
                ),
            ],
            threshold_overrides={
                "cpu_low_threshold": 60.0,  # Raised from default 20%
                "memory_low_threshold": 50.0,  # Raised from default 30%
            },
            custom_prompt="Production database requires 40% CPU headroom for peak loads and failover scenarios.",
        ),
        # Rule 2: Critical Database Protection
        ConditionalRule(
            name="critical_database_protection",
            description="Critical databases avoid rightsizing",
            enabled=True,
            priority=90,
            logic="OR",
            conditions=[
                CustomCondition(
                    field="tag.Criticality",
                    operator=ConditionOperator.EQUALS,
                    value="critical",
                ),
                CustomCondition(
                    field="tag.Application",
                    operator=ConditionOperator.CONTAINS,
                    value="database",
                ),
            ],
            skip_recommendation_types=[RecommendationType.RIGHTSIZING],
            force_recommendation_types=[RecommendationType.PURCHASING_OPTION],
            risk_adjustment="increase",
            custom_prompt="Critical database system - focus only on Reserved Instance savings, no performance changes.",
        ),
    ]

    print("\n1. STEP 1: RULE EVALUATION (in _apply_custom_rules)")
    print("-" * 50)

    processor = RuleProcessor()
    rule_results = processor.apply_rules(rules, resource, metrics)

    print(f"✓ Applied rules: {len([r for r in rules if r.enabled])} total rules")
    print(
        f"✓ Matched rules: {len(rule_results.get('custom_prompts', []))} rules matched"
    )

    print(f"\nRule Results:")
    for key, value in rule_results.items():
        if value:  # Only show non-empty results
            if key == "custom_prompts":
                print(f"  {key}: {len(value)} prompts")
                for i, prompt in enumerate(value, 1):
                    print(f"    {i}. {prompt}")
            else:
                print(f"  {key}: {value}")

    print("\n2. STEP 2: THRESHOLD MODIFICATION (in _merge_thresholds)")
    print("-" * 55)

    base_thresholds = {
        "cpu_low_threshold": 20.0,
        "memory_low_threshold": 30.0,
        "iops_utilization_threshold": 80.0,
    }

    # Simulate threshold merging
    merged_thresholds = base_thresholds.copy()
    merged_thresholds.update(rule_results.get("threshold_overrides", {}))

    print(f"Base thresholds: {base_thresholds}")
    print(f"Rule overrides: {rule_results.get('threshold_overrides', {})}")
    print(f"✓ Merged thresholds: {merged_thresholds}")

    print(f"\nThreshold Changes:")
    for key in base_thresholds:
        if key in rule_results.get("threshold_overrides", {}):
            old_val = base_thresholds[key]
            new_val = merged_thresholds[key]
            print(f"  {key}: {old_val}% → {new_val}% (changed by rule)")
        else:
            print(f"  {key}: {base_thresholds[key]}% (unchanged)")

    print("\n3. STEP 3: LLM PROMPT ENHANCEMENT (in _generate_recommendations)")
    print("-" * 60)

    # Simulate prompt building
    base_user_prompt = """Analyze this EC2 instance for cost optimization:
Resource: i-prod-critical-db (m5.2xlarge)
CPU P95: 85.0%, Memory P95: 88.0%
Environment: production, Criticality: critical"""

    enhanced_prompt = base_user_prompt

    # Add custom prompts
    if rule_results.get("custom_prompts"):
        enhanced_prompt += "\n\nADDITIONAL REQUIREMENTS:\n"
        for custom_prompt in rule_results["custom_prompts"]:
            enhanced_prompt += f"- {custom_prompt}\n"

    # Add skip instructions
    if rule_results.get("skip_recommendation_types"):
        skip_types = [rt.value for rt in rule_results["skip_recommendation_types"]]
        enhanced_prompt += (
            f"\nIMPORTANT: Do NOT recommend these types: {', '.join(skip_types)}\n"
        )

    # Add force instructions
    if rule_results.get("force_recommendation_types"):
        force_types = [rt.value for rt in rule_results["force_recommendation_types"]]
        enhanced_prompt += f"\nIMPORTANT: Always consider these recommendation types: {', '.join(force_types)}\n"

    print("Base prompt:")
    print(f"  {base_user_prompt}")
    print(f"\n✓ Enhanced prompt with custom rules:")
    print("=" * 40)
    print(enhanced_prompt)
    print("=" * 40)

    print("\n4. STEP 4: POST-LLM RECOMMENDATION FILTERING")
    print("-" * 50)

    # Simulate LLM-generated recommendations
    llm_recommendations = [
        {
            "recommendation_type": "rightsizing",
            "resource_id": "i-prod-critical-db",
            "risk_level": "medium",
            "description": "Downsize to m5.xlarge based on utilization",
            "current_monthly_cost": 500.0,
            "estimated_monthly_cost": 250.0,
        },
        {
            "recommendation_type": "purchasing_option",
            "resource_id": "i-prod-critical-db",
            "risk_level": "low",
            "description": "Switch to 1-year Reserved Instance",
            "current_monthly_cost": 500.0,
            "estimated_monthly_cost": 350.0,
        },
    ]

    print(f"LLM generated {len(llm_recommendations)} recommendations:")
    for i, rec in enumerate(llm_recommendations, 1):
        print(
            f"  {i}. {rec['recommendation_type']} - {rec['description']} (Risk: {rec['risk_level']})"
        )

    # Apply filtering (simulate _filter_recommendations_by_rules)
    filtered_recommendations = []
    skip_types = [rt.value for rt in rule_results.get("skip_recommendation_types", [])]
    risk_adjustments = rule_results.get("risk_adjustments", [])

    print(f"\n✓ Applying post-LLM filtering:")
    print(f"  Skip types: {skip_types}")
    print(f"  Risk adjustments: {risk_adjustments}")

    for rec in llm_recommendations:
        rec_type = rec["recommendation_type"]

        # Check if should skip
        if rec_type in skip_types:
            print(
                f"  ❌ FILTERED OUT: {rec_type} - {rec['description']} (blocked by rule)"
            )
            continue

        # Apply risk adjustments
        if "increase" in risk_adjustments:
            original_risk = rec["risk_level"]
            if original_risk == "low":
                rec["risk_level"] = "medium"
            elif original_risk == "medium":
                rec["risk_level"] = "high"
            print(
                f"  🔺 RISK ADJUSTED: {rec_type} - {original_risk} → {rec['risk_level']}"
            )

        filtered_recommendations.append(rec)

    print(f"\n✓ Final recommendations after filtering:")
    for i, rec in enumerate(filtered_recommendations, 1):
        print(
            f"  {i}. {rec['recommendation_type']} - {rec['description']} (Risk: {rec['risk_level']})"
        )

    print(
        f"\nSUMMARY: {len(llm_recommendations)} → {len(filtered_recommendations)} recommendations"
    )

    print("\n" + "=" * 60)
    print("COMPLETE CUSTOM CONDITIONS FLOW DEMONSTRATED")
    print("=" * 60)

    return {
        "rule_results": rule_results,
        "merged_thresholds": merged_thresholds,
        "enhanced_prompt": enhanced_prompt,
        "filtered_recommendations": filtered_recommendations,
    }


if __name__ == "__main__":
    show_where_custom_conditions_are_applied()
