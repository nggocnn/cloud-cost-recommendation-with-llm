#!/usr/bin/env python3
"""
Test to demonstrate the dual approach: LLM prompt guidance + post-filtering
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_cost_recommendation.services.conditions import RuleProcessor
from llm_cost_recommendation.models import (
    Resource,
    ServiceType,
    CustomCondition,
    ConditionalRule,
    ConditionOperator,
    RecommendationType,
    Metrics,
)


def demonstrate_dual_approach():
    """Show how skip/force work in both prompt and post-filtering"""

    print("DUAL APPROACH: LLM Prompt Guidance + Post-Filtering Safety Net")
    print("=" * 65)

    # Create test resource
    resource = Resource(
        resource_id="i-critical-payment",
        service=ServiceType.AWS.EC2,
        region="us-east-1",
        tags={
            "Environment": "production",
            "Criticality": "critical",
            "Application": "payment-processor",
        },
        properties={"instance_type": "m5.large"},
    )

    # Rule that skips rightsizing and forces purchasing options
    rule = ConditionalRule(
        name="critical_payment_protection",
        description="Critical payment systems need special handling",
        enabled=True,
        priority=100,
        conditions=[
            CustomCondition(
                field="tag.Criticality",
                operator=ConditionOperator.EQUALS,
                value="critical",
            )
        ],
        skip_recommendation_types=[
            RecommendationType.RIGHTSIZING,
            RecommendationType.IDLE_RESOURCE,
        ],
        force_recommendation_types=[RecommendationType.PURCHASING_OPTION],
        custom_prompt="Critical payment system - maximize availability and compliance.",
    )

    # Apply rules
    processor = RuleProcessor()
    rule_results = processor.apply_rules([rule], resource)

    print("1. RULE RESULTS:")
    print(
        f"   Skip types: {[rt.value for rt in rule_results['skip_recommendation_types']]}"
    )
    print(
        f"   Force types: {[rt.value for rt in rule_results['force_recommendation_types']]}"
    )

    print("\n2. LLM PROMPT ENHANCEMENT:")
    print("   Base prompt: 'Analyze this EC2 instance for optimization...'")

    # Simulate prompt building (from base.py lines 206-216)
    enhanced_prompt = "Analyze this EC2 instance for optimization..."

    if rule_results.get("custom_prompts"):
        enhanced_prompt += "\n\nADDITIONAL REQUIREMENTS:\n"
        for custom_prompt in rule_results["custom_prompts"]:
            enhanced_prompt += f"- {custom_prompt}\n"

    if rule_results.get("skip_recommendation_types"):
        skip_types = [rt.value for rt in rule_results["skip_recommendation_types"]]
        enhanced_prompt += (
            f"\nIMPORTANT: Do NOT recommend these types: {', '.join(skip_types)}\n"
        )

    if rule_results.get("force_recommendation_types"):
        force_types = [rt.value for rt in rule_results["force_recommendation_types"]]
        enhanced_prompt += f"\nIMPORTANT: Always consider these recommendation types: {', '.join(force_types)}\n"

    print("   ✓ Enhanced prompt sent to LLM:")
    print("   " + "─" * 50)
    print("   " + enhanced_prompt.replace("\n", "\n   "))
    print("   " + "─" * 50)

    print("\n3. SIMULATED LLM RESPONSES:")

    # Scenario A: LLM follows instructions perfectly
    print("\n   Scenario A: LLM follows instructions (ideal case)")
    good_llm_response = [
        {
            "recommendation_type": "purchasing_option",
            "description": "Switch to Reserved Instance for 30% savings",
            "risk_level": "low",
        }
    ]
    print(f"   LLM Response: {len(good_llm_response)} recommendations")
    for rec in good_llm_response:
        print(f"     - {rec['recommendation_type']}: {rec['description']}")

    # Scenario B: LLM ignores instructions (happens sometimes)
    print("\n   Scenario B: LLM ignores instructions (real-world case)")
    bad_llm_response = [
        {
            "recommendation_type": "rightsizing",
            "description": "Downsize to t3.medium to reduce costs",
            "risk_level": "medium",
        },
        {
            "recommendation_type": "purchasing_option",
            "description": "Switch to Reserved Instance for 30% savings",
            "risk_level": "low",
        },
    ]
    print(f"   LLM Response: {len(bad_llm_response)} recommendations")
    for rec in bad_llm_response:
        print(f"     - {rec['recommendation_type']}: {rec['description']}")

    print("\n4. POST-LLM FILTERING (Safety Net):")

    # Apply post-filtering to both scenarios
    def apply_post_filtering(recommendations, rule_results):
        filtered = []
        skip_types = [
            rt.value for rt in rule_results.get("skip_recommendation_types", [])
        ]

        for rec in recommendations:
            if rec["recommendation_type"] in skip_types:
                print(
                    f"     ❌ BLOCKED: {rec['recommendation_type']} - {rec['description']}"
                )
                continue
            else:
                print(
                    f"     ✅ ALLOWED: {rec['recommendation_type']} - {rec['description']}"
                )
                filtered.append(rec)
        return filtered

    print("\n   Scenario A filtering (LLM followed rules):")
    filtered_good = apply_post_filtering(good_llm_response, rule_results)
    print(
        f"     Result: {len(good_llm_response)} → {len(filtered_good)} recommendations"
    )

    print("\n   Scenario B filtering (LLM ignored rules):")
    filtered_bad = apply_post_filtering(bad_llm_response, rule_results)
    print(f"     Result: {len(bad_llm_response)} → {len(filtered_bad)} recommendations")

    print("\n5. WHY DUAL APPROACH?")
    print("   ✓ LLM Prompt: Primary guidance - tells LLM what to do")
    print("   ✓ Post-Filter: Safety net - catches LLM mistakes")
    print("   ✓ Reliability: Even if LLM ignores rules, filtering ensures compliance")
    print("   ✓ Audit Trail: Logging shows what was blocked and why")

    print(f"\n{'=' * 65}")
    print("CONCLUSION: Skip/Force are ALREADY in prompt + post-filtering!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    demonstrate_dual_approach()
