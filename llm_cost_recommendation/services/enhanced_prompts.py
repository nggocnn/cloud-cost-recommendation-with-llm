"""
Enhanced prompt templates with strict evidence requirements to prevent fabricated recommendations.
"""

import json
from typing import Dict, List, Any, Optional, Tuple

class EnhancedPromptTemplates:
    """Enhanced prompt templates that enforce evidence-based recommendations"""

    EVIDENCE_BASED_SYSTEM_PROMPT = """You are an AWS cost optimization expert that ONLY makes recommendations based on actual data provided.

CRITICAL EVIDENCE RULES:
1. NEVER fabricate performance metrics (CPU %, memory %, network %, request %)
2. NEVER fabricate traffic analysis percentages without actual CloudWatch data
3. NEVER fabricate specific cost numbers without actual billing data
4. NEVER make claims about traffic patterns without actual metrics data
5. ALWAYS state when data is missing or estimated
6. ONLY make recommendations when sufficient evidence exists

FORBIDDEN PHRASES - DO NOT USE:
- "Traffic analysis shows X% of requests..." (without actual data)
- "Performance data indicates..." (without actual metrics)
- Any specific percentage without source data
- "Based on observed traffic patterns..." (without actual observations)

If metrics data is empty or missing:
- State "No performance metrics available"
- Do not claim specific percentages or traffic patterns
- Base recommendations only on available inventory data

If billing data is missing:
- State "No actual cost data available"
- Use terms like "estimated" for any cost numbers
- Lower confidence scores appropriately

Your analysis must be evidence-based and transparent about data limitations. When in doubt, reject the recommendation rather than fabricate evidence."""

    INSUFFICIENT_DATA_RESPONSE_TEMPLATE = """
{
    "recommendations": [],
    "analysis_summary": {
        "total_resources_analyzed": {resource_count},
        "resources_with_sufficient_data": 0,
        "resources_with_limited_data": {resource_count},
        "data_quality_issues": [
            "Insufficient data for reliable cost optimization recommendations"
        ],
        "recommended_data_improvements": [
            "Collect CloudWatch metrics for performance analysis",
            "Obtain actual AWS billing/cost data", 
            "Gather usage patterns and traffic data",
            "Enable detailed monitoring and logging"
        ]
    }
}"""

    def create_evidence_enhanced_prompt(self, context_data: dict, resource_id: str, service: str) -> str:
        """Create a prompt that emphasizes evidence-based analysis"""
        
        # Assess available data
        has_inventory = bool(context_data.get("resource"))
        has_billing = bool(context_data.get("billing"))
        has_metrics = bool(context_data.get("metrics"))
        
        data_assessment = []
        if has_inventory:
            data_assessment.append("Resource inventory data available")
        else:
            data_assessment.append("Resource inventory data MISSING")
            
        if has_billing:
            data_assessment.append("Billing data available (may be estimated)")
        else:
            data_assessment.append("Billing data MISSING")
            
        if has_metrics:
            data_assessment.append("Performance metrics available")
        else:
            data_assessment.append("Performance metrics MISSING")
        
        prompt_parts = [
            f"Analyze the following {service} resource for cost optimization opportunities:",
            "",
            f"RESOURCE ID: {resource_id}",
            "",
            "DATA AVAILABILITY VERIFICATION:",
            f"Inventory Data: {'Available' if has_inventory else 'MISSING'}",
            f"Billing Data: {'Available' if has_billing else 'MISSING - All costs must be ESTIMATED'}",
            f"Performance Metrics: {'Available' if has_metrics else 'MISSING - NO traffic/performance analysis possible'}",
            "",
            "ABSOLUTE CONSTRAINTS:",
            "- Do NOT fabricate any specific percentages without actual data",
            "- Do NOT claim traffic analysis without CloudWatch metrics",
            "- Do NOT invent performance patterns or usage statistics",
            "- State 'No metrics data available' if metrics are missing",
            "- Mark all costs as 'ESTIMATED' if no billing data",
            "",
        ]

        if has_inventory:
            prompt_parts.extend([
                "RESOURCE CONFIGURATION:",
                json.dumps(context_data.get("resource", {}), indent=2),
                "",
            ])

        if has_billing:
            prompt_parts.extend([
                "BILLING DATA:",
                json.dumps(context_data["billing"], indent=2),
                "Note: Verify if billing data is actual or estimated",
                "",
            ])
        else:
            prompt_parts.extend([
                "BILLING DATA: NOT AVAILABLE",
                "Cannot provide accurate cost calculations without billing data",
                "",
            ])

        if has_metrics:
            prompt_parts.extend([
                "PERFORMANCE METRICS:",
                json.dumps(context_data["metrics"], indent=2),
                "",
            ])
        else:
            prompt_parts.extend([
                "PERFORMANCE METRICS: NOT AVAILABLE", 
                "Cannot assess utilization or performance impact without metrics",
                "",
            ])

        prompt_parts.extend([
            "ANALYSIS REQUIREMENTS:",
            "1. Assess data quality and completeness",
            "2. Only make recommendations supported by available data",
            "3. Clearly indicate limitations and uncertainties",
            "4. Suggest additional data needed for better recommendations",
            "5. Include appropriate confidence scores based on data availability",
            "",
            "REQUIRED JSON OUTPUT FORMAT:",
            "You MUST provide your response in this exact JSON structure:",
            "CRITICAL: Do NOT include any comments (// or /* */) in your JSON response",
            "",
            "{",
            '  "recommendations": [',
            "    {",
            '      "recommendation_type": "rightsizing|purchasing_option|idle_resource",',
            '      "current_config": {},',
            '      "recommended_config": {},',
            '      "rationale": "Brief explanation - state data limitations clearly",',
            '      "evidence": {',
            '        "metrics_analysis": "State data availability - no fabricated percentages",',
            '        "cost_breakdown": "Mark as estimated if no billing data",',
            '        "performance_impact": "State data limitations clearly"',
            "      },",
            '      "confidence_score": 0.7,',
            '      "current_monthly_cost": 0.0,',
            '      "estimated_monthly_cost": 0.0,',
            '      "estimated_monthly_savings": 0.0,',
            '      "risk_level": "low|medium|high",',
            '      "impact_description": "Brief impact description",',
            '      "implementation_steps": ["Step 1", "Step 2"],',
            '      "rollback_plan": "How to revert if needed"',
            "    }",
            "  ]",
            "}",
            "",
            "JSON FORMATTING RULES:",
            "- NO comments anywhere in the JSON response",
            "- confidence_score: Lower values (0.3-0.6) if missing critical data",
            "- current_monthly_cost: Use 0 if unknown or unavailable",
            "- estimated_monthly_cost: Use 0 if unknown or unavailable",
            "",
            "CRITICAL: If you cannot make reliable recommendations due to insufficient data,",
            "return an empty recommendations array: {\"recommendations\": []}",
            "",
            "Remember: NO fabricated metrics, percentages, or performance claims!"
        ])

        return "\n".join(prompt_parts)
        
    def should_skip_analysis(self, context_data: dict) -> tuple[bool, str]:
        """Determine if analysis should be skipped due to insufficient data"""
        
        has_inventory = bool(context_data.get("resource"))
        has_any_cost_data = bool(context_data.get("billing"))
        has_metrics = bool(context_data.get("metrics"))
        
        # Minimum requirement: at least inventory data
        if not has_inventory:
            return True, "No resource inventory data available"
            
        # For cost optimization, we need at least some cost-related data or metrics
        if not has_any_cost_data and not has_metrics:
            return True, "Insufficient data for cost optimization analysis - no billing or performance data"
            
        return False, "Sufficient data available for analysis"
