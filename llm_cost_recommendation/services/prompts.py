"""
Unified prompt system for LLM cost optimization recommendations.

This module consolidates all prompt templates and provides evidence-based prompting
with strict anti-fabrication rules for accurate AWS cost optimization analysis.
"""

from typing import Dict, Any, List, Optional
import json


class PromptTemplates:
    """Unified prompt templates for AWS cost optimization recommendations."""
    
    # Main system prompt for cost optimization recommendations
    SYSTEM_PROMPT = """You are an AWS cost optimization expert that ONLY makes recommendations based on actual data provided.

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

    @property
    def COMPLETE_SYSTEM_PROMPT(self) -> str:
        """Get the complete system prompt with JSON structure."""
        return self._get_system_prompt()
    
    def create_user_prompt(
        self, 
        resource_data: Dict[str, Any], 
        resource_type: str, 
        service_type: str
    ) -> str:
        """
        Create a user prompt that emphasizes factual analysis
        and prevents fabrication of metrics or claims.
        
        Args:
            resource_data: Complete resource information including metrics
            resource_type: Type of AWS resource (instance, volume, etc.)
            service_type: AWS service (EC2, EBS, S3, etc.)
            
        Returns:
            User prompt string with strict evidence requirements
        """
        base_prompt = self._get_system_prompt()
        
        # Add resource-specific context
        resource_context = self._build_resource_context(resource_data, resource_type, service_type)
        
        # Add service-specific guidance
        service_guidance = self._get_service_specific_guidance(service_type)
        
        # Combine all components
        full_prompt = f"{base_prompt}\n\n{resource_context}\n\n{service_guidance}"
        
        return full_prompt
    
    def _get_system_prompt(self) -> str:
        """Get the main system prompt with strict evidence requirements."""
        return f"""{self.SYSTEM_PROMPT}

ANALYSIS PRINCIPLES:
- Base ALL recommendations on provided evidence (metrics, configuration, usage patterns)
- Identify data gaps and recommend additional monitoring when needed
- Consider cost optimization opportunities only when supported by actual usage data
- Factor in operational requirements and potential business impact
- Provide specific, actionable steps rather than generic advice

RESPONSE FORMAT:
Always respond with valid JSON matching this exact structure:
{{
    "resource_id": "resource-identifier",
    "service": "aws-service-name", 
    "resource_type": "resource-type",
    "recommendation_type": "rightsizing|purchasing_option|lifecycle|topology|storage_class|idle_resource|cost_analysis|general_optimization",
    "current_config": {{
        "instance_type": "current type",
        "storage_size": "current size", 
        "other_relevant_settings": "current values"
    }},
    "recommended_config": {{
        "instance_type": "recommended type",
        "storage_size": "recommended size",
        "other_relevant_settings": "recommended values"
    }},
    "current_monthly_cost": 123.45,
    "estimated_monthly_cost": 98.76,
    "estimated_monthly_savings": 24.69,
    "confidence_score": 0.85,
    "risk_level": "low|medium|high",
    "impact_description": "Detailed explanation of the recommendation and its business impact",
    "rationale": "Technical reasoning behind this recommendation",
    "evidence": {{
        "metrics_analysis": "specific data supporting this recommendation",
        "cost_breakdown": "detailed cost analysis", 
        "performance_impact": "expected performance changes"
    }},
    "implementation_steps": [
        "Step 1: Specific action to take",
        "Step 2: Next action to take"
    ],
    "prerequisites": [
        "Prerequisite 1: Required condition or preparation"
    ],
    "rollback_plan": "Detailed plan to revert changes if needed",
    "business_hours_impact": false,
    "downtime_required": false,
    "sla_impact": "Expected impact on SLA, or null if no impact",
    "warning": "any concerns about recommendation quality (optional)"
}}

CRITICAL JSON REQUIREMENTS:
- Respond ONLY with valid JSON - no extra text before or after
- NO COMMENTS: Do not include // or /* */ comments anywhere in the JSON
- ALL numeric values must be actual calculated numbers (e.g., 123.45, not "insufficient data")
- current_config and recommended_config must be non-empty objects, never null
- evidence must be a non-empty object with relevant analysis data, never null
- prerequisites can be an empty array if no prerequisites exist
- Calculate all costs yourself and provide final dollar amounts
- estimated_monthly_savings = current_monthly_cost - estimated_monthly_cost
- business_hours_impact and downtime_required must be boolean values (true/false)
- sla_impact should be a string describing SLA impact or null if no impact
- All fields are mandatory - include every single field shown above
- Use exact field names (especially "impact_description", not "impact")
- current_monthly_cost, estimated_monthly_cost, estimated_monthly_savings must be numbers
- confidence_score must be a number between 0.0 and 1.0
- If data is insufficient, set confidence_score to low values (0.1-0.3)
- Include warning field if evidence quality is questionable
- Use confidence_score to reflect data quality and recommendation reliability:
  * 0.8-1.0: Strong evidence from comprehensive metrics and billing data
  * 0.6-0.8: Good evidence with some data limitations
  * 0.4-0.6: Moderate evidence, recommendation based on limited data
  * 0.1-0.4: Poor evidence, high uncertainty in recommendation

DETAILED FIELD INSTRUCTIONS:
- evidence: Can include additional fields like compliance_notes, security_implications, etc. as needed
- implementation_steps: Provide 1-10 steps based on complexity:
  * Simple changes (storage class, unused resources): 2-4 steps
  * Medium changes (rightsizing, purchasing options): 3-6 steps
  * Complex changes (architecture, topology): 5-10 steps
- prerequisites: Include 0-5 prerequisites based on actual requirements:
  * Simple changes: may have no prerequisites (empty array [])
  * Complex changes: may require multiple prerequisites
  * Only include if genuinely required for successful implementation

ADAPTIVE LIST SIZING GUIDELINES:
- implementation_steps: Vary 1-10 steps based on actual complexity
  * Simple (delete unused resource): 2-3 steps
  * Medium (rightsizing, storage class): 3-5 steps
  * Complex (architecture changes): 5-10 steps
- prerequisites: Use 0-5 items based on actual requirements
  * Simple changes: often empty array []
  * Complex changes: multiple prerequisites
  * Only include genuine requirements, not generic advice
- evidence: Include 3+ relevant fields, add domain-specific fields as needed
  * Always include: metrics_analysis, cost_breakdown, performance_impact
  * Add as relevant: security_implications, compliance_notes, availability_impact, etc.
- Quality over quantity: each list item should add genuine value"""

    def _build_resource_context(
        self, 
        resource_data: Dict[str, Any], 
        resource_type: str, 
        service_type: str
    ) -> str:
        """Build context section with resource details."""
        context = f"RESOURCE ANALYSIS CONTEXT:\n"
        context += f"Service: {service_type}\n"
        context += f"Resource Type: {resource_type}\n"
        
        if 'id' in resource_data:
            context += f"Resource ID: {resource_data['id']}\n"
        
        if 'configuration' in resource_data:
            context += f"Configuration: {json.dumps(resource_data['configuration'], indent=2)}\n"
        
        if 'metrics' in resource_data:
            context += f"Available Metrics: {json.dumps(resource_data['metrics'], indent=2)}\n"
        else:
            context += "Available Metrics: None provided\n"
        
        if 'billing_data' in resource_data:
            context += f"Billing Information: {json.dumps(resource_data['billing_data'], indent=2)}\n"
        
        return context
    
    def _get_service_specific_guidance(self, service_type: str) -> str:
        """Get service-specific analysis guidance."""
        guidance_map = {
            'EC2': """
EC2 ANALYSIS GUIDANCE:
- Focus on CPU utilization, memory usage, and network patterns
- Analyze time-series trends (increasing, decreasing, stable) and volatility patterns
- Consider usage patterns throughout the monitoring period, not just averages
- Use peak hours information to understand workload characteristics
- Examine CPU time-series data for cyclical patterns, spikes, and baseline utilization
- Evaluate trend direction when making rightsizing decisions:
  * Increasing trends: Be cautious about downsizing
  * Decreasing trends: Consider more aggressive downsizing
  * High volatility: Factor in burst capacity requirements
- Consider right-sizing opportunities based on actual utilization trends
- Evaluate Reserved Instance or Spot Instance potential
- Assess storage optimization (EBS volume types and sizes)
- Only recommend instance type changes if utilization data clearly supports it and trends are considered
""",
            'EBS': """
EBS ANALYSIS GUIDANCE:
- Analyze IOPS utilization vs provisioned IOPS
- Review throughput patterns and volume type appropriateness
- Consider gp3 migration opportunities when usage patterns support it
- Evaluate snapshot policies and lifecycle management
- Only suggest volume type changes when performance data is available
""",
            'S3': """
S3 ANALYSIS GUIDANCE:
- Review storage class distribution and access patterns
- Analyze request patterns (GET, PUT, LIST operations)
- Consider lifecycle policies for infrequently accessed data
- Evaluate transfer costs and CloudFront integration opportunities
- Base storage class recommendations on actual access frequency data
""",
            'RDS': """
RDS ANALYSIS GUIDANCE:
- Focus on database performance metrics (CPU, memory, connections)
- Analyze storage growth patterns and IOPS requirements
- Consider Reserved Instance opportunities for stable workloads
- Evaluate backup retention and cross-region replication costs
- Only recommend instance changes when performance metrics support it
""",
            'Lambda': """
LAMBDA ANALYSIS GUIDANCE:
- Analyze execution duration, memory usage, and invocation patterns
- Consider memory optimization based on actual usage
- Evaluate cold start patterns and provisioned concurrency needs
- Review error rates and timeout configurations
- Base memory recommendations on actual execution profiles
"""
        }
        
        return guidance_map.get(service_type, """
GENERAL ANALYSIS GUIDANCE:
- Focus on resource utilization and cost efficiency
- Consider scaling patterns and usage trends
- Evaluate alternative service options when appropriate
- Assess operational overhead vs cost savings
- Base all recommendations on provided metrics and evidence
""")
    
    def create_simple_prompt(self, resource_type: str, service_type: str) -> str:
        """Create a simplified prompt for basic analysis."""
        return f"""Analyze this {service_type} {resource_type} for cost optimization opportunities.

Provide recommendations in JSON format with:
- resource_id
- service
- current_cost_estimate
- recommendations (with action, rationale, estimated_savings)
- confidence_level

Base analysis only on provided data. Do not fabricate metrics."""

    def get_json_cleaning_instructions(self) -> str:
        """Get instructions for cleaning LLM JSON responses."""
        return """
Instructions for cleaning JSON responses:
1. Remove any text before the opening {
2. Remove any text after the closing }
3. Remove inline comments (// text)
4. Remove block comments (/* text */)
5. Fix common JSON formatting issues
6. Ensure proper string escaping
"""

    def create_coordinator_prompt(self, resources_summary: Dict[str, Any]) -> str:
        """Create prompt for coordinator-level analysis."""
        return f"""You are coordinating cost optimization analysis across multiple AWS resources.

RESOURCE SUMMARY:
{json.dumps(resources_summary, indent=2)}

Provide a high-level analysis focusing on:
1. Cross-service optimization opportunities
2. Overall spending patterns
3. Priority recommendations
4. Resource dependencies and constraints

Respond in JSON format with coordinated recommendations."""

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
