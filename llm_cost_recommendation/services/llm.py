"""
LLM service for the cost recommendation system.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel

from .config import LLMConfig
from .logging import get_logger

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM"""

    content: str
    usage_tokens: int = 0
    model: str
    response_time_ms: float


class LoggingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for logging LLM interactions"""

    def __init__(self):
        self.start_time = None
        self.tokens_used = 0

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts running"""
        import time

        self.start_time = time.time()
        logger.info(
            "LLM request started", prompt_length=len(prompts[0]) if prompts else 0
        )

    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running"""
        import time

        if self.start_time:
            response_time = (time.time() - self.start_time) * 1000
            logger.info("LLM request completed", response_time_ms=response_time)


class LLMService:
    """Service for interacting with LLM"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._create_llm()
        self.model = config.model
        self.temperature = config.temperature  
        self.max_tokens = config.max_tokens
        
        # Create direct OpenAI client for batch processing
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance with JSON mode enabled"""
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            openai_api_key=self.config.api_key,
            openai_api_base=self.config.base_url,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    async def generate_recommendation(
        self, system_prompt: str, user_prompt: str, context_data: Dict[str, Any]
    ) -> LLMResponse:
        """Generate recommendation using LLM"""
        try:
            import time

            start_time = time.time()

            # Format the user prompt with context data
            formatted_prompt = self._format_prompt(user_prompt, context_data)

            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=formatted_prompt),
            ]

            logger.info(
                "Generating LLM recommendation",
                model=self.config.model,
                system_prompt_length=len(system_prompt),
                user_prompt_length=len(formatted_prompt),
            )

            # Generate response with JSON mode
            response = await self.llm.ainvoke(messages)

            response_time = (time.time() - start_time) * 1000

            # Clean and validate JSON response (handle markdown code blocks)
            response_content = response.content.strip()
            
            # Log the raw LLM response for debugging
            logger.info(
                "Raw LLM response received",
                model=self.config.model,
                response_length=len(response_content),
                response_content=response_content[:500] + ("..." if len(response_content) > 500 else ""),
            )
            
            # Remove markdown code blocks if present
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # Remove ```json
            elif response_content.startswith("```"):
                response_content = response_content[3:]   # Remove ```
                
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # Remove trailing ```
                
            response_content = response_content.strip()

            try:
                json_response = json.loads(response_content)
                # Update the response content with clean JSON
                response.content = json.dumps(json_response, indent=2)
                
                logger.info(
                    "JSON mode response validated",
                    model=self.config.model,
                    response_time_ms=response_time,
                    recommendations_count=len(json_response.get("recommendations", [])),
                )
            except json.JSONDecodeError as e:
                logger.error(
                    "Invalid JSON response from LLM with JSON mode",
                    model=self.config.model,
                    error=str(e),
                    response_content=response_content[:500],
                )
                # Return valid JSON structure on parse failure
                response.content = json.dumps({
                    "recommendations": [],
                    "error": "Failed to parse LLM response"
                })

            return LLMResponse(
                content=response.content,
                usage_tokens=getattr(response, "usage_metadata", {}).get(
                    "total_tokens", 0
                ),
                model=self.config.model,
                response_time_ms=response_time,
            )

        except Exception as e:
            logger.error("Failed to generate LLM recommendation", error=str(e))
            raise

    def _format_prompt(self, template: str, context_data: Dict[str, Any]) -> str:
        """Format prompt template with context data"""
        try:
            # Convert context data to a readable format
            context_str = self._format_context_data(context_data)

            # Check if template actually has template placeholders (simple word placeholders only)
            # Avoid treating JSON-like structures as templates
            import re
            simple_placeholder_pattern = r'\{[a-zA-Z_][a-zA-Z0-9_]*\}'
            placeholders = re.findall(simple_placeholder_pattern, template)
            
            if placeholders:
                # Create a safe formatting dict with common expected keys
                format_dict = {
                    'resource_id': context_data.get('resource', {}).get('resource_id', ''),
                    'service': context_data.get('resource', {}).get('service', ''),
                    'region': context_data.get('resource', {}).get('region', ''),
                    'account_id': context_data.get('resource', {}).get('account_id', ''),
                    'analysis_window_days': context_data.get('analysis_window_days', 30)
                }

                # Only format if all placeholders are available
                missing_keys = [p.strip('{}') for p in placeholders if p.strip('{}') not in format_dict]
                if missing_keys:
                    logger.debug("Template placeholders not available, using template as-is", 
                               missing_keys=missing_keys, available_keys=list(format_dict.keys()))
                    formatted_prompt = template
                else:
                    formatted_prompt = template.format(**format_dict)
            else:
                # No simple placeholders, use template as-is
                formatted_prompt = template

            # Add context data at the end
            formatted_prompt += f"\n\nContext Data:\n{context_str}"

            return formatted_prompt

        except KeyError as e:
            logger.warning("Missing key in prompt formatting", missing_key=str(e))
            # Return template as-is if formatting fails
            return (
                template
                + f"\n\nContext Data:\n{self._format_context_data(context_data)}"
            )

        except Exception as e:
            logger.error("Error formatting prompt", error=str(e))
            return (
                template
                + f"\n\nContext Data:\n{self._format_context_data(context_data)}"
            )

    def _format_context_data(self, context_data: Dict[str, Any]) -> str:
        """Format context data into readable string"""
        formatted_lines = []

        for key, value in context_data.items():
            if isinstance(value, dict):
                formatted_lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    formatted_lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                formatted_lines.append(f"{key}: {', '.join(map(str, value))}")
            else:
                formatted_lines.append(f"{key}: {value}")

        return "\n".join(formatted_lines)

    async def analyze_resource_batch(
        self,
        system_prompt: str,
        resources_data: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> List[LLMResponse]:
        """Analyze multiple resources in batches"""
        results = []

        for i in range(0, len(resources_data), batch_size):
            batch = resources_data[i : i + batch_size]

            # Create batch prompt
            batch_prompt = self._create_batch_prompt(batch)
 
            try:
                # Use direct client call to avoid template formatting for batch prompts
                import time
                start_time = time.time()
                
                # Calculate dynamic max_tokens based on batch size
                # Each resource typically needs 300-500 tokens for a complete recommendation
                # Add buffer for prompt and overhead
                estimated_tokens_per_resource = 400
                dynamic_max_tokens = max(
                    self.max_tokens, 
                    len(batch) * estimated_tokens_per_resource + 1000
                )
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=dynamic_max_tokens,
                    response_format={"type": "json_object"}
                )
                
                response_time = (time.time() - start_time) * 1000
                
                # Validate JSON response (handle markdown code blocks)
                response_content = response.choices[0].message.content.strip()
                
                # Log the raw batch LLM response for debugging
                logger.info(
                    "Raw batch LLM response received",
                    model=self.model,
                    batch_index=i,
                    batch_size=len(batch),
                    response_length=len(response_content),
                    response_content=response_content[:500] + ("..." if len(response_content) > 500 else ""),
                )
                
                # Remove markdown code blocks if present
                if response_content.startswith("```json"):
                    response_content = response_content[7:]  # Remove ```json
                elif response_content.startswith("```"):
                    response_content = response_content[3:]   # Remove ```
                    
                if response_content.endswith("```"):
                    response_content = response_content[:-3]  # Remove trailing ```
                    
                response_content = response_content.strip()
                
                # Try to parse JSON with repair mechanisms
                json_response = None
                try:
                    json_response = json.loads(response_content)
                    # Clean the response content
                    response_content = json.dumps(json_response, indent=2)
                    
                    logger.info(
                        "Batch JSON response validated",
                        batch_index=i,
                        batch_size=len(batch),
                        recommendations_count=len(json_response.get("recommendations", [])),
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid JSON in batch response",
                        batch_index=i,
                        error=str(e),
                        response_content=response_content[:500],
                    )
                    # Create valid JSON structure on parse failure
                    response_content = json.dumps({
                        "recommendations": [],
                        "error": "Failed to parse batch LLM response"
                    })
                
                # Create LLMResponse manually
                llm_response = LLMResponse(
                    content=response_content,
                    usage_tokens=response.usage.total_tokens if response.usage else 0,
                    model=self.model,
                    response_time_ms=response_time,
                )
                results.append(llm_response)

                # Rate limiting - wait between batches
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    "Failed to analyze resource batch", batch_index=i, error=str(e)
                )
                # Continue with next batch
                continue

        return results

    def _create_batch_prompt(self, resources: List[Dict[str, Any]]) -> str:
        """Create prompt for batch resource analysis"""
        prompt_parts = [
            "Analyze the following AWS resources for cost optimization opportunities.",
            "Respond with valid JSON only - no additional text outside the JSON structure.\n"
        ]

        for idx, resource in enumerate(resources, 1):
            resource_id = resource.get("resource_id", f"resource-{idx}")
            prompt_parts.append(f"\n=== RESOURCE {idx}: {resource_id} ===")
            prompt_parts.append(self._format_context_data(resource))
            prompt_parts.append("-" * 60)

        prompt_parts.extend([
            "\nBATCH ANALYSIS INSTRUCTIONS:",
            "1. Analyze each resource individually with its specific resource_id",
            "2. Look for optimization patterns across similar resources", 
            "3. Consider bulk purchasing opportunities when applicable",
            "4. Provide specific recommendations for each resource",
            "5. Include exact resource_id in each recommendation",
            "6. IMPORTANT: Complete the JSON for ALL resources before ending",
            "",
            "REQUIRED JSON RESPONSE FORMAT (return valid JSON only):",
            "{",
            '  "recommendations": [',
            '    {',
            '      "resource_id": "exact-resource-id-from-above",',
            '      "recommendation_type": "rightsizing|purchasing_option|idle_resource|lifecycle",',
            '      "impact_description": "Detailed recommendation description and business impact analysis",',
            '      "rationale": "Technical reasoning for this recommendation",',
            '      "current_config": {},',
            '      "recommended_config": {},',
            '      "current_monthly_cost": 100.00,',
            '      "estimated_monthly_cost": 75.00,',
            '      "estimated_monthly_savings": 25.00,',
            '      "confidence_score": 0.85,',
            '      "risk_level": "low|medium|high",',
            '      "implementation_steps": ["step 1", "step 2", "step 3"],',
            '      "rollback_plan": "how to revert this change if needed"',
            '    }',
            '  ]',
            '}',
            "",
            "CRITICAL: Return only valid JSON. Every recommendation MUST include ALL fields above.",
        ])

        return "\n".join(prompt_parts)

    def update_config(self, new_config: LLMConfig):
        """Update LLM configuration"""
        self.config = new_config
        self.llm = self._create_llm()
        logger.info("LLM configuration updated", model=new_config.model)


class PromptTemplates:
    """Collection of prompt templates"""

    BASE_SYSTEM_PROMPT = """You are an expert AWS cost optimization specialist with deep knowledge of cloud infrastructure, pricing models, and best practices. Your goal is to analyze AWS resources and provide actionable cost optimization recommendations.

IMPORTANT: You MUST respond in valid JSON format only. No other text outside the JSON structure.

Key principles:
1. Minimize cost while preserving performance and reliability
2. Consider business impact and risk levels
3. Provide specific, actionable recommendations
4. Include exact cost calculations and savings estimates
5. Consider implementation complexity and timeline

REQUIRED JSON Response format (ALL fields are mandatory):
{
    "recommendations": [
        {
            "resource_id": "exact resource identifier",
            "recommendation_type": "rightsizing|purchasing_option|lifecycle|topology|storage_class|idle_resource",
            "current_config": {
                "instance_type": "current type",
                "storage_size": "current size",
                "other_relevant_settings": "current values"
            },
            "recommended_config": {
                "instance_type": "recommended type", 
                "storage_size": "recommended size",
                "other_relevant_settings": "recommended values"
            },
            "current_monthly_cost": 123.45,
            "estimated_monthly_cost": 98.76,
            "estimated_monthly_savings": 24.69,
            "risk_level": "low|medium|high",
            "impact_description": "Detailed explanation of the recommendation and its business impact",
            "rationale": "Technical reasoning behind this recommendation",
            "implementation_steps": [
                "Step 1: Specific action to take",
                "Step 2: Next action to take",
                "Step 3: Final verification step"
            ],
            "rollback_plan": "Detailed plan to revert changes if needed",
            "confidence_score": 0.85
        }
    ]
}

CRITICAL REQUIREMENTS:
- Every recommendation MUST include ALL fields above
- Use exact field names as specified (especially "impact_description")
- Provide realistic cost estimates based on AWS pricing
- Include detailed implementation steps
- Always include proper rollback procedures"""