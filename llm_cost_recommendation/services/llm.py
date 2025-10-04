"""
LLM service for the cost recommendation system.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from .config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM"""

    content: str
    usage_tokens: int = 0
    model: str
    response_time_ms: float


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
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def generate_recommendation(
        self, system_prompt: str, user_prompt: str, context_data: Dict[str, Any] = None
    ) -> LLMResponse:
        """Generate single recommendation using LLM"""
        return await self._generate_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_data=context_data,
        )

    async def generate_batch_recommendations(
        self,
        system_prompt: str,
        resources_data: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> List[LLMResponse]:
        """Generate LLM recommendations for multiple resources in batches"""
        results = []
        logger.debug(
            "Starting batch processing",
            batch_size=batch_size,
            total_resources=len(resources_data),
        )

        for i in range(0, len(resources_data), batch_size):
            batch = resources_data[i : i + batch_size]

            # Create batch prompt
            batch_prompt = self._create_batch_prompt(batch)

            # Generate response for this batch
            batch_response = await self._generate_llm_response(
                system_prompt=system_prompt,
                user_prompt=batch_prompt,
                max_tokens_override=self._calculate_batch_tokens(len(batch)),
            )

            results.append(batch_response)

            # Rate limiting - wait between batches
            await asyncio.sleep(0.1)

        return results

    async def _generate_llm_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context_data: Dict[str, Any] = None,
        max_tokens_override: int = None,
    ) -> LLMResponse:
        """Core LLM response generation logic"""
        try:
            import time

            start_time = time.time()

            # Format the user prompt with context data if provided
            if context_data:
                formatted_prompt = self._format_prompt(user_prompt, context_data)
            else:
                formatted_prompt = user_prompt

            # Determine max tokens
            max_tokens = max_tokens_override or self.max_tokens

            logger.debug(
                "Generating LLM response",
                model=self.config.model,
                system_prompt_length=len(system_prompt),
                user_prompt_length=len(formatted_prompt),
                max_tokens=max_tokens,
            )

            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            response_time = (time.time() - start_time) * 1000

            # Clean and validate JSON response
            response_content = response.choices[0].message.content.strip()

            logger.debug(
                "Raw LLM response received",
                model=self.config.model,
                response_length=len(response_content),
                response_content=response_content[:200]
                + ("..." if len(response_content) > 200 else ""),
            )

            # Log full response content for debugging
            logger.debug(
                "Full LLM Response Content",
                model=self.config.model,
                full_response=response_content,
            )

            # Remove markdown code blocks if present
            response_content = self._clean_json_response(response_content)

            # Validate JSON
            try:
                json_response = json.loads(response_content)
                response_content = json.dumps(json_response, indent=2)

                logger.debug(
                    "JSON response validated",
                    model=self.config.model,
                    response_time_ms=response_time,
                    recommendations_count=len(json_response.get("recommendations", [])),
                )
            except json.JSONDecodeError as e:
                logger.error(
                    "Invalid JSON response from LLM",
                    model=self.config.model,
                    error=str(e),
                    response_content_first_500=response_content[:500],
                )
                logger.error(
                    "FULL INVALID JSON RESPONSE",
                    model=self.config.model,
                    full_invalid_response=response_content,
                )
                # Return valid JSON structure on parse failure
                response_content = json.dumps(
                    {"recommendations": [], "error": "Failed to parse LLM response"}
                )

            return LLMResponse(
                content=response_content,
                usage_tokens=response.usage.total_tokens if response.usage else 0,
                model=self.config.model,
                response_time_ms=response_time,
            )

        except asyncio.TimeoutError as e:
            logger.error("LLM request timeout", error=str(e), timeout=self.config.timeout)
            raise ValueError(f"LLM request timed out after {self.config.timeout} seconds")
        except Exception as e:
            # Handle different types of LLM errors
            error_type = type(e).__name__
            error_msg = str(e)
            
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                logger.warning("LLM rate limit exceeded", error=error_msg)
                raise ValueError("LLM API rate limit exceeded. Please try again later.")
            elif "auth" in error_msg.lower() or "key" in error_msg.lower():
                logger.error("LLM authentication failed", error_type=error_type)
                raise ValueError("LLM API authentication failed. Check your API key.")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                logger.error("LLM network error", error_type=error_type)
                raise ValueError("Network error connecting to LLM API. Please check your connection.")
            else:
                logger.error("LLM API error", error_type=error_type, error=error_msg)
                raise ValueError(f"LLM API error: {error_msg}")

    def _clean_json_response(self, response_content: str) -> str:
        """Remove markdown code blocks and comments from JSON response"""
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        elif response_content.startswith("```"):
            response_content = response_content[3:]

        if response_content.endswith("```"):
            response_content = response_content[:-3]

        # Remove JSON comments (// style comments)
        import re

        # Remove single-line comments like "// comment"
        response_content = re.sub(r"\s*//.*?(?=\n|$)", "", response_content)

        return response_content.strip()

    def _calculate_batch_tokens(self, batch_size: int) -> int:
        """Calculate dynamic max tokens for batch processing"""
        estimated_tokens_per_resource = 400
        return max(self.max_tokens, batch_size * estimated_tokens_per_resource + 1000)

    def _format_prompt(self, template: str, context_data: Dict[str, Any]) -> str:
        """Format prompt template with context data"""
        try:
            # Convert context data to a readable format
            context_str = self._format_context_data(context_data)

            # Check if template actually has template placeholders (simple word placeholders only)
            # Avoid treating JSON-like structures as templates
            import re

            simple_placeholder_pattern = r"\{[a-zA-Z_][a-zA-Z0-9_]*\}"
            placeholders = re.findall(simple_placeholder_pattern, template)

            if placeholders:
                # Create a safe formatting dict with common expected keys
                format_dict = {
                    "resource_id": context_data.get("resource", {}).get(
                        "resource_id", ""
                    ),
                    "service": context_data.get("resource", {}).get("service", ""),
                    "region": context_data.get("resource", {}).get("region", ""),
                    "analysis_window_days": context_data.get(
                        "analysis_window_days", 30
                    ),
                }

                # Only format if all placeholders are available
                missing_keys = [
                    p.strip("{}")
                    for p in placeholders
                    if p.strip("{}") not in format_dict
                ]
                if missing_keys:
                    logger.debug(
                        "Template placeholders not available, using template as-is",
                        missing_keys=missing_keys,
                        available_keys=list(format_dict.keys()),
                    )
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

    def _create_batch_prompt(self, resources: List[Dict[str, Any]]) -> str:
        """Create prompt for batch resource analysis"""
        prompt_parts = [
            "Analyze the following AWS resources for cost optimization opportunities.",
            "Respond with valid JSON only - no additional text outside the JSON structure.\n",
        ]

        for idx, resource in enumerate(resources, 1):
            resource_id = resource.get("resource_id", f"resource-{idx}")
            prompt_parts.append(f"\n=== RESOURCE {idx}: {resource_id} ===")
            prompt_parts.append(self._format_context_data(resource))
            prompt_parts.append("-" * 60)

        prompt_parts.extend(
            [
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
                "    {",
                '      "resource_id": "exact-resource-id-from-above",',
                '      "recommendation_type": "rightsizing|purchasing_option|idle_resource|lifecycle|topology|storage_class|cost_analysis|general_optimization",',
                '      "impact_description": "Detailed recommendation description and business impact analysis",',
                '      "rationale": "Technical reasoning for this recommendation",',
                '      "evidence": {',
                '        "metrics_analysis": "specific data supporting this recommendation",',
                '        "cost_breakdown": "detailed cost analysis",',
                '        "performance_impact": "expected performance changes"',
                "        // Add additional evidence fields as needed (compliance_notes, security_implications, etc.)",
                "      },",
                '      "current_config": {},',
                '      "recommended_config": {},',
                '      "current_monthly_cost": 100.00,',
                '      "estimated_monthly_cost": 75.00,',
                '      "estimated_monthly_savings": 25.00,',
                '      "confidence_score": 0.85,',
                '      "risk_level": "low|medium|high",',
                '      "implementation_steps": [',
                "        // Provide 1-10 implementation steps based on complexity",
                "        // Simple changes: 2-4 steps, Complex changes: 5-10 steps",
                '        "Step 1: Initial preparation step",',
                '        "Step 2: Main implementation action",',
                '        "Step 3: Verification and monitoring"',
                "        // Add more steps if the change is complex",
                "      ],",
                '      "prerequisites": [',
                "        // Include 0-5 prerequisites based on requirements",
                "        // Simple changes may have no prerequisites (empty array [])",
                "        // Complex changes may require multiple prerequisites",
                '        "Example: Backup verification required"',
                "        // Add more prerequisites only if actually needed",
                "      ],",
                '      "rollback_plan": "how to revert this change if needed",',
                '      "business_hours_impact": false,',
                '      "downtime_required": false,',
                '      "sla_impact": "Expected SLA impact or null"',
                "    }",
                "  ]",
                "}",
                "",
                "ADAPTIVE LIST SIZING GUIDELINES:",
                "- implementation_steps: Use 1-10 steps based on actual complexity",
                "- prerequisites: Use 0-5 items, empty array [] if none needed",
                "- evidence: Include 3+ relevant analysis fields, add custom fields as needed",
                "- Determine list sizes based on actual recommendation complexity",
                "- Quality over quantity - only include meaningful items",
                "",
                "CRITICAL: Return only valid JSON. Every recommendation MUST include ALL fields above.",
            ]
        )

        return "\n".join(prompt_parts)
