"""
LLM service for the cost recommendation system.
"""
import asyncio
from typing import Dict, List, Any, Optional
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel

from .config import LLMConfig

logger = structlog.get_logger(__name__)


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
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts running"""
        import time
        self.start_time = time.time()
        logger.info("LLM request started", prompt_length=len(prompts[0]) if prompts else 0)
    
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
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance"""
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            openai_api_key=self.config.api_key,
            openai_api_base=self.config.base_url
        )
    
    async def generate_recommendation(
        self,
        system_prompt: str,
        user_prompt: str,
        context_data: Dict[str, Any]
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
                HumanMessage(content=formatted_prompt)
            ]
            
            logger.info("Generating LLM recommendation", 
                       model=self.config.model,
                       system_prompt_length=len(system_prompt),
                       user_prompt_length=len(formatted_prompt))
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            
            response_time = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.content,
                usage_tokens=getattr(response, 'usage_metadata', {}).get('total_tokens', 0),
                model=self.config.model,
                response_time_ms=response_time
            )
            
        except Exception as e:
            logger.error("Failed to generate LLM recommendation", error=str(e))
            raise
    
    def _format_prompt(self, template: str, context_data: Dict[str, Any]) -> str:
        """Format prompt template with context data"""
        try:
            # Convert context data to a readable format
            context_str = self._format_context_data(context_data)
            
            # Replace placeholders in template
            formatted_prompt = template.format(**context_data)
            
            # Add context data at the end
            formatted_prompt += f"\n\nContext Data:\n{context_str}"
            
            return formatted_prompt
            
        except KeyError as e:
            logger.warning("Missing key in prompt formatting", missing_key=str(e))
            # Return template as-is if formatting fails
            return template + f"\n\nContext Data:\n{self._format_context_data(context_data)}"
    
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
        batch_size: int = 5
    ) -> List[LLMResponse]:
        """Analyze multiple resources in batches"""
        results = []
        
        for i in range(0, len(resources_data), batch_size):
            batch = resources_data[i:i + batch_size]
            
            # Create batch prompt
            batch_prompt = self._create_batch_prompt(batch)
            
            try:
                response = await self.generate_recommendation(
                    system_prompt=system_prompt,
                    user_prompt=batch_prompt,
                    context_data={"batch_size": len(batch)}
                )
                results.append(response)
                
                # Rate limiting - wait between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Failed to analyze resource batch", batch_index=i, error=str(e))
                # Continue with next batch
                continue
        
        return results
    
    def _create_batch_prompt(self, resources: List[Dict[str, Any]]) -> str:
        """Create prompt for batch resource analysis"""
        prompt_parts = ["Analyze the following resources for cost optimization opportunities:\n"]
        
        for idx, resource in enumerate(resources, 1):
            prompt_parts.append(f"\nResource {idx}:")
            prompt_parts.append(self._format_context_data(resource))
            prompt_parts.append("-" * 50)
        
        prompt_parts.append("\nProvide recommendations for each resource in JSON format.")
        
        return "\n".join(prompt_parts)
    
    def update_config(self, new_config: LLMConfig):
        """Update LLM configuration"""
        self.config = new_config
        self.llm = self._create_llm()
        logger.info("LLM configuration updated", model=new_config.model)


class PromptTemplates:
    """Collection of prompt templates"""
    
    BASE_SYSTEM_PROMPT = """You are an expert AWS cost optimization specialist with deep knowledge of cloud infrastructure, pricing models, and best practices. Your goal is to analyze AWS resources and provide actionable cost optimization recommendations.

Key principles:
1. Minimize cost while preserving performance and reliability
2. Consider business impact and risk levels
3. Provide specific, actionable recommendations
4. Include exact cost calculations and savings estimates
5. Consider implementation complexity and timeline

Response format:
Provide your recommendations in JSON format with the following structure:
{
    "recommendations": [
        {
            "resource_id": "string",
            "recommendation_type": "rightsizing|purchasing_option|lifecycle|topology|storage_class|idle_resource",
            "current_config": {...},
            "recommended_config": {...},
            "current_monthly_cost": number,
            "estimated_monthly_cost": number,
            "estimated_monthly_savings": number,
            "risk_level": "low|medium|high",
            "rationale": "string",
            "implementation_steps": ["string"],
            "confidence_score": number
        }
    ]
}"""
    
    EC2_PROMPT = """Analyze EC2 instances for cost optimization. Focus on:
1. CPU and memory utilization patterns over the analysis period
2. Right-sizing opportunities based on actual usage
3. Purchase option recommendations (On-Demand vs Reserved vs Spot)
4. Idle or underutilized instances
5. Instance family/generation upgrades for better price-performance

Consider the resource properties, metrics, and billing data provided.
If CPU utilization is consistently below 20% and memory below 30%, recommend downsizing.
If utilization patterns are predictable, recommend Reserved Instances.
Identify completely idle instances (< 5% CPU for extended periods) for termination."""
    
    EBS_PROMPT = """Analyze EBS volumes for cost optimization. Focus on:
1. IOPS and throughput utilization vs provisioned capacity
2. Volume type optimization (gp2 -> gp3, oversized Provisioned IOPS)
3. Unattached or unused volumes
4. Snapshot management and lifecycle
5. Storage rightsizing based on actual usage

Consider burst credits usage for gp2 volumes and recommend gp3 for better cost control.
Identify volumes with low utilization for downsizing or deletion."""
    
    S3_PROMPT = """Analyze S3 storage for cost optimization. Focus on:
1. Storage class optimization based on access patterns
2. Lifecycle policy implementation for automatic transitions
3. Incomplete multipart uploads cleanup
4. Intelligent Tiering opportunities
5. Cross-region replication costs

Recommend storage class transitions:
- Standard to IA after 30 days if access is infrequent
- IA to Glacier after 90 days for archival data
- Glacier to Deep Archive for long-term retention"""
    
    RDS_PROMPT = """Analyze RDS instances for cost optimization. Focus on:
1. CPU, memory, and IOPS utilization patterns
2. Instance class rightsizing opportunities
3. Purchase option recommendations (On-Demand vs Reserved)
4. Storage type optimization and sizing
5. Multi-AZ vs Single-AZ based on requirements

Consider database engine efficiency and version upgrades.
Recommend Reserved Instances for predictable workloads."""
    
    LAMBDA_PROMPT = """Analyze Lambda functions for cost optimization. Focus on:
1. Memory allocation vs actual usage
2. Execution duration optimization
3. Cold start reduction strategies
4. Invocation patterns and frequency
5. Architecture improvements for cost efficiency

Recommend memory adjustments based on actual usage patterns.
Consider Provisioned Concurrency for frequently invoked functions."""
