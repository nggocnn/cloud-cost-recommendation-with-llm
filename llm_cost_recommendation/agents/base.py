"""
Base agent class for service-specific cost optimization agents.
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

from ..models import (
    Resource, Metrics, BillingData, Recommendation, 
    ServiceType, RecommendationType, RiskLevel, ServiceAgentConfig
)
from ..services.llm import LLMService, PromptTemplates
from ..services.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for service-specific cost optimization agents"""
    
    def __init__(self, config: ServiceAgentConfig, llm_service: LLMService):
        self.config = config
        self.llm_service = llm_service
        self.service = config.service
        self.agent_id = config.agent_id
        
        logger.info("Agent initialized", agent_id=self.agent_id, service=self.service)
    
    @abstractmethod
    async def analyze_resource(
        self, 
        resource: Resource, 
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> List[Recommendation]:
        """Analyze a single resource and generate recommendations"""
        pass
    
    async def analyze_resources(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None
    ) -> List[Recommendation]:
        """Analyze multiple resources"""
        recommendations = []
        
        for resource in resources:
            if resource.service != self.service:
                continue
            
            try:
                resource_metrics = metrics_data.get(resource.resource_id) if metrics_data else None
                resource_billing = billing_data.get(resource.resource_id) if billing_data else None
                
                resource_recommendations = await self.analyze_resource(
                    resource, resource_metrics, resource_billing
                )
                
                recommendations.extend(resource_recommendations)
                
            except Exception as e:
                logger.error("Failed to analyze resource", 
                           resource_id=resource.resource_id, 
                           agent_id=self.agent_id,
                           error=str(e))
                continue
        
        logger.info("Resource analysis completed", 
                   agent_id=self.agent_id,
                   total_resources=len(resources),
                   total_recommendations=len(recommendations))
        
        return recommendations
    
    def _prepare_context_data(
        self, 
        resource: Resource, 
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> Dict[str, Any]:
        """Prepare context data for LLM analysis"""
        context = {
            'resource': {
                'resource_id': resource.resource_id,
                'service': resource.service.value,
                'region': resource.region,
                'availability_zone': resource.availability_zone,
                'account_id': resource.account_id,
                'tags': resource.tags,
                'properties': resource.properties,
                'extensions': resource.extensions
            }
        }
        
        if metrics:
            context['metrics'] = {
                'period_days': metrics.period_days,
                'is_idle': metrics.is_idle,
                'cpu_utilization_p50': metrics.cpu_utilization_p50,
                'cpu_utilization_p90': metrics.cpu_utilization_p90,
                'cpu_utilization_p95': metrics.cpu_utilization_p95,
                'memory_utilization_p50': metrics.memory_utilization_p50,
                'memory_utilization_p90': metrics.memory_utilization_p90,
                'memory_utilization_p95': metrics.memory_utilization_p95,
                'iops_read': metrics.iops_read,
                'iops_write': metrics.iops_write,
                'throughput_read': metrics.throughput_read,
                'throughput_write': metrics.throughput_write,
                'network_in': metrics.network_in,
                'network_out': metrics.network_out,
                'other_metrics': metrics.metrics
            }
        
        if billing_data:
            context['billing'] = {
                'total_monthly_cost': sum(bd.unblended_cost for bd in billing_data),
                'usage_patterns': [
                    {
                        'usage_type': bd.usage_type,
                        'usage_amount': bd.usage_amount,
                        'usage_unit': bd.usage_unit,
                        'cost': bd.unblended_cost
                    }
                    for bd in billing_data
                ]
            }
        
        # Add service-specific thresholds
        context['thresholds'] = self.config.capability.thresholds
        context['analysis_window_days'] = self.config.capability.analysis_window_days
        
        return context
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        return f"{PromptTemplates.BASE_SYSTEM_PROMPT}\n\n{self.config.base_prompt}\n\n{self.config.service_specific_prompt}"
    
    def _create_user_prompt(self, context_data: Dict[str, Any]) -> str:
        """Create user prompt with context data"""
        prompt_parts = [
            f"Analyze the following {self.service.value} resource for cost optimization opportunities:",
            "",
            "Resource Information:",
            json.dumps(context_data.get('resource', {}), indent=2),
            ""
        ]
        
        if 'metrics' in context_data:
            prompt_parts.extend([
                "Performance Metrics:",
                json.dumps(context_data['metrics'], indent=2),
                ""
            ])
        
        if 'billing' in context_data:
            prompt_parts.extend([
                "Billing Information:",
                json.dumps(context_data['billing'], indent=2),
                ""
            ])
        
        prompt_parts.extend([
            "Configuration:",
            f"- Analysis window: {context_data.get('analysis_window_days', 30)} days",
            f"- Service thresholds: {json.dumps(context_data.get('thresholds', {}), indent=2)}",
            "",
            "Please provide cost optimization recommendations in the specified JSON format."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _generate_recommendations_from_llm(
        self, 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using LLM"""
        try:
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(context_data)
            
            response = await self.llm_service.generate_recommendation(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_data=context_data
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response.content)
                recommendations = response_data.get('recommendations', [])
                
                logger.info("LLM recommendations generated", 
                           agent_id=self.agent_id,
                           recommendations_count=len(recommendations),
                           response_time_ms=response.response_time_ms)
                
                return recommendations
                
            except json.JSONDecodeError as e:
                logger.error("Failed to parse LLM response as JSON", 
                           agent_id=self.agent_id,
                           response_content=response.content[:500],
                           error=str(e))
                return []
                
        except Exception as e:
            logger.error("Failed to generate recommendations from LLM", 
                        agent_id=self.agent_id,
                        error=str(e))
            return []
    
    def _convert_llm_recommendation_to_model(
        self, 
        llm_recommendation: Dict[str, Any], 
        resource: Resource
    ) -> Optional[Recommendation]:
        """Convert LLM recommendation dict to Recommendation model"""
        try:
            # Generate unique ID
            rec_id = f"{self.agent_id}_{resource.resource_id}_{datetime.utcnow().isoformat()}"
            
            # Parse recommendation type
            rec_type_str = llm_recommendation.get('recommendation_type', 'rightsizing')
            try:
                rec_type = RecommendationType(rec_type_str)
            except ValueError:
                rec_type = RecommendationType.RIGHTSIZING
            
            # Parse risk level
            risk_str = llm_recommendation.get('risk_level', 'medium')
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM
            
            # Calculate savings
            current_cost = float(llm_recommendation.get('current_monthly_cost', 0))
            estimated_cost = float(llm_recommendation.get('estimated_monthly_cost', 0))
            monthly_savings = current_cost - estimated_cost
            annual_savings = monthly_savings * 12
            
            # Apply minimum cost threshold
            if monthly_savings < self.config.min_cost_threshold:
                logger.debug("Recommendation below minimum cost threshold", 
                           resource_id=resource.resource_id,
                           monthly_savings=monthly_savings,
                           threshold=self.config.min_cost_threshold)
                return None
            
            # Apply confidence threshold
            confidence = float(llm_recommendation.get('confidence_score', 0.5))
            if confidence < self.config.confidence_threshold:
                logger.debug("Recommendation below confidence threshold", 
                           resource_id=resource.resource_id,
                           confidence=confidence,
                           threshold=self.config.confidence_threshold)
                return None
            
            recommendation = Recommendation(
                id=rec_id,
                resource_id=resource.resource_id,
                service=self.service,
                recommendation_type=rec_type,
                current_config=llm_recommendation.get('current_config', {}),
                recommended_config=llm_recommendation.get('recommended_config', {}),
                current_monthly_cost=current_cost,
                estimated_monthly_cost=estimated_cost,
                estimated_monthly_savings=monthly_savings,
                annual_savings=annual_savings,
                risk_level=risk_level,
                impact_description=llm_recommendation.get('impact_description', ''),
                rollback_plan=llm_recommendation.get('rollback_plan', ''),
                rationale=llm_recommendation.get('rationale', ''),
                implementation_steps=llm_recommendation.get('implementation_steps', []),
                confidence_score=confidence,
                agent_id=self.agent_id
            )
            
            return recommendation
            
        except Exception as e:
            logger.error("Failed to convert LLM recommendation to model", 
                        agent_id=self.agent_id,
                        resource_id=resource.resource_id,
                        error=str(e))
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
            'agent_id': self.agent_id,
            'service': self.service.value,
            'supported_recommendation_types': [rt.value for rt in self.config.capability.supported_recommendation_types],
            'required_metrics': self.config.capability.required_metrics,
            'optional_metrics': self.config.capability.optional_metrics,
            'thresholds': self.config.capability.thresholds,
            'enabled': self.config.enabled
        }


class RuleBasedAgent(BaseAgent):
    """Agent that applies deterministic rules for obvious optimizations"""
    
    async def analyze_resource(
        self, 
        resource: Resource, 
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> List[Recommendation]:
        """Apply rule-based analysis"""
        recommendations = []
        
        if not self._validate_resource_data(resource):
            return recommendations
        
        # Apply service-specific rules
        rule_recommendations = self._apply_rules(resource, metrics, billing_data)
        recommendations.extend(rule_recommendations)
        
        return recommendations
    
    def _apply_rules(
        self, 
        resource: Resource, 
        metrics: Optional[Metrics] = None,
        billing_data: Optional[List[BillingData]] = None
    ) -> List[Recommendation]:
        """Apply deterministic rules based on service type"""
        recommendations = []
        
        # Common rules for idle resources
        if metrics and metrics.is_idle:
            rec = self._create_idle_resource_recommendation(resource, metrics, billing_data)
            if rec:
                recommendations.append(rec)
        
        # Service-specific rules
        if self.service == ServiceType.ELASTIC_IP:
            rec = self._check_idle_elastic_ip(resource, billing_data)
            if rec:
                recommendations.append(rec)
        
        elif self.service == ServiceType.EBS:
            rec = self._check_unattached_ebs(resource)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _create_idle_resource_recommendation(
        self, 
        resource: Resource, 
        metrics: Metrics,
        billing_data: Optional[List[BillingData]] = None
    ) -> Optional[Recommendation]:
        """Create recommendation for idle resource"""
        current_cost = sum(bd.unblended_cost for bd in billing_data) if billing_data else 0
        
        rec_id = f"rule_{self.agent_id}_{resource.resource_id}_idle"
        
        return Recommendation(
            id=rec_id,
            resource_id=resource.resource_id,
            service=self.service,
            recommendation_type=RecommendationType.IDLE_RESOURCE,
            current_config={'status': 'running'},
            recommended_config={'status': 'terminated'},
            current_monthly_cost=current_cost,
            estimated_monthly_cost=0.0,
            estimated_monthly_savings=current_cost,
            annual_savings=current_cost * 12,
            risk_level=RiskLevel.LOW,
            impact_description="Resource appears to be idle and can be safely terminated",
            rollback_plan="Resource can be recreated if needed",
            rationale=f"Resource shows idle pattern over {metrics.period_days} days",
            implementation_steps=["Verify resource is not needed", "Terminate resource"],
            confidence_score=0.9,
            agent_id=self.agent_id
        )
    
    def _check_idle_elastic_ip(
        self, 
        resource: Resource, 
        billing_data: Optional[List[BillingData]] = None
    ) -> Optional[Recommendation]:
        """Check for idle Elastic IP addresses"""
        # This is a simplified rule - in practice, you'd check if EIP is attached
        if resource.properties.get('status') == 'available':  # Not attached
            current_cost = sum(bd.unblended_cost for bd in billing_data) if billing_data else 0
            
            rec_id = f"rule_{self.agent_id}_{resource.resource_id}_idle_eip"
            
            return Recommendation(
                id=rec_id,
                resource_id=resource.resource_id,
                service=self.service,
                recommendation_type=RecommendationType.IDLE_RESOURCE,
                current_config={'status': 'available'},
                recommended_config={'status': 'released'},
                current_monthly_cost=current_cost,
                estimated_monthly_cost=0.0,
                estimated_monthly_savings=current_cost,
                annual_savings=current_cost * 12,
                risk_level=RiskLevel.LOW,
                impact_description="Unattached Elastic IP incurs charges",
                rollback_plan="Allocate new EIP if needed",
                rationale="Elastic IP is not attached to any instance",
                implementation_steps=["Verify EIP is not needed", "Release EIP"],
                confidence_score=0.95,
                agent_id=self.agent_id
            )
        
        return None
    
    def _check_unattached_ebs(self, resource: Resource) -> Optional[Recommendation]:
        """Check for unattached EBS volumes"""
        if resource.properties.get('state') == 'available':  # Not attached
            rec_id = f"rule_{self.agent_id}_{resource.resource_id}_unattached_ebs"
            
            return Recommendation(
                id=rec_id,
                resource_id=resource.resource_id,
                service=self.service,
                recommendation_type=RecommendationType.IDLE_RESOURCE,
                current_config={'state': 'available'},
                recommended_config={'state': 'deleted'},
                current_monthly_cost=0.0,  # Would need billing data
                estimated_monthly_cost=0.0,
                estimated_monthly_savings=0.0,
                annual_savings=0.0,
                risk_level=RiskLevel.MEDIUM,
                impact_description="Unattached EBS volume incurs storage charges",
                rollback_plan="Restore from snapshot if needed",
                rationale="EBS volume is not attached to any instance",
                implementation_steps=["Create snapshot if needed", "Delete volume"],
                confidence_score=0.8,
                agent_id=self.agent_id
            )
        
        return None
