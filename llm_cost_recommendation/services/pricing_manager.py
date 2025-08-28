"""
Pricing Manager - Orchestrates pricing services across cloud providers.
"""

from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import structlog
from pathlib import Path

from ..models.pricing import (
    ServicePricing, CostCalculation, PricingModel, Currency
)
from ..models import Resource, CloudProvider
from .config import ConfigManager
from .pricing import AWSPricingService

logger = structlog.get_logger(__name__)


class PricingManager:
    """
    Multi-cloud pricing manager that provides a unified interface
    to pricing services across different cloud providers.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.pricing_services: Dict[CloudProvider, Any] = {}
        
        # Initialize pricing services based on configuration
        self._initialize_pricing_services()
    
    def _initialize_pricing_services(self):
        """Initialize pricing services for each cloud provider"""
        try:
            # Initialize AWS pricing service
            self.pricing_services[CloudProvider.AWS] = AWSPricingService(
                config_manager=self.config_manager
            )
            logger.info("Initialized AWS pricing service")
            
            # TODO: Initialize Azure and GCP pricing services when implemented
            # self.pricing_services[CloudProvider.AZURE] = AzurePricingService(...)
            # self.pricing_services[CloudProvider.GCP] = GCPPricingService(...)
            
        except Exception as e:
            logger.error("Failed to initialize pricing services", error=str(e))
            raise
    
    async def get_service_pricing(
        self,
        cloud_provider: CloudProvider,
        service_type: str,
        region: str,
        **kwargs
    ) -> Optional[ServicePricing]:
        """
        Get pricing for a service from the appropriate cloud provider.
        
        Args:
            cloud_provider: Cloud provider (AWS, Azure, GCP)
            service_type: Service type (e.g., 'AWS.EC2', 'Azure.VirtualMachines')
            region: Cloud region
            **kwargs: Additional service-specific parameters
        
        Returns:
            ServicePricing object or None if pricing unavailable
        """
        pricing_service = self.pricing_services.get(cloud_provider)
        if not pricing_service:
            logger.warning("No pricing service available", provider=cloud_provider)
            return None
        
        try:
            pricing = await pricing_service.get_service_pricing(
                service_type=service_type,
                region=region,
                **kwargs
            )
            
            if pricing:
                logger.debug("Retrieved pricing", 
                           provider=cloud_provider, 
                           service=service_type,
                           region=region)
            else:
                logger.warning("No pricing found",
                             provider=cloud_provider,
                             service=service_type,
                             region=region)
            
            return pricing
            
        except Exception as e:
            logger.error("Failed to get service pricing",
                        provider=cloud_provider,
                        service=service_type,
                        error=str(e))
            return None
    
    async def calculate_resource_cost(
        self,
        resource: Resource,
        usage_hours: float = 730,  # Default: 1 month
        pricing_model: PricingModel = PricingModel.ON_DEMAND,
        **kwargs
    ) -> Optional[CostCalculation]:
        """
        Calculate cost for a specific resource.
        
        Args:
            resource: Resource to calculate cost for
            usage_hours: Usage hours for calculation (default: 730 hours/month)
            pricing_model: Pricing model to use
            **kwargs: Additional parameters for cost calculation
        
        Returns:
            CostCalculation object or None if calculation failed
        """
        logger.info("Starting resource cost calculation",
                   resource_id=resource.resource_id,
                   service=resource.service,
                   region=resource.region,
                   usage_hours=usage_hours,
                   pricing_model=pricing_model.value)
        
        # Determine cloud provider from service type
        cloud_provider = self._get_cloud_provider_from_service(resource.service)
        if not cloud_provider:
            logger.error("Cannot determine cloud provider", 
                        resource_id=resource.resource_id,
                        service=resource.service)
            return None

        logger.debug("Cloud provider determined",
                    resource_id=resource.resource_id,
                    provider=cloud_provider.value)

        pricing_service = self.pricing_services.get(cloud_provider)
        if not pricing_service:
            logger.warning("No pricing service available", 
                          resource_id=resource.resource_id,
                          provider=cloud_provider.value)
            return None

        try:
            cost_calculation = await pricing_service.calculate_cost(
                resource=resource,
                usage_hours=usage_hours,
                pricing_model=pricing_model,
                **kwargs
            )
            
            if cost_calculation:
                logger.info("Resource cost calculation successful",
                           resource_id=resource.resource_id,
                           service=resource.service,
                           monthly_cost=float(cost_calculation.current_monthly_cost),
                           annual_cost=float(cost_calculation.current_annual_cost))
            else:
                logger.warning("Resource cost calculation returned no result",
                             resource_id=resource.resource_id,
                             service=resource.service)
            
            return cost_calculation
            
        except Exception as e:
            logger.error("Failed to calculate resource cost",
                        resource_id=resource.resource_id,
                        service=resource.service,
                        error=str(e))
            return None
    
    async def calculate_bulk_costs(
        self,
        resources: List[Resource],
        usage_hours: float = 730,
        pricing_model: PricingModel = PricingModel.ON_DEMAND
    ) -> Dict[str, CostCalculation]:
        """
        Calculate costs for multiple resources efficiently.
        
        Args:
            resources: List of resources to calculate costs for
            usage_hours: Usage hours for calculation
            pricing_model: Pricing model to use
        
        Returns:
            Dictionary mapping resource_id to CostCalculation
        """
        cost_calculations = {}
        
        # Group resources by cloud provider for efficient processing
        resources_by_provider = self._group_resources_by_provider(resources)
        
        for cloud_provider, provider_resources in resources_by_provider.items():
            pricing_service = self.pricing_services.get(cloud_provider)
            if not pricing_service:
                logger.warning("No pricing service for provider", provider=cloud_provider)
                continue
            
            # Calculate costs for this provider's resources
            for resource in provider_resources:
                try:
                    cost_calc = await self.calculate_resource_cost(
                        resource=resource,
                        usage_hours=usage_hours,
                        pricing_model=pricing_model
                    )
                    
                    if cost_calc:
                        cost_calculations[resource.resource_id] = cost_calc
                        
                except Exception as e:
                    logger.error("Failed to calculate cost for resource",
                               resource_id=resource.resource_id,
                               error=str(e))
        
        logger.info("Calculated bulk costs",
                   total_resources=len(resources),
                   successful_calculations=len(cost_calculations))
        
        return cost_calculations
    
    async def compare_pricing_models(
        self,
        resource: Resource,
        pricing_models: List[PricingModel],
        usage_hours: float = 730
    ) -> Dict[PricingModel, Optional[CostCalculation]]:
        """
        Compare costs across different pricing models for a resource.
        
        Args:
            resource: Resource to analyze
            pricing_models: List of pricing models to compare
            usage_hours: Usage hours for calculation
        
        Returns:
            Dictionary mapping pricing model to cost calculation
        """
        comparisons = {}
        
        for pricing_model in pricing_models:
            try:
                cost_calc = await self.calculate_resource_cost(
                    resource=resource,
                    usage_hours=usage_hours,
                    pricing_model=pricing_model
                )
                comparisons[pricing_model] = cost_calc
                
            except Exception as e:
                logger.error("Failed to calculate cost for pricing model",
                           resource_id=resource.resource_id,
                           pricing_model=pricing_model,
                           error=str(e))
                comparisons[pricing_model] = None
        
        return comparisons
    
    def _get_cloud_provider_from_service(self, service_type: str) -> Optional[CloudProvider]:
        """Extract cloud provider from service type"""
        if service_type.startswith("AWS."):
            return CloudProvider.AWS
        elif service_type.startswith("Azure.") or service_type.startswith("AZURE."):
            return CloudProvider.AZURE
        elif service_type.startswith("GCP.") or service_type.startswith("Google."):
            return CloudProvider.GCP
        else:
            # Try to infer from service name
            service_lower = service_type.lower()
            if any(aws_svc in service_lower for aws_svc in ['ec2', 's3', 'rds', 'lambda', 'ebs']):
                return CloudProvider.AWS
            elif any(azure_svc in service_lower for azure_svc in ['vm', 'blob', 'sql', 'functions']):
                return CloudProvider.AZURE
            elif any(gcp_svc in service_lower for gcp_svc in ['compute', 'storage', 'cloudsql']):
                return CloudProvider.GCP
        
        return None
    
    def _group_resources_by_provider(self, resources: List[Resource]) -> Dict[CloudProvider, List[Resource]]:
        """Group resources by cloud provider for efficient bulk processing"""
        grouped = {}
        
        for resource in resources:
            provider = self._get_cloud_provider_from_service(resource.service)
            if provider:
                if provider not in grouped:
                    grouped[provider] = []
                grouped[provider].append(resource)
        
        return grouped
    
    def get_pricing_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from all pricing services"""
        stats = {}
        
        for provider, service in self.pricing_services.items():
            if hasattr(service, 'get_cache_stats'):
                stats[provider.value] = service.get_cache_stats()
        
        return stats
    
    def clear_pricing_cache(self, cloud_provider: Optional[CloudProvider] = None):
        """Clear pricing cache for specified provider or all providers"""
        if cloud_provider:
            service = self.pricing_services.get(cloud_provider)
            if service and hasattr(service, 'pricing_cache'):
                service.pricing_cache.clear()
                logger.info("Cleared pricing cache", provider=cloud_provider)
        else:
            # Clear cache for all providers
            for provider, service in self.pricing_services.items():
                if hasattr(service, 'pricing_cache'):
                    service.pricing_cache.clear()
            logger.info("Cleared all pricing caches")
    
    async def validate_pricing_configuration(self) -> Dict[str, Any]:
        """Validate pricing configuration and service availability"""
        validation_results = {
            "overall_status": "healthy",
            "providers": {},
            "issues": []
        }
        
        for provider, service in self.pricing_services.items():
            provider_status = {
                "available": True,
                "configuration_loaded": False,
                "api_accessible": False,
                "cache_enabled": False
            }
            
            try:
                # Check if configuration is loaded
                if hasattr(service, 'pricing_rules') and service.pricing_rules:
                    provider_status["configuration_loaded"] = True
                
                # Check if caching is enabled
                if hasattr(service, 'pricing_cache'):
                    provider_status["cache_enabled"] = True
                
                # TODO: Add API accessibility check
                # This would involve making a test API call to verify connectivity
                
            except Exception as e:
                provider_status["available"] = False
                validation_results["issues"].append(f"{provider}: {str(e)}")
                validation_results["overall_status"] = "degraded"
            
            validation_results["providers"][provider.value] = provider_status
        
        return validation_results
