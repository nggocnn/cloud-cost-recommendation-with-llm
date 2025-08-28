"""
AWS Pricing Service - Config-driven pricing API integration.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import structlog
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import yaml
from pathlib import Path

from ..models.pricing import (
    ServicePricing, PricingUnit, ReservationPricing, PricingRule,
    CostCalculation, PricingCache, PricingModel, Currency,
    ReservationTerm, PaymentOption
)
from ..models import Resource, ServiceType
from .config import ConfigManager

logger = structlog.get_logger(__name__)


class AWSPricingService:
    """
    Config-driven AWS pricing service that prevents code explosion
    by using YAML configurations for service-specific pricing rules.
    """
    
    def __init__(self, config_manager: ConfigManager, region: str = "us-east-1"):
        self.config_manager = config_manager
        self.region = region
        self.pricing_client = None
        self.ec2_client = None
        self.pricing_cache: Dict[str, PricingCache] = {}
        
        # Load pricing configuration
        self.pricing_config = self._load_pricing_config()
        self.pricing_rules = self._load_pricing_rules()
        
        # Initialize AWS clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients"""
        try:
            # Pricing API is only available in us-east-1 and ap-south-1
            self.pricing_client = boto3.client(
                'pricing',
                region_name='us-east-1'  # Pricing API region
            )
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            logger.info("AWS pricing clients initialized", region=self.region)
        except Exception as e:
            logger.error("Failed to initialize AWS clients", error=str(e))
            raise
    
    def _load_pricing_config(self) -> Dict[str, Any]:
        """Load pricing configuration from YAML"""
        config_path = Path("config/pricing/aws/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                "enabled": True,
                "cache_duration_hours": 24,
                "api_rate_limit_per_second": 10,
                "enable_fallback_pricing": True,
                "default_currency": "USD"
            }
    
    def _load_pricing_rules(self) -> Dict[str, PricingRule]:
        """Load service-specific pricing rules from YAML files"""
        pricing_rules = {}
        pricing_dir = Path("config/pricing/aws")
        
        for rule_file in pricing_dir.glob("*.yaml"):
            if rule_file.name in ["config.yaml", "config_api.yaml", "fallback.yaml"]:
                continue
                
            try:
                with open(rule_file, 'r') as f:
                    rule_data = yaml.safe_load(f)
                    service_type = rule_data.get('service_type')
                    if service_type:
                        pricing_rules[service_type] = PricingRule(**rule_data)
                        logger.debug("Loaded pricing rule", service=service_type, file=rule_file.name)
            except Exception as e:
                logger.error("Failed to load pricing rule", file=rule_file.name, error=str(e))
        
        logger.info("Loaded pricing rules", count=len(pricing_rules))
        return pricing_rules
    
    def _generate_cache_key(self, service_type: str, region: str, **kwargs) -> str:
        """Generate cache key for pricing lookup"""
        key_parts = [service_type, region]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def _get_cached_pricing(self, cache_key: str) -> Optional[ServicePricing]:
        """Get pricing from cache if valid"""
        if cache_key in self.pricing_cache:
            cache_entry = self.pricing_cache[cache_key]
            if cache_entry.is_valid():
                cache_entry.hit_count += 1
                logger.info("Cache hit for pricing",
                           cache_key=cache_key,
                           hit_count=cache_entry.hit_count,
                           source=cache_entry.service_pricing.source)
                return cache_entry.service_pricing
            else:
                # Remove expired entry
                del self.pricing_cache[cache_key]
                logger.info("Cache entry expired and removed",
                           cache_key=cache_key,
                           expired_at=cache_entry.expires_at)
        else:
            logger.debug("Cache miss for pricing", cache_key=cache_key)
        return None
    
    def _cache_pricing(self, cache_key: str, pricing: ServicePricing):
        """Cache pricing data"""
        if self.pricing_config.get("enable_pricing_cache", True):
            expires_at = datetime.utcnow() + timedelta(
                hours=self.pricing_config.get("cache_duration_hours", 24)
            )
            self.pricing_cache[cache_key] = PricingCache(
                cache_key=cache_key,
                service_pricing=pricing,
                expires_at=expires_at
            )
            logger.info("Pricing data cached",
                       cache_key=cache_key,
                       source=pricing.source,
                       expires_at=expires_at,
                       cache_size=len(self.pricing_cache))
        else:
            logger.debug("Pricing cache disabled, not caching", cache_key=cache_key)
    
    async def get_service_pricing(
        self,
        service_type: str,
        region: str,
        instance_type: Optional[str] = None,
        storage_type: Optional[str] = None,
        **kwargs
    ) -> Optional[ServicePricing]:
        """
        Get pricing for a service using config-driven rules.
        This prevents code explosion by using YAML configuration.
        """
        logger.info("Starting pricing lookup",
                   service=service_type,
                   region=region,
                   instance_type=instance_type,
                   storage_type=storage_type)
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            service_type, region, 
            instance_type=instance_type,
            storage_type=storage_type,
            **kwargs
        )
        
        # Check cache first
        cached_pricing = self._get_cached_pricing(cache_key)
        if cached_pricing:
            logger.info("Pricing retrieved from cache",
                       service=service_type,
                       source=cached_pricing.source,
                       cache_key=cache_key)
            return cached_pricing
        
        # Get pricing rule for service
        pricing_rule = self.pricing_rules.get(service_type)
        if not pricing_rule:
            logger.warning("No pricing rule found for service", service=service_type)
            return None
        
        logger.info("Using pricing rule",
                   service=service_type,
                   calculation_method=pricing_rule.calculation_method,
                   enable_fallback=pricing_rule.enable_fallback)

        try:
            # Check if we should force fallback mode (useful for demo/testing)
            if self.pricing_config.get("force_fallback_mode", False):
                logger.info("Force fallback mode enabled, using fallback pricing",
                           service=service_type)
                if pricing_rule.enable_fallback:
                    pricing = await self._get_fallback_pricing(
                        pricing_rule, region, instance_type, storage_type
                    )
                    if pricing:
                        logger.info("Fallback pricing retrieved successfully",
                                   service=service_type,
                                   source=pricing.source,
                                   amount=float(pricing.on_demand.amount))
                    return pricing
                else:
                    logger.warning("Fallback disabled for service", service=service_type)
                    return None

            # Get pricing based on calculation method in config
            if pricing_rule.calculation_method == "aws_pricing_api":
                logger.info("Using AWS Pricing API", service=service_type)
                pricing = await self._get_pricing_from_api(
                    pricing_rule, region, instance_type, storage_type, **kwargs
                )
            elif pricing_rule.calculation_method == "ec2_describe_instance_types":
                logger.info("Using EC2 Describe Instance Types API", service=service_type)
                pricing = await self._get_ec2_pricing_from_api(
                    pricing_rule, region, instance_type
                )
            elif pricing_rule.calculation_method == "static_lookup":
                logger.info("Using static lookup", service=service_type)
                pricing = await self._get_static_pricing(
                    pricing_rule, region, instance_type, storage_type
                )
            else:
                logger.error("Unknown calculation method", 
                           service=service_type,
                           method=pricing_rule.calculation_method)
                return None
            
            if pricing:
                logger.info("Pricing retrieved successfully",
                           service=service_type,
                           source=pricing.source,
                           amount=float(pricing.on_demand.amount) if pricing.on_demand else "N/A")
                self._cache_pricing(cache_key, pricing)
            else:
                logger.warning("No pricing data returned from method",
                             service=service_type,
                             method=pricing_rule.calculation_method)

            return pricing
            
        except Exception as e:
            logger.error("Failed to get service pricing", 
                        service=service_type, region=region, error=str(e))
            
            # Try fallback pricing if enabled
            if pricing_rule.enable_fallback:
                logger.info("Attempting fallback pricing due to error",
                           service=service_type,
                           error=str(e))
                fallback_pricing = await self._get_fallback_pricing(
                    pricing_rule, region, instance_type, storage_type
                )
                if fallback_pricing:
                    logger.info("Fallback pricing retrieved successfully",
                               service=service_type,
                               source=fallback_pricing.source,
                               amount=float(fallback_pricing.on_demand.amount))
                return fallback_pricing
            
            return None
    
    async def _get_pricing_from_api(
        self,
        pricing_rule: PricingRule,
        region: str,
        instance_type: Optional[str] = None,
        storage_type: Optional[str] = None,
        **kwargs
    ) -> Optional[ServicePricing]:
        """Get pricing from AWS Pricing API using config-driven filters"""
        logger.info("Calling AWS Pricing API",
                   service=pricing_rule.service_type,
                   region=region,
                   instance_type=instance_type,
                   storage_type=storage_type)
        
        try:
            # Build filters from pricing rule configuration
            filters = self._build_api_filters(pricing_rule, region, instance_type, storage_type)
            
            logger.debug("API filters built",
                        service=pricing_rule.service_type,
                        filter_count=len(filters))
            
            # Call AWS Pricing API
            service_code = pricing_rule.api_filters.get('service_code', 'AmazonEC2')
            logger.info("Making AWS Pricing API call",
                       service_code=service_code,
                       service=pricing_rule.service_type)
            
            response = self.pricing_client.get_products(
                ServiceCode=service_code,
                Filters=filters
            )
            
            logger.info("AWS Pricing API response received",
                       service=pricing_rule.service_type,
                       product_count=len(response.get('PriceList', [])))
            
            # Parse response using config-driven approach
            pricing = self._parse_pricing_response(response, pricing_rule)
            
            if pricing:
                logger.info("AWS API pricing parsed successfully",
                           service=pricing_rule.service_type,
                           amount=float(pricing.on_demand.amount) if pricing.on_demand else "N/A")
            else:
                logger.warning("Failed to parse API response",
                             service=pricing_rule.service_type)
            
            return pricing
            
        except ClientError as e:
            logger.error("AWS Pricing API client error",
                        service=pricing_rule.service_type,
                        error_code=e.response.get('Error', {}).get('Code'),
                        error_message=str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error in pricing API call",
                        service=pricing_rule.service_type,
                        error=str(e))
            return None
    
    async def _get_ec2_pricing_from_api(
        self,
        pricing_rule: PricingRule,
        region: str,
        instance_type: str
    ) -> Optional[ServicePricing]:
        """Get EC2 pricing using describe-instance-types API"""
        try:
            # Get instance type information
            response = self.ec2_client.describe_instance_types(
                InstanceTypes=[instance_type]
            )
            
            if not response['InstanceTypes']:
                return None
            
            instance_info = response['InstanceTypes'][0]
            
            # Build pricing based on instance specifications and config
            pricing = self._build_ec2_pricing_from_specs(
                instance_info, pricing_rule, region
            )
            
            return pricing
            
        except ClientError as e:
            logger.error("EC2 API error", error=str(e))
            return None
    
    async def _get_static_pricing(
        self,
        pricing_rule: PricingRule,
        region: str,
        instance_type: Optional[str] = None,
        storage_type: Optional[str] = None
    ) -> Optional[ServicePricing]:
        """Get pricing from static configuration"""
        logger.info("Using static pricing lookup",
                   service=pricing_rule.service_type,
                   region=region,
                   instance_type=instance_type,
                   storage_type=storage_type)
        
        # Load static pricing data from config
        static_pricing_file = pricing_rule.attributes.get('static_pricing_file')
        if not static_pricing_file:
            logger.warning("No static pricing file configured",
                          service=pricing_rule.service_type)
            return None
        
        pricing_path = Path(f"config/pricing/aws/{static_pricing_file}")
        if not pricing_path.exists():
            logger.error("Static pricing file not found",
                        service=pricing_rule.service_type,
                        file_path=str(pricing_path))
            return None
        
        try:
            logger.debug("Loading static pricing file",
                        service=pricing_rule.service_type,
                        file=static_pricing_file)
            
            with open(pricing_path, 'r') as f:
                static_data = yaml.safe_load(f)
            
            # Lookup pricing based on configuration
            key = f"{region}:{instance_type or storage_type}"
            pricing_data = static_data.get(key)
            
            logger.debug("Static pricing lookup",
                        service=pricing_rule.service_type,
                        lookup_key=key,
                        found=pricing_data is not None)
            
            if pricing_data:
                pricing = ServicePricing(
                    service_type=pricing_rule.service_type,
                    region=region,
                    instance_type=instance_type,
                    storage_type=storage_type,
                    on_demand=PricingUnit(
                        amount=Decimal(str(pricing_data['on_demand'])),
                        currency=Currency.USD,
                        unit=pricing_data.get('unit', 'hour')
                    ),
                    source="static_config"
                )
                
                logger.info("Static pricing found",
                           service=pricing_rule.service_type,
                           amount=float(pricing.on_demand.amount),
                           unit=pricing.on_demand.unit)
                
                return pricing
            else:
                logger.warning("No static pricing data found for lookup key",
                             service=pricing_rule.service_type,
                             lookup_key=key)

        except Exception as e:
            logger.error("Failed to load static pricing",
                        service=pricing_rule.service_type,
                        file=static_pricing_file,
                        error=str(e))
        
        return None
    
    async def _get_fallback_pricing(
        self,
        pricing_rule: PricingRule,
        region: str,
        instance_type: Optional[str] = None,
        storage_type: Optional[str] = None
    ) -> Optional[ServicePricing]:
        """Get fallback pricing with estimated values"""
        logger.info("Using fallback pricing estimation",
                   service=pricing_rule.service_type,
                   region=region,
                   instance_type=instance_type,
                   storage_type=storage_type,
                   multiplier=pricing_rule.fallback_multiplier)
        
        # Use fallback multiplier from config
        multiplier = pricing_rule.fallback_multiplier
        
        # Basic fallback pricing logic based on service type
        base_price = self._get_base_fallback_price(pricing_rule.service_type, instance_type)
        
        logger.debug("Fallback base price calculation",
                    service=pricing_rule.service_type,
                    base_price=base_price,
                    multiplier=multiplier)
        
        if base_price:
            final_price = base_price * multiplier
            pricing = ServicePricing(
                service_type=pricing_rule.service_type,
                region=region,
                instance_type=instance_type,
                storage_type=storage_type,
                on_demand=PricingUnit(
                    amount=Decimal(str(final_price)),
                    currency=Currency.USD,
                    unit="hour"
                ),
                source="fallback_estimate"
            )
            
            logger.info("Fallback pricing generated",
                       service=pricing_rule.service_type,
                       base_price=base_price,
                       final_price=final_price,
                       multiplier=multiplier)
            
            return pricing
        else:
            logger.warning("No base fallback price available",
                          service=pricing_rule.service_type)
        
        return None
    
    def _build_api_filters(
        self,
        pricing_rule: PricingRule,
        region: str,
        instance_type: Optional[str] = None,
        storage_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build AWS Pricing API filters from configuration"""
        filters = []
        
        # Add configured filters from pricing rule
        for filter_config in pricing_rule.api_filters.get('filters', []):
            filter_dict = {
                'Type': filter_config['type'],
                'Field': filter_config['field'],
                'Value': filter_config['value']
            }
            
            # Support dynamic values
            if filter_dict['Value'] == '${region}':
                filter_dict['Value'] = region
            elif filter_dict['Value'] == '${instance_type}' and instance_type:
                filter_dict['Value'] = instance_type
            elif filter_dict['Value'] == '${storage_type}' and storage_type:
                filter_dict['Value'] = storage_type
            
            filters.append(filter_dict)
        
        return filters
    
    def _parse_pricing_response(
        self,
        response: Dict[str, Any],
        pricing_rule: PricingRule
    ) -> Optional[ServicePricing]:
        """Parse AWS Pricing API response using config-driven approach"""
        # This is a complex parsing logic that would be driven by
        # configuration in the pricing rule
        price_list = response.get('PriceList', [])
        if not price_list:
            return None
        
        # Parse first price entry (simplified)
        price_item = json.loads(price_list[0])
        
        # Extract pricing based on config-defined dimensions
        dimensions = pricing_rule.price_dimensions
        on_demand_price = self._extract_price_from_dimensions(price_item, dimensions)
        
        if on_demand_price:
            return ServicePricing(
                service_type=pricing_rule.service_type,
                region=price_item.get('product', {}).get('attributes', {}).get('location', ''),
                instance_type=price_item.get('product', {}).get('attributes', {}).get('instanceType'),
                on_demand=PricingUnit(
                    amount=Decimal(str(on_demand_price)),
                    currency=Currency.USD,
                    unit="hour"
                ),
                source="aws_pricing_api"
            )
        
        return None
    
    def _extract_price_from_dimensions(
        self,
        price_item: Dict[str, Any],
        dimensions: List[str]
    ) -> Optional[float]:
        """Extract price from pricing dimensions"""
        # Navigate through the pricing structure based on configured dimensions
        terms = price_item.get('terms', {})
        on_demand = terms.get('OnDemand', {})
        
        for term_key, term_value in on_demand.items():
            price_dimensions = term_value.get('priceDimensions', {})
            for dim_key, dim_value in price_dimensions.items():
                price_per_unit = dim_value.get('pricePerUnit', {})
                usd_price = price_per_unit.get('USD')
                if usd_price:
                    return float(usd_price)
        
        return None
    
    def _build_ec2_pricing_from_specs(
        self,
        instance_info: Dict[str, Any],
        pricing_rule: PricingRule,
        region: str
    ) -> ServicePricing:
        """Build EC2 pricing from instance specifications"""
        # Use pricing formula from config
        vcpus = instance_info.get('VCpuInfo', {}).get('DefaultVCpus', 1)
        memory = instance_info.get('MemoryInfo', {}).get('SizeInMiB', 1024) / 1024  # Convert to GB
        
        # Apply pricing formula from configuration
        formula = pricing_rule.attributes.get('pricing_formula', {})
        base_price = formula.get('base_price', 0.01)
        vcpu_multiplier = formula.get('vcpu_multiplier', 0.02)
        memory_multiplier = formula.get('memory_multiplier', 0.01)
        
        calculated_price = base_price + (vcpus * vcpu_multiplier) + (memory * memory_multiplier)
        
        return ServicePricing(
            service_type=pricing_rule.service_type,
            region=region,
            instance_type=instance_info['InstanceType'],
            on_demand=PricingUnit(
                amount=Decimal(str(calculated_price)),
                currency=Currency.USD,
                unit="hour"
            ),
            source="calculated_from_specs"
        )
    
    def _get_base_fallback_price(self, service_type: str, instance_type: Optional[str] = None) -> Optional[float]:
        """Get base fallback pricing"""
        # Simple fallback pricing based on service type
        fallback_prices = {
            "AWS.EC2": 0.05,  # Base hourly rate
            "AWS.EBS": 0.10,  # Per GB-month
            "AWS.S3": 0.023,  # Per GB-month
            "AWS.RDS": 0.20,  # Base hourly rate
            "AWS.Lambda": 0.0000166667,  # Per GB-second
        }
        
        return fallback_prices.get(service_type)
    
    async def calculate_cost(
        self,
        resource: Resource,
        usage_hours: float = 730,  # Default: 1 month
        pricing_model: PricingModel = PricingModel.ON_DEMAND
    ) -> Optional[CostCalculation]:
        """Calculate cost for a resource"""
        logger.info("Starting cost calculation",
                   resource_id=resource.resource_id,
                   service=resource.service,
                   region=resource.region,
                   usage_hours=usage_hours,
                   pricing_model=pricing_model.value)
        
        # Get pricing for the resource
        pricing = await self.get_service_pricing(
            service_type=resource.service,
            region=resource.region,
            instance_type=resource.properties.get('instance_type'),
            storage_type=resource.properties.get('storage_type')
        )
        
        if not pricing:
            logger.warning("No pricing available for cost calculation",
                          resource_id=resource.resource_id,
                          service=resource.service)
            return None
        
        logger.info("Pricing retrieved for cost calculation",
                   resource_id=resource.resource_id,
                   service=resource.service,
                   source=pricing.source,
                   hourly_rate=float(pricing.on_demand.amount) if pricing.on_demand else "N/A")
        
        # Calculate costs based on pricing model
        if pricing_model == PricingModel.ON_DEMAND and pricing.on_demand:
            hourly_cost = pricing.on_demand.amount
        else:
            logger.warning("Pricing model not supported or not available",
                          resource_id=resource.resource_id,
                          model=pricing_model,
                          on_demand_available=pricing.on_demand is not None)
            return None
        
        monthly_cost = hourly_cost * Decimal(str(usage_hours))
        annual_cost = monthly_cost * 12
        
        cost_calculation = CostCalculation(
            resource_id=resource.resource_id,
            service_type=resource.service,
            pricing_model=pricing_model,
            current_hourly_cost=hourly_cost,
            current_monthly_cost=monthly_cost,
            current_annual_cost=annual_cost
        )
        
        logger.info("Cost calculation completed",
                   resource_id=resource.resource_id,
                   service=resource.service,
                   pricing_source=pricing.source,
                   hourly_cost=float(hourly_cost),
                   monthly_cost=float(monthly_cost),
                   annual_cost=float(annual_cost))
        
        return cost_calculation
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get pricing cache statistics"""
        total_entries = len(self.pricing_cache)
        valid_entries = sum(1 for cache in self.pricing_cache.values() if cache.is_valid())
        total_hits = sum(cache.hit_count for cache in self.pricing_cache.values())
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "total_cache_hits": total_hits,
            "cache_hit_rate": total_hits / max(total_entries, 1)
        }
