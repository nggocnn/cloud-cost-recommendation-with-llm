"""
Agent Manager - Handles service agent initialization and lifecycle management.
Extracted from CoordinatorAgent to reduce complexity and improve maintainability.
"""

from typing import Dict, Optional, List, Any
from ..models import ServiceType
from ..services.llm import LLMService
from ..services.config import ConfigManager
from ..utils.logging import get_logger
from .base import ServiceAgent

logger = get_logger(__name__)


class AgentManager:
    """Manages service agents and their initialization."""
    
    def __init__(self, config_manager: ConfigManager, llm_service: LLMService):
        self.config_manager = config_manager
        self.llm_service = llm_service
        self.service_agents: Dict[ServiceType, ServiceAgent] = {}
        self._initialize_agents()
        self._validate_default_agent()
    
    def _initialize_agents(self):
        """Initialize service agents based on configuration."""
        logger.info("Initializing service agents")
        
        # Initialize enabled services
        for service in self.config_manager.global_config.enabled_services:
            try:
                agent = self._create_service_agent(service)
                self.service_agents[service] = agent
                
                logger.info(
                    "Service agent initialized",
                    service=service.value,
                    agent_id=agent.agent_id,
                )
                
            except Exception as e:
                logger.error(
                    "Failed to initialize service agent",
                    service=service.value,
                    error=str(e),
                )
        
        # Always try to initialize DEFAULT agent for fallback coverage
        if ServiceType.DEFAULT not in self.service_agents:
            try:
                default_agent = self._create_service_agent(ServiceType.DEFAULT)
                self.service_agents[ServiceType.DEFAULT] = default_agent
                
                logger.info(
                    "Default fallback agent initialized",
                    agent_id=default_agent.agent_id,
                )
                
            except Exception as e:
                logger.warning(
                    "Failed to initialize default fallback agent",
                    error=str(e),
                )
    
    def _create_service_agent(self, service: ServiceType) -> ServiceAgent:
        """Create a service agent for the given service type."""
        agent_config = self.config_manager.service_configs.get(service)
        if not agent_config:
            raise ValueError(f"No configuration found for service: {service}")
            
        return ServiceAgent(
            agent_config=agent_config,
            llm_service=self.llm_service,
            global_config=self.config_manager.global_config,
        )
    
    def _validate_default_agent(self):
        """Ensure default agent is available for system coverage."""
        if ServiceType.DEFAULT not in self.service_agents:
            logger.critical(
                "CRITICAL: Default agent not available - system will have coverage gaps"
            )
            raise RuntimeError(
                "Failed to initialize default agent - system cannot guarantee coverage"
            )
    
    def get_agent_for_service(self, service: ServiceType) -> Optional[ServiceAgent]:
        """Get agent for specific service with fallback to default."""
        agent = self.service_agents.get(service)
        if agent:
            return agent
            
        # Fallback to default agent
        default_agent = self.service_agents.get(ServiceType.DEFAULT)
        if default_agent:
            logger.debug(
                "Using default agent for service",
                service=service.value
            )
            return default_agent
            
        logger.warning(
            "No agent available for service",
            service=service.value
        )
        return None
    
    def get_available_services(self) -> List[ServiceType]:
        """Get list of services with available agents."""
        return list(self.service_agents.keys())
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get capabilities summary for all agents."""
        capabilities = {}
        
        for service, agent in self.service_agents.items():
            try:
                capabilities[service.value] = agent.get_capabilities()
            except Exception as e:
                logger.error(
                    "Failed to get capabilities for agent",
                    service=service.value,
                    error=str(e)
                )
                capabilities[service.value] = {"error": str(e)}
        
        return capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all service agents."""
        return {
            "total_agents": len(self.service_agents),
            "enabled_services": [s.value for s in self.service_agents.keys()],
            "default_agent_available": ServiceType.DEFAULT in self.service_agents,
            "agent_capabilities": self.get_agent_capabilities(),
        }
    
    def cleanup(self):
        """Cleanup resources used by agents."""
        logger.info("Cleaning up service agents")
        
        for service, agent in self.service_agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
            except Exception as e:
                logger.error(
                    "Error cleaning up agent",
                    service=service.value,
                    error=str(e)
                )