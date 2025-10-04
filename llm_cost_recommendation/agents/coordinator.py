"""
Simplified CoordinatorAgent - Orchestrates analysis using extracted components.
This version is significantly reduced in complexity by delegating to specialized components.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from ..models import (
    Resource,
    Metrics,
    BillingData,
    Recommendation,
    RecommendationReport,
    ServiceType,
)
from ..services.llm import LLMService
from ..services.config import ConfigManager
from ..utils.logging import get_logger

# Import the extracted components
from .agent_manager import AgentManager
from .resource_processor import ResourceProcessor
from .recommendation_processor import RecommendationProcessor
from .report_generator import ReportGenerator

logger = get_logger(__name__)


class CoordinatorAgent:
    """Simplified coordinator that orchestrates analysis using specialized components."""
    
    def __init__(self, config_manager: ConfigManager, llm_service: LLMService):
        self.config_manager = config_manager
        self.llm_service = llm_service
        self.config = config_manager.global_config
        
        # Initialize specialized components
        self.agent_manager = AgentManager(config_manager, llm_service)
        self.resource_processor = ResourceProcessor(config_manager)
        self.recommendation_processor = RecommendationProcessor(config_manager.global_config)
        self.report_generator = ReportGenerator()
        
        logger.info(
            "Simplified coordinator agent initialized",
            enabled_services=len(self.agent_manager.service_agents),
            agents=[agent.value for agent in self.agent_manager.service_agents.keys()],
        )
    
    @property
    def service_agents(self):
        """Access to service agents for backward compatibility with tests."""
        return self.agent_manager.service_agents
    
    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Deduplicate recommendations (backward compatibility with tests)."""
        return self.recommendation_processor._deduplicate_recommendations(recommendations)
    
    def _rank_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Rank recommendations (backward compatibility with tests)."""
        return self.recommendation_processor._rank_recommendations(recommendations)
    
    async def analyze_resources_and_generate_report(
        self,
        resources: List[Resource],
        metrics_data: Dict[str, Metrics] = None,
        billing_data: Dict[str, List[BillingData]] = None,
        batch_mode: bool = True,
    ) -> RecommendationReport:
        """Analyze resources and generate comprehensive report using component delegation."""
        logger.info(
            "Starting coordinated resource analysis",
            total_resources=len(resources),
            batch_mode=batch_mode
        )
        start_time = datetime.now(timezone.utc)
        
        try:
            # Step 1: Process resources using appropriate strategy
            recommendations, resources_without_recs = await self.resource_processor.process_resources(
                resources=resources,
                agent_manager=self.agent_manager,
                metrics_data=metrics_data,
                billing_data=billing_data,
                batch_mode=batch_mode
            )
            
            logger.info(
                "Resource processing completed",
                raw_recommendations=len(recommendations),
                resources_without_recs=len(resources_without_recs)
            )
            
            # Step 2: Post-process recommendations (filter, deduplicate, rank)
            processed_recommendations = self.recommendation_processor.post_process_recommendations(
                recommendations
            )
            
            logger.info(
                "Recommendation post-processing completed",
                final_recommendations=len(processed_recommendations)
            )
            
            # Step 3: Generate comprehensive report
            report = self.report_generator.generate_report(
                recommendations=processed_recommendations,
                resources=resources,
                start_time=start_time,
                resources_without_recommendations=resources_without_recs
            )
            
            # Log final analysis summary
            analysis_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                "Analysis completed successfully",
                report_id=report.id,
                total_recommendations=len(processed_recommendations),
                total_monthly_savings=report.total_monthly_savings,
                analysis_time_seconds=analysis_duration,
                efficiency_resources_per_second=len(resources) / analysis_duration if analysis_duration > 0 else 0
            )
            
            return report
            
        except Exception as e:
            logger.error(
                "Analysis failed with error", 
                error=str(e),
                resources_count=len(resources)
            )
            raise
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all coordinator components."""
        return {
            "coordinator": {
                "enabled_services": [s.value for s in self.config.enabled_services],
                "config": {
                    "similarity_threshold": getattr(self.config, 'similarity_threshold', 0.8),
                    "max_recommendations_per_service": getattr(self.config, 'max_recommendations_per_service', 50),
                    "include_low_impact": getattr(self.config, 'include_low_impact', True),
                },
            },
            "agent_manager": self.agent_manager.get_status(),
            "resource_processor": self.resource_processor.get_processing_stats(),
            "components": {
                "agent_manager": "initialized",
                "resource_processor": "initialized", 
                "recommendation_processor": "initialized",
                "report_generator": "initialized"
            }
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities summary for the coordinator."""
        return {
            "processing_modes": ["batch", "individual"],
            "supported_services": [s.value for s in self.agent_manager.get_available_services()],
            "post_processing_features": [
                "low_impact_filtering",
                "deduplication", 
                "composite_ranking"
            ],
            "report_features": [
                "financial_metrics",
                "risk_distribution",
                "timeline_categorization",
                "coverage_analysis",
                "data_quality_assessment"
            ],
            "components": {
                "agent_manager": self.agent_manager.get_agent_capabilities(),
                "processing_stats": self.resource_processor.get_processing_stats()
            }
        }
    
    def cleanup(self):
        """Cleanup resources used by coordinator components."""
        logger.info("Cleaning up coordinator resources")
        
        try:
            self.agent_manager.cleanup()
            logger.info("Coordinator cleanup completed")
        except Exception as e:
            logger.error(f"Error during coordinator cleanup: {str(e)}")

