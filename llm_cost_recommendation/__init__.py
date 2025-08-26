"""
LLM Cost Recommendation System

A multi-agent system for AWS cost optimization using Large Language Models.
"""

from .cli import CostRecommendationApp
from .services.config import ConfigManager
from .services.llm import LLMService
from .services.ingestion import DataIngestionService
from .agents.coordinator import CoordinatorAgent

__version__ = "1.0.0"
__author__ = "AWS Cost Optimization Team"

__all__ = [
    "CostRecommendationApp",
    "ConfigManager",
    "LLMService",
    "DataIngestionService",
    "CoordinatorAgent",
]
