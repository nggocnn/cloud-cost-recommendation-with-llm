"""
Recommendation Processor - Handles recommendation post-processing.
Extracted from CoordinatorAgent to focus on deduplication, filtering, and ranking.
"""

from typing import List
from collections import defaultdict

from ..models import Recommendation, RiskLevel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationProcessor:
    """Handles recommendation post-processing operations."""
    
    def __init__(self, config):
        self.config = config
        self._similarity_threshold = getattr(config, 'similarity_threshold', 0.8)
        self._include_low_impact = getattr(config, 'include_low_impact', True)
        
        # Weights for composite scoring
        self._savings_weight = getattr(config, 'savings_weight', 0.4)
        self._risk_weight = getattr(config, 'risk_weight', 0.3)
        self._confidence_weight = getattr(config, 'confidence_weight', 0.2)
        self._implementation_weight = getattr(config, 'implementation_weight', 0.1)
    
    def post_process_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Post-process recommendations: filter, deduplicate, and rank."""
        if not recommendations:
            logger.info("No recommendations to post-process")
            return []
        
        logger.info(
            "Starting recommendation post-processing",
            initial_count=len(recommendations)
        )
        
        # Step 1: Filter low-impact recommendations if configured
        filtered = self._filter_low_impact_recommendations(recommendations)
        
        # Step 2: Deduplicate similar recommendations
        deduplicated = self._deduplicate_recommendations(filtered)
        
        # Step 3: Rank by composite score
        ranked = self._rank_recommendations(deduplicated)
        
        logger.info(
            "Recommendation post-processing completed",
            initial_count=len(recommendations),
            after_filtering=len(filtered),
            after_deduplication=len(deduplicated),
            final_count=len(ranked)
        )
        
        return ranked
    
    def _filter_low_impact_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Filter out low-impact recommendations if configured."""
        if self._include_low_impact:
            logger.debug("Including all recommendations (low impact allowed)")
            return recommendations
        
        filtered = []
        filtered_out_count = 0
        
        for rec in recommendations:
            if self._should_include_recommendation(rec):
                filtered.append(rec)
            else:
                filtered_out_count += 1
                logger.debug(
                    "Filtered out low-impact recommendation",
                    resource_id=rec.resource_id,
                    savings=rec.estimated_monthly_savings
                )
        
        logger.info(
            "Low-impact filtering completed",
            kept=len(filtered),
            filtered_out=filtered_out_count
        )
        
        return filtered
    
    def _should_include_recommendation(self, rec: Recommendation) -> bool:
        """Determine if recommendation should be included based on impact."""
        # Include if significant savings
        if rec.estimated_monthly_savings >= 10.0:
            return True
        
        # Include if it's a data quality or error recommendation
        if rec.warnings:
            for warning in rec.warnings:
                if any(keyword in warning for keyword in [
                    "DATA QUALITY ISSUE",
                    "DEGRADED RECOMMENDATION", 
                    "CONVERSION ERROR"
                ]):
                    return True
        
        # Filter out if low impact and no special conditions
        return False
    
    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations by keeping best savings per resource."""
        if not recommendations:
            return []
        
        logger.debug("Starting deduplication", initial_count=len(recommendations))
        
        # Group by resource_id
        by_resource = defaultdict(list)
        for rec in recommendations:
            by_resource[rec.resource_id].append(rec)
        
        deduplicated = []
        duplicates_removed = 0
        
        for resource_id, resource_recs in by_resource.items():
            if len(resource_recs) == 1:
                # Single recommendation, keep it
                deduplicated.extend(resource_recs)
            else:
                # Multiple recommendations, keep the best one
                best_rec = self._select_best_recommendation(resource_recs)
                deduplicated.append(best_rec)
                duplicates_removed += len(resource_recs) - 1
                
                logger.debug(
                    "Deduplicated recommendations for resource",
                    resource_id=resource_id,
                    original_count=len(resource_recs),
                    kept_recommendation_id=best_rec.id,
                    kept_savings=best_rec.estimated_monthly_savings
                )
        
        logger.info(
            "Deduplication completed",
            duplicates_removed=duplicates_removed,
            final_count=len(deduplicated)
        )
        
        return deduplicated
    
    def _select_best_recommendation(self, recommendations: List[Recommendation]) -> Recommendation:
        """Select the best recommendation from a list of candidates."""
        # Primary criteria: highest monthly savings
        best_by_savings = max(recommendations, key=lambda r: r.estimated_monthly_savings)
        
        # If there's a clear winner by savings (>20% better), use it
        max_savings = best_by_savings.estimated_monthly_savings
        threshold_savings = max_savings * 0.8  # 20% threshold
        
        top_candidates = [
            rec for rec in recommendations 
            if rec.estimated_monthly_savings >= threshold_savings
        ]
        
        if len(top_candidates) == 1:
            return top_candidates[0]
        
        # If multiple candidates are close in savings, consider other factors
        return max(top_candidates, key=self._calculate_recommendation_score)
    
    def _rank_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Rank recommendations by composite score."""
        if not recommendations:
            return []
        
        logger.debug("Starting recommendation ranking", count=len(recommendations))
        
        # Calculate scores for all recommendations
        scored_recommendations = [
            (self._calculate_recommendation_score(rec), rec) 
            for rec in recommendations
        ]
        
        # Sort by score (descending)
        scored_recommendations.sort(key=lambda x: x[0], reverse=True)
        
        # Extract ranked recommendations
        ranked = [rec for _, rec in scored_recommendations]
        
        # Log top recommendations for debugging
        if ranked:
            top_rec = ranked[0]
            logger.info(
                "Top recommendation after ranking",
                resource_id=top_rec.resource_id,
                savings=top_rec.estimated_monthly_savings,
                risk_level=top_rec.risk_level.value if hasattr(top_rec.risk_level, 'value') else str(top_rec.risk_level),
                confidence=top_rec.confidence_score
            )
        
        return ranked
    
    def _calculate_recommendation_score(self, rec: Recommendation) -> float:
        """Calculate composite score for recommendation ranking."""
        # Normalize savings (0-1 scale, capped at $1000/month = 1.0)
        savings_score = min(rec.estimated_monthly_savings / 1000.0, 1.0)
        
        # Risk scores (lower risk = higher score)
        risk_scores = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.2,
        }
        risk_score = risk_scores.get(rec.risk_level, 0.5)
        
        # Confidence score (already 0-1)
        confidence_score = rec.confidence_score
        
        # Implementation ease (fewer steps = easier = higher score)
        max_steps = 10  # Reasonable maximum
        impl_steps = len(rec.implementation_steps) if rec.implementation_steps else 1
        implementation_score = max(1.0 - (impl_steps / max_steps), 0.1)
        
        # Calculate weighted composite score
        composite_score = (
            savings_score * self._savings_weight +
            risk_score * self._risk_weight + 
            confidence_score * self._confidence_weight +
            implementation_score * self._implementation_weight
        )
        
        return composite_score
    
    def get_processing_statistics(self, original_count: int, final_count: int) -> dict:
        """Get statistics about the post-processing operations."""
        return {
            "original_recommendations": original_count,
            "final_recommendations": final_count,
            "reduction_percentage": ((original_count - final_count) / original_count * 100) if original_count > 0 else 0,
            "filtering_enabled": not self._include_low_impact,
            "similarity_threshold": self._similarity_threshold,
            "scoring_weights": {
                "savings": self._savings_weight,
                "risk": self._risk_weight,
                "confidence": self._confidence_weight,
                "implementation": self._implementation_weight
            }
        }