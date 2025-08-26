"""
Command line interface for the LLM cost recommendation system.
"""
import asyncio
import argparse
from pathlib import Path
import structlog
import sys

from .services.config import ConfigManager
from .services.llm import LLMService
from .services.ingestion import DataIngestionService
from .services.logging import configure_logging, get_logger
from .agents.coordinator import CoordinatorAgent

logger = get_logger(__name__)


class CostRecommendationApp:
    """Main application class"""
    
    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        self.config_manager = ConfigManager(config_dir)
        self.data_service = DataIngestionService(data_dir)
        self.llm_service = LLMService(self.config_manager.llm_config)
        self.coordinator = CoordinatorAgent(self.config_manager, self.llm_service)
        
        logger.info("Application initialized", 
                   config_dir=config_dir, 
                   data_dir=data_dir)
    
    async def run_analysis(
        self,
        account_id: str,
        billing_file: str = None,
        inventory_file: str = None,
        metrics_file: str = None,
        use_sample_data: bool = False
    ):
        """Run complete cost analysis"""
        logger.info("Starting cost analysis", 
                   account_id=account_id,
                   use_sample_data=use_sample_data)
        
        try:
            # Load or create sample data
            if use_sample_data:
                logger.info("Creating and using sample data")
                self.data_service.create_sample_data()
                
                billing_file = str(self.data_service.data_dir / "billing" / "sample_billing.csv")
                inventory_file = str(self.data_service.data_dir / "inventory" / "sample_inventory.json")
                metrics_file = str(self.data_service.data_dir / "metrics" / "sample_metrics.csv")
            
            # Load data
            resources = []
            metrics_data = {}
            billing_data = {}
            
            if inventory_file and Path(inventory_file).exists():
                logger.info("Loading inventory data", file=inventory_file)
                resources = self.data_service.ingest_inventory_data(inventory_file)
                logger.info("Loaded resources", count=len(resources))
            
            if metrics_file and Path(metrics_file).exists():
                logger.info("Loading metrics data", file=metrics_file)
                metrics_list = self.data_service.ingest_metrics_data(metrics_file)
                metrics_data = {m.resource_id: m for m in metrics_list}
                logger.info("Loaded metrics", count=len(metrics_data))
            
            if billing_file and Path(billing_file).exists():
                logger.info("Loading billing data", file=billing_file)
                billing_list = self.data_service.ingest_billing_data(billing_file)
                
                # Group billing data by resource_id
                from collections import defaultdict
                billing_grouped = defaultdict(list)
                for bill in billing_list:
                    if bill.resource_id:
                        billing_grouped[bill.resource_id].append(bill)
                
                billing_data = dict(billing_grouped)
                logger.info("Loaded billing data", count=len(billing_list))
            
            if not resources:
                logger.error("No resources found to analyze")
                return None
            
            # Run analysis
            logger.info("Starting coordinator analysis")
            report = await self.coordinator.analyze_account(
                account_id=account_id,
                resources=resources,
                metrics_data=metrics_data,
                billing_data=billing_data
            )
            
            # Log summary
            logger.info("Analysis completed", 
                       total_recommendations=report.total_recommendations,
                       monthly_savings=report.total_monthly_savings,
                       annual_savings=report.total_annual_savings)
            
            return report
            
        except Exception as e:
            logger.error("Analysis failed", error=str(e))
            raise
    
    def print_report_summary(self, report):
        """Print a human-readable report summary"""
        print("\n" + "="*80)
        print(f"AWS COST OPTIMIZATION REPORT")
        print("="*80)
        print(f"Account ID: {report.account_id}")
        print(f"Generated: {report.generated_at}")
        print(f"Total Recommendations: {report.total_recommendations}")
        print(f"Monthly Savings: ${report.total_monthly_savings:,.2f}")
        print(f"Annual Savings: ${report.total_annual_savings:,.2f}")
        print()
        
        # Risk distribution
        print("RISK DISTRIBUTION:")
        print(f"  Low Risk:    {report.low_risk_count} recommendations")
        print(f"  Medium Risk: {report.medium_risk_count} recommendations")
        print(f"  High Risk:   {report.high_risk_count} recommendations")
        print()
        
        # Savings by service
        if report.savings_by_service:
            print("SAVINGS BY SERVICE:")
            for service, savings in sorted(report.savings_by_service.items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {service.value}: ${savings:,.2f}/month")
            print()
        
        # Implementation timeline
        print("IMPLEMENTATION TIMELINE:")
        print(f"  Quick Wins:   {len(report.quick_wins)} recommendations")
        print(f"  Medium Term:  {len(report.medium_term)} recommendations")
        print(f"  Long Term:    {len(report.long_term)} recommendations")
        print()
        
        # Top recommendations
        if report.recommendations:
            print("TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"\n{i}. {rec.recommendation_type.value.title()} - {rec.service.value}")
                print(f"   Resource: {rec.resource_id}")
                print(f"   Monthly Savings: ${rec.estimated_monthly_savings:,.2f}")
                print(f"   Risk Level: {rec.risk_level.value}")
                print(f"   Rationale: {rec.rationale[:100]}...")
        
        print("\n" + "="*80)
    
    def get_status(self):
        """Get application status"""
        status = {
            'config': {
                'llm_model': self.config_manager.llm_config.model,
                'enabled_services': [s.value for s in self.config_manager.coordinator_config.enabled_services]
            },
            'agents': self.coordinator.get_agent_status()
        }
        return status


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AWS Cost Optimization using LLM",
        prog="python -m llm_cost_recommendation"
    )
    parser.add_argument("--account-id", help="AWS Account ID to analyze")
    parser.add_argument("--billing-file", help="Path to billing CSV file")
    parser.add_argument("--inventory-file", help="Path to inventory JSON file")
    parser.add_argument("--metrics-file", help="Path to metrics CSV file")
    parser.add_argument("--sample-data", action="store_true", help="Use sample data for testing")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-file", help="Output file for JSON report")
    parser.add_argument("--status", action="store_true", help="Show application status")
    
    # Logging options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output (WARNING+ only)")
    parser.add_argument("--log-format", choices=["auto", "json", "human"], default="auto",
                       help="Log output format (auto=detect based on terminal)")
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.quiet:
        log_level = "WARNING"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    configure_logging(level=log_level, format_type=args.log_format)
    
    try:
        # Initialize application
        app = CostRecommendationApp(args.config_dir, args.data_dir)
        
        if args.status:
            # Show status
            status = app.get_status()
            import json
            print(json.dumps(status, indent=2))
            return
        
        # Validate required arguments for analysis
        if not args.account_id:
            parser.error("--account-id is required for analysis")
        
        # Run analysis
        report = await app.run_analysis(
            account_id=args.account_id,
            billing_file=args.billing_file,
            inventory_file=args.inventory_file,
            metrics_file=args.metrics_file,
            use_sample_data=args.sample_data
        )
        
        if report:
            # Print summary
            app.print_report_summary(report)
            
            # Save JSON report if requested
            if args.output_file:
                import json
                with open(args.output_file, 'w') as f:
                    json.dump(report.model_dump(), f, indent=2, default=str)
                print(f"\nDetailed report saved to: {args.output_file}")
    
    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)
