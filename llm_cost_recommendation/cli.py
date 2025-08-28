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
from .utils.logging import configure_logging, get_logger
from .services.pricing_manager import PricingManager
from .agents.coordinator import CoordinatorAgent
from .models import Resource, CloudProvider

logger = get_logger(__name__)


class CostRecommendationApp:
    """Main application class"""

    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        self.config_manager = ConfigManager(config_dir)
        self.data_service = DataIngestionService(data_dir)
        self.llm_service = LLMService(self.config_manager.llm_config)
        self.pricing_manager = PricingManager(self.config_manager)
        self.coordinator = CoordinatorAgent(self.config_manager, self.llm_service)

        logger.info("Application initialized", config_dir=config_dir, data_dir=data_dir)

    async def run_analysis(
        self,
        billing_file: str = None,
        inventory_file: str = None,
        metrics_file: str = None,
        use_sample_data: bool = False,
        individual_processing: bool = False,
    ):
        """Run complete cost analysis"""
        logger.info(
            "Starting cost analysis",
            use_sample_data=use_sample_data,
            individual_processing=individual_processing,
        )

        try:
            # Load or create sample data
            if use_sample_data:
                logger.info("Creating and using sample data")
                self.data_service.create_sample_data()

                billing_file = str(
                    self.data_service.data_dir / "billing" / "sample_billing.csv"
                )
                inventory_file = str(
                    self.data_service.data_dir / "inventory" / "sample_inventory.json"
                )
                metrics_file = str(
                    self.data_service.data_dir / "metrics" / "sample_metrics.csv"
                )

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
            # Determine batch mode (invert individual_processing)
            batch_mode = not individual_processing
            mode_desc = "batched" if batch_mode else "individual"
            
            logger.info(f"Starting coordinator analysis ({mode_desc})")
            report = await self.coordinator.analyze_account(
                resources=resources,
                metrics_data=metrics_data,
                billing_data=billing_data,
                batch_mode=batch_mode,
            )

            # Log summary
            logger.info(
                "Analysis completed",
                total_recommendations=report.total_recommendations,
                monthly_savings=report.total_monthly_savings,
                annual_savings=report.total_annual_savings,
            )

            return report

        except Exception as e:
            logger.error("Analysis failed", error=str(e))
            raise

    async def test_pricing(self):
        """Test pricing module functionality"""
        logger.info("Testing pricing module functionality")
        
        try:
            # Test 1: Validate pricing configuration
            print("\n🔧 Testing pricing configuration...")
            validation_results = await self.pricing_manager.validate_pricing_configuration()
            print(f"Overall status: {validation_results['overall_status']}")
            
            for provider, status in validation_results['providers'].items():
                print(f"  {provider}: {'✅' if status['available'] else '❌'}")
                if status['configuration_loaded']:
                    print(f"    - Configuration loaded: ✅")
                if status['cache_enabled']:
                    print(f"    - Cache enabled: ✅")
            
            if validation_results['issues']:
                print("  Issues found:")
                for issue in validation_results['issues']:
                    print(f"    - {issue}")
            
            # Test 2: Create test resources
            print("\n💰 Testing cost calculations...")
            test_resources = [
                Resource(
                    resource_id="test-ec2-1",
                    service="AWS.EC2",
                    region="us-east-1",
                    account_id="123456789012",
                    cloud_provider="AWS",
                    properties={
                        "instance_type": "t3.medium",
                        "state": "running"
                    }
                ),
                Resource(
                    resource_id="test-s3-1", 
                    service="AWS.S3",
                    region="us-east-1",
                    account_id="123456789012",
                    cloud_provider="AWS",
                    properties={
                        "storage_class": "STANDARD",
                        "size_gb": 100
                    }
                ),
                Resource(
                    resource_id="test-rds-1",
                    service="AWS.RDS", 
                    region="us-east-1",
                    account_id="123456789012",
                    cloud_provider="AWS",
                    properties={
                        "instance_type": "db.t3.micro",
                        "engine": "mysql",
                        "deployment_option": "Single-AZ"
                    }
                )
            ]
            
            # Test 3: Calculate costs for test resources
            for resource in test_resources:
                print(f"\n  Testing {resource.service} ({resource.resource_id})...")
                
                try:
                    cost_calc = await self.pricing_manager.calculate_resource_cost(resource)
                    if cost_calc:
                        print(f"    ✅ Monthly cost: ${cost_calc.current_monthly_cost:.2f}")
                        print(f"    ✅ Annual cost: ${cost_calc.current_annual_cost:.2f}")
                    else:
                        print(f"    ❌ Failed to calculate cost (no pricing data)")
                        
                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
            
            # Test 4: Get pricing cache stats
            print("\n📊 Pricing cache statistics...")
            cache_stats = self.pricing_manager.get_pricing_cache_stats()
            for provider, stats in cache_stats.items():
                print(f"  {provider}:")
                print(f"    - Total entries: {stats['total_entries']}")
                print(f"    - Valid entries: {stats['valid_entries']}")
                print(f"    - Cache hits: {stats['total_cache_hits']}")
                print(f"    - Hit rate: {stats['cache_hit_rate']:.2%}")
            
            # Test 5: Test specific service pricing
            print("\n🎯 Testing specific service pricing...")
            try:
                ec2_pricing = await self.pricing_manager.get_service_pricing(
                    cloud_provider=CloudProvider.AWS,
                    service_type="AWS.EC2",
                    region="us-east-1",
                    instance_type="t3.medium"
                )
                
                if ec2_pricing and ec2_pricing.on_demand:
                    print(f"  ✅ EC2 t3.medium pricing: ${ec2_pricing.on_demand.amount}/hour")
                    print(f"    Source: {ec2_pricing.source}")
                else:
                    print(f"  ❌ Could not retrieve EC2 pricing")
                    
            except Exception as e:
                print(f"  ❌ EC2 pricing error: {str(e)}")
            
            print("\n✅ Pricing module test completed!")
            
        except Exception as e:
            logger.error("Pricing test failed", error=str(e))
            print(f"\n❌ Pricing test failed: {str(e)}")
            raise

    def print_report_summary(self, report):
        """Print a human-readable report summary"""
        print("\n" + "=" * 80)
        print(f"AWS COST OPTIMIZATION REPORT")
        print("=" * 80)
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
            for service, savings in sorted(
                report.savings_by_service.items(), key=lambda x: x[1], reverse=True
            ):
                service_name = service.value if hasattr(service, 'value') else str(service)
                print(f"  {service_name}: ${savings:,.2f}/month")
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
                print(
                    f"\n{i}. {rec.recommendation_type.value.title()} - {rec.service.value}"
                )
                print(f"   Resource: {rec.resource_id}")
                print(f"   Monthly Savings: ${rec.estimated_monthly_savings:,.2f}")
                print(f"   Risk Level: {rec.risk_level.value}")
                print(f"   Rationale: {rec.rationale}")

        print("\n" + "=" * 80)

    def export_report(self, report, output_file: str, format_type: str = "json"):
        """Export report in specified format"""
        import json
        import csv
        from pathlib import Path
        
        try:
            if format_type.lower() == "json":
                # Export full JSON report
                with open(output_file, "w") as f:
                    json.dump(report.model_dump(), f, indent=2, default=str)
                logger.info("JSON report exported", file=output_file)
                
            elif format_type.lower() == "csv":
                # Export CSV summary
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        "Resource ID", "Service", "Recommendation Type", "Risk Level",
                        "Current Cost", "Recommended Cost", "Monthly Savings", "Annual Savings",
                        "Rationale", "Implementation Steps"
                    ])
                    
                    # Write recommendations
                    for rec in report.recommendations:
                        implementation_steps = "; ".join(rec.implementation_steps) if rec.implementation_steps else ""
                        writer.writerow([
                            rec.resource_id,
                            rec.service.value if hasattr(rec.service, 'value') else str(rec.service),
                            rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type),
                            rec.risk_level.value if hasattr(rec.risk_level, 'value') else str(rec.risk_level),
                            f"${rec.current_monthly_cost:.2f}",
                            f"${rec.estimated_monthly_cost:.2f}",
                            f"${rec.estimated_monthly_savings:.2f}",
                            f"${rec.annual_savings:.2f}",
                            rec.rationale[:200] + "..." if len(rec.rationale) > 200 else rec.rationale,
                            implementation_steps[:300] + "..." if len(implementation_steps) > 300 else implementation_steps
                        ])
                logger.info("CSV report exported", file=output_file)
                
            elif format_type.lower() == "excel":
                # Export Excel with multiple sheets
                try:
                    import pandas as pd
                    
                    # Summary sheet data
                    summary_data = {
                        "Metric": [
                            "Total Recommendations",
                            "Monthly Savings",
                            "Annual Savings",
                            "Low Risk Recommendations",
                            "Medium Risk Recommendations", 
                            "High Risk Recommendations"
                        ],
                        "Value": [
                            report.total_recommendations,
                            f"${report.total_monthly_savings:.2f}",
                            f"${report.total_annual_savings:.2f}",
                            len([r for r in report.recommendations if r.risk_level.value == "low"]),
                            len([r for r in report.recommendations if r.risk_level.value == "medium"]),
                            len([r for r in report.recommendations if r.risk_level.value == "high"])
                        ]
                    }
                    
                    # Recommendations sheet data
                    recommendations_data = []
                    for rec in report.recommendations:
                        recommendations_data.append({
                            "Resource ID": rec.resource_id,
                            "Service": rec.service.value if hasattr(rec.service, 'value') else str(rec.service),
                            "Recommendation Type": rec.recommendation_type.value if hasattr(rec.recommendation_type, 'value') else str(rec.recommendation_type),
                            "Risk Level": rec.risk_level.value if hasattr(rec.risk_level, 'value') else str(rec.risk_level),
                            "Current Monthly Cost": rec.current_monthly_cost,
                            "Estimated Monthly Cost": rec.estimated_monthly_cost,
                            "Monthly Savings": rec.estimated_monthly_savings,
                            "Annual Savings": rec.annual_savings,
                            "Confidence Score": rec.confidence_score,
                            "Rationale": rec.rationale,
                            "Impact Description": rec.impact_description,
                            "Implementation Steps": "; ".join(rec.implementation_steps) if rec.implementation_steps else "",
                            "Rollback Plan": rec.rollback_plan
                        })
                    
                    # Create Excel file with multiple sheets
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        # Summary sheet
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Recommendations sheet
                        pd.DataFrame(recommendations_data).to_excel(writer, sheet_name='Recommendations', index=False)
                        
                        # Service breakdown sheet
                        service_data = []
                        for service, savings in report.savings_by_service.items():
                            service_name = service.value if hasattr(service, 'value') else str(service)
                            service_recs = [r for r in report.recommendations if (r.service.value if hasattr(r.service, 'value') else str(r.service)) == service_name]
                            service_data.append({
                                "Service": service_name,
                                "Monthly Savings": savings,
                                "Annual Savings": savings * 12,
                                "Recommendation Count": len(service_recs),
                                "Average Savings per Recommendation": savings / len(service_recs) if service_recs else 0
                            })
                        pd.DataFrame(service_data).to_excel(writer, sheet_name='Service Breakdown', index=False)
                    
                    logger.info("Excel report exported", file=output_file, sheets=3)
                    
                except ImportError:
                    logger.error("pandas and openpyxl are required for Excel export. Install with: pip install pandas openpyxl")
                    # Fallback to CSV
                    csv_file = output_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
                    self.export_report(report, csv_file, "csv")
                    print(f"Excel export not available. Exported CSV instead: {csv_file}")
                    
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error("Export failed", format=format_type, file=output_file, error=str(e))
            raise

    def get_status(self):
        """Get application status"""
        status = {
            "config": {
                "llm_model": self.config_manager.llm_config.model,
                "enabled_services": [
                    s.value
                    for s in self.config_manager.coordinator_config.enabled_services
                ],
            },
            "agents": self.coordinator.get_agent_status(),
        }
        return status


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AWS Cost Optimization using LLM",
        prog="python -m llm_cost_recommendation",
    )
    parser.add_argument("--account-id", help="AWS Account ID to analyze (deprecated - no longer used)")
    parser.add_argument("--billing-file", help="Path to billing CSV file")
    parser.add_argument("--inventory-file", help="Path to inventory JSON file")
    parser.add_argument("--metrics-file", help="Path to metrics CSV file")
    parser.add_argument(
        "--sample-data", action="store_true", help="Use sample data for testing"
    )
    parser.add_argument(
        "--batch-mode", action="store_true", default=True, help="Use batch processing mode (default). Use --no-batch-mode for individual processing"
    )
    parser.add_argument(
        "--no-batch-mode", action="store_true", help="Process resources individually (one by one) instead of batching"
    )
    parser.add_argument(
        "--config-dir", default="config", help="Configuration directory"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-file", help="Output file for detailed report")
    parser.add_argument(
        "--output-format", 
        choices=["json", "csv", "excel"], 
        default="json",
        help="Output format for detailed report (json=full details, csv=summary table, excel=multiple sheets)"
    )
    parser.add_argument("--status", action="store_true", help="Show application status")
    parser.add_argument("--test-pricing", action="store_true", help="Test pricing module functionality")

    # Logging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output (WARNING+ only)"
    )
    parser.add_argument(
        "--log-format",
        choices=["auto", "json", "human"],
        default="auto",
        help="Log output format (auto=detect based on terminal)",
    )

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

        if args.test_pricing:
            # Test pricing module
            await app.test_pricing()
            return

        # Run analysis
        report = await app.run_analysis(
            billing_file=args.billing_file,
            inventory_file=args.inventory_file,
            metrics_file=args.metrics_file,
            use_sample_data=args.sample_data,
            individual_processing=args.no_batch_mode,
        )

        if report:
            # Print summary
            app.print_report_summary(report)

            # Save detailed report if requested
            if args.output_file:
                app.export_report(report, args.output_file, args.output_format)
                print(f"\nDetailed report saved to: {args.output_file}")

    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)
