"""
Command line interface for the LLM cost recommendation system.
"""

import asyncio
import argparse
from pathlib import Path
import sys

from .services.config import ConfigManager
from .services.llm import LLMService
from .services.ingestion import DataIngestionService
from .utils.logging import configure_logging, get_logger
from .agents.coordinator import CoordinatorAgent
from .models import Resource

logger = get_logger(__name__)


class CostRecommendationApp:
    """Main application class"""

    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        self.config_manager = ConfigManager(config_dir)
        self.data_service = DataIngestionService(data_dir)
        self.llm_service = LLMService(self.config_manager.llm_config)
        self.coordinator = CoordinatorAgent(self.config_manager, self.llm_service)

        logger.debug(
            "Application initialized", config_dir=config_dir, data_dir=data_dir
        )

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

            # Execute comprehensive cost optimization analysis
            # Determine processing strategy (batch processing for efficiency vs individual for precision)
            batch_mode = not individual_processing
            processing_strategy = "batched" if batch_mode else "individual"

            logger.info(
                f"Starting comprehensive cost analysis ({processing_strategy} processing)"
            )

            # Generate cost optimization recommendations across all cloud resources
            cost_optimization_report = (
                await self.coordinator.analyze_resources_and_generate_report(
                    resources=resources,
                    metrics_data=metrics_data,
                    billing_data=billing_data,
                    batch_mode=batch_mode,
                )
            )

            # Log analysis completion summary
            logger.info(
                "Cost optimization analysis completed successfully",
                total_recommendations=cost_optimization_report.total_recommendations,
                monthly_savings=cost_optimization_report.total_monthly_savings,
                annual_savings=cost_optimization_report.total_annual_savings,
            )

            return cost_optimization_report

        except Exception as e:
            logger.critical("Analysis failed", error=str(e))
            raise

    def print_report_summary(self, report):
        """Print a human-readable report summary"""
        print("\n" + "=" * 80)
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
                service_name = (
                    service.value if hasattr(service, "value") else str(service)
                )
                print(f"  {service_name}: ${savings:,.2f}/month")
            print()

        # Implementation timeline
        print("IMPLEMENTATION TIMELINE:")
        print(f"  Quick Wins:   {len(report.quick_wins)} recommendations")
        print(f"  Medium Term:  {len(report.medium_term)} recommendations")
        print(f"  Long Term:    {len(report.long_term)} recommendations")
        print()

        # Coverage analysis
        if report.coverage:
            print("ANALYSIS BREAKDOWN:")
            print(f"  Total Resources:     {report.coverage.get('total_resources', 0)}")
            print(f"  Specific Agents:     {report.coverage.get('resources_with_specific_agents', 0)} resources")
            print(f"  Default Agent:       {report.coverage.get('resources_falling_back_to_default', 0)} resources")
            
            # Show fallback resources if any
            fallback_resources = report.coverage.get('resources_falling_back_to_default_list', [])
            if fallback_resources:
                print(f"  Resources using default agent: {', '.join(fallback_resources)}")
            
            # Show fallback services if any
            fallback_services = report.coverage.get('services_falling_back_to_default_list', [])
            if fallback_services:
                print(f"  Services needing specific agents: {', '.join(fallback_services)}")
            
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
        """Export report in specified format
        
        Returns:
            dict: Information about exported files
        """
        import json
        import csv
        from pathlib import Path

        try:
            if format_type.lower() == "json":
                # Export full JSON report
                with open(output_file, "w") as f:
                    json.dump(report.model_dump(), f, indent=2, default=str)
                logger.info("JSON report exported", file=output_file)
                return {"format": "json", "files": [output_file]}

            elif format_type.lower() == "csv":
                # Export multiple CSV files
                from pathlib import Path
                
                base_path = Path(output_file)
                base_name = base_path.stem
                base_dir = base_path.parent
                
                # Summary CSV
                summary_file = base_dir / f"{base_name}_summary.csv"
                with open(summary_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    
                    # Calculate risk distribution
                    low_risk = sum(1 for r in report.recommendations if r.risk_level.value.lower() == "low")
                    medium_risk = sum(1 for r in report.recommendations if r.risk_level.value.lower() == "medium")
                    high_risk = sum(1 for r in report.recommendations if r.risk_level.value.lower() == "high")
                    
                    writer.writerows([
                        ["Total Recommendations", report.total_recommendations],
                        ["Monthly Savings", f"${report.total_monthly_savings:.2f}"],
                        ["Annual Savings", f"${report.total_annual_savings:.2f}"],
                        ["Low Risk Recommendations", low_risk],
                        ["Medium Risk Recommendations", medium_risk],
                        ["High Risk Recommendations", high_risk],
                    ])

                # Recommendations CSV (detailed)
                recommendations_file = base_dir / f"{base_name}_recommendations.csv"
                with open(recommendations_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    # Write header with all fields
                    writer.writerow([
                        "Resource ID", "Service", "Recommendation Type", "Risk Level",
                        "Current Monthly Cost", "Estimated Monthly Cost", "Monthly Savings", 
                        "Annual Savings", "Confidence Score", "Rationale", "Impact Description",
                        "Evidence", "Implementation Steps", "Prerequisites", "Rollback Plan",
                        "Business Hours Impact", "Downtime Required", "SLA Impact"
                    ])

                    # Write recommendations with full data
                    for rec in report.recommendations:
                        implementation_steps = (
                            "; ".join(rec.implementation_steps)
                            if rec.implementation_steps else ""
                        )
                        prerequisites = (
                            "; ".join(rec.prerequisites)
                            if rec.prerequisites else ""
                        )
                        evidence = str(rec.evidence) if rec.evidence else ""
                        
                        writer.writerow([
                            rec.resource_id,
                            rec.service.value if hasattr(rec.service, "value") else str(rec.service),
                            rec.recommendation_type.value if hasattr(rec.recommendation_type, "value") else str(rec.recommendation_type),
                            rec.risk_level.value if hasattr(rec.risk_level, "value") else str(rec.risk_level),
                            f"${rec.current_monthly_cost:.2f}",
                            f"${rec.estimated_monthly_cost:.2f}",
                            f"${rec.estimated_monthly_savings:.2f}",
                            f"${rec.annual_savings:.2f}",
                            rec.confidence_score,
                            rec.rationale,
                            rec.impact_description,
                            evidence,
                            implementation_steps,
                            prerequisites,
                            rec.rollback_plan,
                            rec.business_hours_impact,
                            rec.downtime_required,
                            rec.sla_impact or "",
                        ])

                # Service Breakdown CSV
                service_breakdown_file = base_dir / f"{base_name}_service_breakdown.csv"
                with open(service_breakdown_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Service", "Monthly Savings", "Annual Savings", 
                        "Recommendation Count", "Average Savings per Recommendation"
                    ])
                    
                    for service, savings in report.savings_by_service.items():
                        service_name = service.value if hasattr(service, "value") else str(service)
                        service_recs = [
                            r for r in report.recommendations 
                            if (r.service.value if hasattr(r.service, "value") else str(r.service)) == service_name
                        ]
                        
                        writer.writerow([
                            service_name,
                            f"${savings:.2f}",
                            f"${savings * 12:.2f}",
                            len(service_recs),
                            f"${savings / len(service_recs):.2f}" if service_recs else "$0.00"
                        ])

                logger.info("Multiple CSV reports exported", 
                           summary=str(summary_file),
                           recommendations=str(recommendations_file), 
                           service_breakdown=str(service_breakdown_file))
                
                return {
                    "format": "csv", 
                    "files": [str(summary_file), str(recommendations_file), str(service_breakdown_file)]
                }

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
                            "High Risk Recommendations",
                        ],
                        "Value": [
                            report.total_recommendations,
                            f"${report.total_monthly_savings:.2f}",
                            f"${report.total_annual_savings:.2f}",
                            len(
                                [
                                    r
                                    for r in report.recommendations
                                    if r.risk_level.value == "low"
                                ]
                            ),
                            len(
                                [
                                    r
                                    for r in report.recommendations
                                    if r.risk_level.value == "medium"
                                ]
                            ),
                            len(
                                [
                                    r
                                    for r in report.recommendations
                                    if r.risk_level.value == "high"
                                ]
                            ),
                        ],
                    }

                    # Recommendations sheet data
                    recommendations_data = []
                    for rec in report.recommendations:
                        recommendations_data.append(
                            {
                                "Resource ID": rec.resource_id,
                                "Service": (
                                    rec.service.value
                                    if hasattr(rec.service, "value")
                                    else str(rec.service)
                                ),
                                "Recommendation Type": (
                                    rec.recommendation_type.value
                                    if hasattr(rec.recommendation_type, "value")
                                    else str(rec.recommendation_type)
                                ),
                                "Risk Level": (
                                    rec.risk_level.value
                                    if hasattr(rec.risk_level, "value")
                                    else str(rec.risk_level)
                                ),
                                "Current Monthly Cost": rec.current_monthly_cost,
                                "Estimated Monthly Cost": rec.estimated_monthly_cost,
                                "Monthly Savings": rec.estimated_monthly_savings,
                                "Annual Savings": rec.annual_savings,
                                "Confidence Score": rec.confidence_score,
                                "Rationale": rec.rationale,
                                "Impact Description": rec.impact_description,
                                "Evidence": (
                                    str(rec.evidence) if rec.evidence else ""
                                ),
                                "Implementation Steps": (
                                    "; ".join(rec.implementation_steps)
                                    if rec.implementation_steps
                                    else ""
                                ),
                                "Prerequisites": (
                                    "; ".join(rec.prerequisites)
                                    if rec.prerequisites
                                    else ""
                                ),
                                "Rollback Plan": rec.rollback_plan,
                                "Business Hours Impact": rec.business_hours_impact,
                                "Downtime Required": rec.downtime_required,
                                "SLA Impact": rec.sla_impact or "",
                            }
                        )

                    # Create Excel file with multiple sheets
                    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                        # Summary sheet
                        pd.DataFrame(summary_data).to_excel(
                            writer, sheet_name="Summary", index=False
                        )

                        # Recommendations sheet
                        pd.DataFrame(recommendations_data).to_excel(
                            writer, sheet_name="Recommendations", index=False
                        )

                        # Service breakdown sheet
                        service_data = []
                        for service, savings in report.savings_by_service.items():
                            service_name = (
                                service.value
                                if hasattr(service, "value")
                                else str(service)
                            )
                            service_recs = [
                                r
                                for r in report.recommendations
                                if (
                                    r.service.value
                                    if hasattr(r.service, "value")
                                    else str(r.service)
                                )
                                == service_name
                            ]
                            service_data.append(
                                {
                                    "Service": service_name,
                                    "Monthly Savings": savings,
                                    "Annual Savings": savings * 12,
                                    "Recommendation Count": len(service_recs),
                                    "Average Savings per Recommendation": (
                                        savings / len(service_recs)
                                        if service_recs
                                        else 0
                                    ),
                                }
                            )
                        pd.DataFrame(service_data).to_excel(
                            writer, sheet_name="Service Breakdown", index=False
                        )

                    logger.info("Excel report exported", file=output_file, sheets=3)
                    return {"format": "excel", "files": [output_file]}

                except ImportError:
                    logger.error(
                        "pandas and openpyxl are required for Excel export. Install with: pip install pandas openpyxl"
                    )
                    # Fallback to CSV
                    csv_file = output_file.replace(".xlsx", ".csv").replace(
                        ".xls", ".csv"
                    )
                    csv_result = self.export_report(report, csv_file, "csv")
                    print(
                        f"Excel export not available. Exported CSV instead."
                    )
                    return csv_result

            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            logger.error("Export failed", error=str(e), format=format_type)
            return {"format": format_type, "files": [], "error": str(e)}

    def get_status(self):
        """Get application status"""
        status = {
            "config": {
                "llm_model": self.config_manager.llm_config.model,
                "enabled_services": [
                    s for s in self.config_manager.global_config.enabled_services
                ],
            },
            "agents": self.coordinator.get_agent_status(),
        }
        return status


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Cost Optimization - CLI and API Server",
        prog="python -m llm_cost_recommendation",
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # CLI analysis mode (default)
    cli_parser = subparsers.add_parser("analyze", help="Run cost analysis (default mode)")
    cli_parser.add_argument("--billing-file", help="Path to billing CSV file")
    cli_parser.add_argument("--inventory-file", help="Path to inventory JSON file")
    cli_parser.add_argument("--metrics-file", help="Path to metrics CSV file")
    cli_parser.add_argument(
        "--sample-data", action="store_true", help="Use sample data for testing"
    )
    cli_parser.add_argument(
        "--individual-processing",
        action="store_true",
        help="Process resources individually instead of batch processing (slower but more precise)",
    )
    cli_parser.add_argument("--output-file", help="Output file for detailed report")
    cli_parser.add_argument(
        "--output-format",
        choices=["json", "csv", "excel"],
        default="json",
        help="Output format for detailed report (json=full details, csv=summary table, excel=multiple sheets)",
    )
    cli_parser.add_argument("--status", action="store_true", help="Show application status")
    
    # API server mode
    server_parser = subparsers.add_parser("serve", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    server_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    
    # Common arguments
    for subparser in [cli_parser, server_parser]:
        subparser.add_argument(
            "--config-dir", default="config", help="Configuration directory"
        )
        subparser.add_argument("--data-dir", default="data", help="Data directory")
        
        # Logging options
        subparser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
        )
        subparser.add_argument(
            "--quiet", "-q", action="store_true", help="Reduce output (WARNING+ only)"
        )
        subparser.add_argument(
            "--log-format",
            choices=["auto", "json", "human"],
            default="auto",
            help="Log output format (auto=detect based on terminal)",
        )

    args = parser.parse_args()
    
    # Default to analyze mode if no subcommand specified
    if args.mode is None:
        args.mode = "analyze"

    # Configure logging based on arguments
    if args.quiet:
        log_level = "WARNING"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    configure_logging(level=log_level, format_type=args.log_format, component="cli")

    try:
        if args.mode == "serve":
            # Start API server
            from .api import run_server
            
            logger.info(
                "Starting API server mode",
                host=args.host,
                port=args.port,
                workers=args.workers,
                config_dir=args.config_dir,
                data_dir=args.data_dir,
            )
            
            run_server(
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level=log_level,
                config_dir=args.config_dir,
                data_dir=args.data_dir,
                reload=args.reload,
                log_format=args.log_format,
            )
            
        elif args.mode == "analyze":
            # Run CLI analysis
            app = CostRecommendationApp(args.config_dir, args.data_dir)

            if args.status:
                # Show status
                status = app.get_status()
                import json

                print(json.dumps(status, indent=2))
                return

            # Run analysis
            report = await app.run_analysis(
                billing_file=args.billing_file,
                inventory_file=args.inventory_file,
                metrics_file=args.metrics_file,
                use_sample_data=args.sample_data,
                individual_processing=args.individual_processing,
            )

            if report:
                # Print summary
                app.print_report_summary(report)

                # Save detailed report if requested
                if args.output_file:
                    export_result = app.export_report(report, args.output_file, args.output_format)
                    
                    if export_result.get("error"):
                        print(f"\nExport failed: {export_result['error']}")
                    elif export_result["format"] == "csv" and len(export_result["files"]) > 1:
                        print(f"\nDetailed reports saved to:")
                        for file_path in export_result["files"]:
                            file_name = file_path.split('/')[-1]
                            if "_summary.csv" in file_name:
                                print(f"  Summary: {file_name}")
                            elif "_recommendations.csv" in file_name:
                                print(f"  Recommendations: {file_name}")
                            elif "_service_breakdown.csv" in file_name:
                                print(f"  Service Breakdown: {file_name}")
                    else:
                        print(f"\nDetailed report saved to: {export_result['files'][0] if export_result['files'] else args.output_file}")
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.critical("Application failed", error=str(e))
        sys.exit(1)
