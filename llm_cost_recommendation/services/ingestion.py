"""
Data ingestion service for AWS cost and usage data.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import structlog

from ..models import Resource, Metrics, BillingData, ServiceType
from .logging import get_logger

logger = get_logger(__name__)


class DataIngestionService:
    """Service for ingesting and normalizing AWS data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Create subdirectories for different data types
        (self.data_dir / "billing").mkdir(exist_ok=True)
        (self.data_dir / "inventory").mkdir(exist_ok=True)
        (self.data_dir / "metrics").mkdir(exist_ok=True)

    def ingest_billing_data(self, csv_file_path: str) -> List[BillingData]:
        """Ingest billing data from CSV file"""
        logger.info("Ingesting billing data", file_path=csv_file_path)

        try:
            df = pd.read_csv(csv_file_path)
            billing_records = []

            # Required columns mapping
            column_mapping = {
                "bill_period_start": "bill/BillingPeriodStartDate",
                "bill_period_end": "bill/BillingPeriodEndDate",
                "account_id": "lineItem/UsageAccountId",
                "service": "product/ProductName",
                "resource_id": "lineItem/ResourceId",
                "region": "product/region",
                "usage_type": "lineItem/UsageType",
                "usage_amount": "lineItem/UsageAmount",
                "usage_unit": "lineItem/UsageUnit",
                "unblended_cost": "lineItem/UnblendedCost",
                "amortized_cost": "lineItem/NetAmortizedCost",
            }

            # Check if columns exist, try alternative names
            actual_columns = {}
            for key, preferred_col in column_mapping.items():
                if preferred_col in df.columns:
                    actual_columns[key] = preferred_col
                else:
                    # Try to find similar column names
                    for col in df.columns:
                        if (
                            key.lower() in col.lower()
                            or preferred_col.split("/")[-1].lower() in col.lower()
                        ):
                            actual_columns[key] = col
                            break

            logger.info("Column mapping", mapping=actual_columns)

            for _, row in df.iterrows():
                try:
                    # Extract tags from columns starting with resourceTags/
                    tags = {}
                    for col in df.columns:
                        if col.startswith("resourceTags/"):
                            tag_key = col.replace("resourceTags/", "")
                            if pd.notna(row[col]):
                                tags[tag_key] = str(row[col])

                    billing_record = BillingData(
                        account_id=str(
                            row.get(actual_columns.get("account_id", ""), "")
                        ),
                        service=str(row.get(actual_columns.get("service", ""), "")),
                        resource_id=(
                            str(row.get(actual_columns.get("resource_id", ""), ""))
                            if pd.notna(row.get(actual_columns.get("resource_id", "")))
                            else None
                        ),
                        region=str(row.get(actual_columns.get("region", ""), "")),
                        usage_type=str(
                            row.get(actual_columns.get("usage_type", ""), "")
                        ),
                        usage_amount=float(
                            row.get(actual_columns.get("usage_amount", ""), 0)
                        ),
                        usage_unit=str(
                            row.get(actual_columns.get("usage_unit", ""), "")
                        ),
                        unblended_cost=float(
                            row.get(actual_columns.get("unblended_cost", ""), 0)
                        ),
                        amortized_cost=float(
                            row.get(actual_columns.get("amortized_cost", ""), 0)
                        ),
                        bill_period_start=pd.to_datetime(
                            row.get(actual_columns.get("bill_period_start", ""))
                        ),
                        bill_period_end=pd.to_datetime(
                            row.get(actual_columns.get("bill_period_end", ""))
                        ),
                        tags=tags,
                    )

                    billing_records.append(billing_record)

                except Exception as e:
                    logger.warning(
                        "Failed to parse billing record", row_index=_, error=str(e)
                    )
                    continue

            logger.info("Billing data ingested", total_records=len(billing_records))
            return billing_records

        except Exception as e:
            logger.error("Failed to ingest billing data", error=str(e))
            raise

    def ingest_inventory_data(self, json_file_path: str) -> List[Resource]:
        """Ingest resource inventory from JSON file"""
        logger.info("Ingesting inventory data", file_path=json_file_path)

        try:
            with open(json_file_path, "r") as f:
                inventory_data = json.load(f)

            resources = []

            for item in inventory_data:
                try:
                    # Normalize service name to ServiceType
                    service_name = item.get("service", "").upper()
                    service_type = self._normalize_service_type(service_name)

                    if not service_type:
                        logger.warning("Unknown service type", service=service_name)
                        continue

                    resource = Resource(
                        resource_id=item.get("resource_id", ""),
                        service=service_type,
                        region=item.get("region", ""),
                        availability_zone=item.get("availability_zone"),
                        account_id=item.get("account_id", ""),
                        tags=item.get("tags", {}),
                        properties=item.get("properties", {}),
                        created_at=(
                            pd.to_datetime(item.get("created_at"))
                            if item.get("created_at")
                            else None
                        ),
                        extensions=item.get("extensions", {}),
                    )

                    resources.append(resource)

                except Exception as e:
                    logger.warning(
                        "Failed to parse inventory record", item=item, error=str(e)
                    )
                    continue

            logger.info("Inventory data ingested", total_resources=len(resources))
            return resources

        except Exception as e:
            logger.error("Failed to ingest inventory data", error=str(e))
            raise

    def ingest_metrics_data(self, csv_file_path: str) -> List[Metrics]:
        """Ingest metrics data from CSV file"""
        logger.info("Ingesting metrics data", file_path=csv_file_path)

        try:
            df = pd.read_csv(csv_file_path)
            metrics_records = []

            for _, row in df.iterrows():
                try:
                    # Extract metrics columns (those that are numeric and not metadata)
                    metrics_dict = {}
                    for col in df.columns:
                        if col not in [
                            "resource_id",
                            "timestamp",
                            "period_days",
                        ] and pd.notna(row[col]):
                            try:
                                metrics_dict[col] = float(row[col])
                            except (ValueError, TypeError):
                                continue

                    metrics_record = Metrics(
                        resource_id=str(row.get("resource_id", "")),
                        timestamp=pd.to_datetime(
                            row.get("timestamp", datetime.utcnow())
                        ),
                        metrics=metrics_dict,
                        period_days=int(row.get("period_days", 30)),
                        # Extract common metrics
                        cpu_utilization_p50=row.get("cpu_utilization_p50"),
                        cpu_utilization_p90=row.get("cpu_utilization_p90"),
                        cpu_utilization_p95=row.get("cpu_utilization_p95"),
                        memory_utilization_p50=row.get("memory_utilization_p50"),
                        memory_utilization_p90=row.get("memory_utilization_p90"),
                        memory_utilization_p95=row.get("memory_utilization_p95"),
                        iops_read=row.get("iops_read"),
                        iops_write=row.get("iops_write"),
                        throughput_read=row.get("throughput_read"),
                        throughput_write=row.get("throughput_write"),
                        network_in=row.get("network_in"),
                        network_out=row.get("network_out"),
                        is_idle=bool(row.get("is_idle", False)),
                    )

                    metrics_records.append(metrics_record)

                except Exception as e:
                    logger.warning(
                        "Failed to parse metrics record", row_index=_, error=str(e)
                    )
                    continue

            logger.info("Metrics data ingested", total_records=len(metrics_records))
            return metrics_records

        except Exception as e:
            logger.error("Failed to ingest metrics data", error=str(e))
            raise

    def _normalize_service_type(self, service_name: str) -> Optional[ServiceType]:
        """Normalize service name to ServiceType enum"""
        service_mapping = {
            "EC2": ServiceType.EC2,
            "ELASTIC COMPUTE CLOUD": ServiceType.EC2,
            "AMAZON ELASTIC COMPUTE CLOUD": ServiceType.EC2,
            "EBS": ServiceType.EBS,
            "ELASTIC BLOCK STORE": ServiceType.EBS,
            "AMAZON ELASTIC BLOCK STORE": ServiceType.EBS,
            "S3": ServiceType.S3,
            "SIMPLE STORAGE SERVICE": ServiceType.S3,
            "AMAZON SIMPLE STORAGE SERVICE": ServiceType.S3,
            "EFS": ServiceType.EFS,
            "ELASTIC FILE SYSTEM": ServiceType.EFS,
            "AMAZON ELASTIC FILE SYSTEM": ServiceType.EFS,
            "RDS": ServiceType.RDS,
            "RELATIONAL DATABASE SERVICE": ServiceType.RDS,
            "AMAZON RELATIONAL DATABASE SERVICE": ServiceType.RDS,
            "DYNAMODB": ServiceType.DYNAMODB,
            "AMAZON DYNAMODB": ServiceType.DYNAMODB,
            "LAMBDA": ServiceType.LAMBDA,
            "AWS LAMBDA": ServiceType.LAMBDA,
            "CLOUDFRONT": ServiceType.CLOUDFRONT,
            "AMAZON CLOUDFRONT": ServiceType.CLOUDFRONT,
            "ELASTIC LOAD BALANCING": ServiceType.ALB,
            "APPLICATION LOAD BALANCER": ServiceType.ALB,
            "NETWORK LOAD BALANCER": ServiceType.NLB,
            "GATEWAY LOAD BALANCER": ServiceType.GWLB,
            "ELASTIC IP": ServiceType.ELASTIC_IP,
            "NAT GATEWAY": ServiceType.NAT_GATEWAY,
            "VPC ENDPOINT": ServiceType.VPC_ENDPOINTS,
            "SQS": ServiceType.SQS,
            "SIMPLE QUEUE SERVICE": ServiceType.SQS,
            "AMAZON SIMPLE QUEUE SERVICE": ServiceType.SQS,
            "SNS": ServiceType.SNS,
            "SIMPLE NOTIFICATION SERVICE": ServiceType.SNS,
            "AMAZON SIMPLE NOTIFICATION SERVICE": ServiceType.SNS,
        }

        return service_mapping.get(service_name.upper())

    def create_sample_data(self):
        """Create sample data files for testing"""
        logger.info("Creating sample data files")

        # Create sample billing data
        billing_data = [
            {
                "bill/BillingPeriodStartDate": "2024-01-01",
                "bill/BillingPeriodEndDate": "2024-01-31",
                "lineItem/UsageAccountId": "123456789012",
                "product/ProductName": "Amazon Elastic Compute Cloud",
                "lineItem/ResourceId": "i-1234567890abcdef0",
                "product/region": "us-west-2",
                "lineItem/UsageType": "BoxUsage:m5.large",
                "lineItem/UsageAmount": 744.0,
                "lineItem/UsageUnit": "Hrs",
                "lineItem/UnblendedCost": 89.28,
                "lineItem/NetAmortizedCost": 89.28,
                "resourceTags/Environment": "production",
                "resourceTags/Owner": "team-alpha",
            },
            {
                "bill/BillingPeriodStartDate": "2024-01-01",
                "bill/BillingPeriodEndDate": "2024-01-31",
                "lineItem/UsageAccountId": "123456789012",
                "product/ProductName": "Amazon Elastic Block Store",
                "lineItem/ResourceId": "vol-1234567890abcdef0",
                "product/region": "us-west-2",
                "lineItem/UsageType": "VolumeUsage.gp3",
                "lineItem/UsageAmount": 100.0,
                "lineItem/UsageUnit": "GB-Mo",
                "lineItem/UnblendedCost": 8.00,
                "lineItem/NetAmortizedCost": 8.00,
                "resourceTags/Environment": "production",
            },
        ]

        billing_df = pd.DataFrame(billing_data)
        billing_df.to_csv(self.data_dir / "billing" / "sample_billing.csv", index=False)

        # Create sample inventory data
        inventory_data = [
            {
                "resource_id": "i-1234567890abcdef0",
                "service": "EC2",
                "region": "us-west-2",
                "availability_zone": "us-west-2a",
                "account_id": "123456789012",
                "tags": {"Environment": "production", "Owner": "team-alpha"},
                "properties": {
                    "instance_type": "m5.large",
                    "state": "running",
                    "cpu_count": 2,
                    "memory_gb": 8,
                    "storage_gb": 20,
                },
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "resource_id": "vol-1234567890abcdef0",
                "service": "EBS",
                "region": "us-west-2",
                "availability_zone": "us-west-2a",
                "account_id": "123456789012",
                "tags": {"Environment": "production"},
                "properties": {
                    "volume_type": "gp3",
                    "size_gb": 100,
                    "iops": 3000,
                    "throughput": 125,
                    "state": "in-use",
                },
            },
        ]

        with open(self.data_dir / "inventory" / "sample_inventory.json", "w") as f:
            json.dump(inventory_data, f, indent=2)

        # Create sample metrics data
        metrics_data = [
            {
                "resource_id": "i-1234567890abcdef0",
                "timestamp": "2024-01-31T23:59:59Z",
                "period_days": 30,
                "cpu_utilization_p50": 15.2,
                "cpu_utilization_p90": 28.5,
                "cpu_utilization_p95": 35.1,
                "memory_utilization_p50": 45.8,
                "memory_utilization_p90": 62.3,
                "memory_utilization_p95": 71.2,
                "network_in": 1024000,
                "network_out": 2048000,
                "is_idle": False,
            },
            {
                "resource_id": "vol-1234567890abcdef0",
                "timestamp": "2024-01-31T23:59:59Z",
                "period_days": 30,
                "iops_read": 150.5,
                "iops_write": 75.2,
                "throughput_read": 15.5,
                "throughput_write": 8.2,
                "is_idle": False,
            },
        ]

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(self.data_dir / "metrics" / "sample_metrics.csv", index=False)

        logger.info(
            "Sample data files created",
            billing_file=str(self.data_dir / "billing" / "sample_billing.csv"),
            inventory_file=str(self.data_dir / "inventory" / "sample_inventory.json"),
            metrics_file=str(self.data_dir / "metrics" / "sample_metrics.csv"),
        )

    def get_resource_data(self, resource_id: str) -> Dict[str, Any]:
        """Get combined data for a specific resource"""
        # This would typically query a database or combine data from multiple sources
        # For now, return a placeholder implementation
        return {
            "resource_id": resource_id,
            "billing_data": [],
            "inventory_data": None,
            "metrics_data": [],
        }
