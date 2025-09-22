"""
Data ingestion service for AWS cost and usage data.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from ..models import Resource, Metrics, BillingData, ServiceType
from ..utils.logging import get_logger

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

    def _parse_json_field(self, field_value):
        """Parse JSON field from CSV"""
        import json
        if not field_value or pd.isna(field_value) or field_value == '':
            return {}
        try:
            if isinstance(field_value, str):
                return json.loads(field_value)
            return field_value
        except (json.JSONDecodeError, TypeError):
            return {}

    def _parse_list_field(self, field_value):
        """Parse list field from CSV (comma-separated or JSON)"""
        import json
        if not field_value or pd.isna(field_value) or field_value == '':
            return []
        try:
            if isinstance(field_value, str):
                # Try JSON first
                if field_value.startswith('[') and field_value.endswith(']'):
                    return json.loads(field_value)
                # Otherwise, comma-separated
                return [int(x.strip()) for x in field_value.split(',') if x.strip().isdigit()]
            return field_value if isinstance(field_value, list) else []
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def _safe_string_field(self, field_value):
        """Safely handle string fields that might be NaN"""
        if pd.isna(field_value) or field_value == '':
            return None
        return str(field_value)

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

            logger.debug("Column mapping", mapping=actual_columns)

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
                        cpu_utilization_min=row.get("cpu_utilization_min"),
                        cpu_utilization_max=row.get("cpu_utilization_max"),
                        cpu_utilization_stddev=row.get("cpu_utilization_stddev"),
                        cpu_utilization_trend=self._safe_string_field(row.get("cpu_utilization_trend")),
                        cpu_utilization_volatility=self._safe_string_field(row.get("cpu_utilization_volatility")),
                        cpu_utilization_peak_hours=self._parse_list_field(row.get("cpu_utilization_peak_hours", "")),
                        cpu_utilization_patterns=self._parse_json_field(row.get("cpu_utilization_patterns", "{}")),
                        cpu_timeseries_data=self._parse_json_field(row.get("cpu_timeseries_data", "[]")),
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

    def _normalize_service_type(
        self, service_name: str
    ) -> Optional[Union[ServiceType.AWS, ServiceType.Azure, ServiceType.GCP]]:
        """Normalize service name to ServiceType enum"""
        service_mapping = {
            "EC2": ServiceType.AWS.EC2,
            "ELASTIC COMPUTE CLOUD": ServiceType.AWS.EC2,
            "AMAZON ELASTIC COMPUTE CLOUD": ServiceType.AWS.EC2,
            "EBS": ServiceType.AWS.EBS,
            "ELASTIC BLOCK STORE": ServiceType.AWS.EBS,
            "AMAZON ELASTIC BLOCK STORE": ServiceType.AWS.EBS,
            "S3": ServiceType.AWS.S3,
            "SIMPLE STORAGE SERVICE": ServiceType.AWS.S3,
            "AMAZON SIMPLE STORAGE SERVICE": ServiceType.AWS.S3,
            "EFS": ServiceType.AWS.EFS,
            "ELASTIC FILE SYSTEM": ServiceType.AWS.EFS,
            "AMAZON ELASTIC FILE SYSTEM": ServiceType.AWS.EFS,
            "RDS": ServiceType.AWS.RDS,
            "RELATIONAL DATABASE SERVICE": ServiceType.AWS.RDS,
            "AMAZON RELATIONAL DATABASE SERVICE": ServiceType.AWS.RDS,
            "DYNAMODB": ServiceType.AWS.DYNAMODB,
            "AMAZON DYNAMODB": ServiceType.AWS.DYNAMODB,
            "LAMBDA": ServiceType.AWS.LAMBDA,
            "AWS LAMBDA": ServiceType.AWS.LAMBDA,
            "CLOUDFRONT": ServiceType.AWS.CLOUDFRONT,
            "AMAZON CLOUDFRONT": ServiceType.AWS.CLOUDFRONT,
            "ALB": ServiceType.AWS.ALB,
            "ELASTIC LOAD BALANCING": ServiceType.AWS.ALB,
            "APPLICATION LOAD BALANCER": ServiceType.AWS.ALB,
            "NETWORK LOAD BALANCER": ServiceType.AWS.NLB,
            "GATEWAY LOAD BALANCER": ServiceType.AWS.GWLB,
            "ELASTIC IP": ServiceType.AWS.ELASTIC_IP,
            "NATGATEWAY": ServiceType.AWS.NAT_GATEWAY,
            "NAT GATEWAY": ServiceType.AWS.NAT_GATEWAY,
            "AMAZON VIRTUAL PRIVATE CLOUD": ServiceType.AWS.NAT_GATEWAY,
            "VPC ENDPOINT": ServiceType.AWS.VPC_ENDPOINTS,
            "SQS": ServiceType.AWS.SQS,
            "SIMPLE QUEUE SERVICE": ServiceType.AWS.SQS,
            "AMAZON SIMPLE QUEUE SERVICE": ServiceType.AWS.SQS,
            "SNS": ServiceType.AWS.SNS,
            "SIMPLE NOTIFICATION SERVICE": ServiceType.AWS.SNS,
            "AMAZON SIMPLE NOTIFICATION SERVICE": ServiceType.AWS.SNS,
        }

        return service_mapping.get(service_name.upper())

    def create_sample_data(self, num_resources: int = 200):
        """Create realistic AWS data files for comprehensive testing"""
        logger.info("Creating realistic AWS data files", num_resources=num_resources)

        import random
        from datetime import datetime, timedelta

        # Realistic AWS service configurations
        aws_services = {
            "Amazon Elastic Compute Cloud": {
                "service_name": "EC2",
                "usage_types": [
                    "BoxUsage:t3.micro",
                    "BoxUsage:t3.small",
                    "BoxUsage:t3.medium",
                    "BoxUsage:t3.large",
                    "BoxUsage:t3.xlarge",
                    "BoxUsage:m5.large",
                    "BoxUsage:m5.xlarge",
                    "BoxUsage:c5.large",
                    "BoxUsage:r5.large",
                ],
                "cost_range": (10, 500),
                "usage_range": (100, 744),  # hours per month
            },
            "Amazon Elastic Block Store": {
                "service_name": "EBS",
                "usage_types": [
                    "VolumeUsage.gp3",
                    "VolumeUsage.gp2",
                    "VolumeUsage.io2",
                ],
                "cost_range": (5, 200),
                "usage_range": (50, 2000),  # GB-Mo
            },
            "Amazon Simple Storage Service": {
                "service_name": "S3",
                "usage_types": [
                    "StorageUsage.StandardStorage",
                    "StorageUsage.IntelligentTieringFAStorage",
                    "StorageUsage.StandardIAStorage",
                    "StorageUsage.OneZoneIAStorage",
                ],
                "cost_range": (1, 1000),
                "usage_range": (10, 50000),  # GB-Mo
            },
            "Amazon Relational Database Service": {
                "service_name": "RDS",
                "usage_types": [
                    "InstanceUsage:db.t3.micro",
                    "InstanceUsage:db.t3.small",
                    "InstanceUsage:db.r5.large",
                    "InstanceUsage:db.r5.xlarge",
                ],
                "cost_range": (20, 800),
                "usage_range": (100, 744),
            },
            "AWS Lambda": {
                "service_name": "Lambda",
                "usage_types": ["Request-ARM", "Duration-ARM", "Request", "Duration"],
                "cost_range": (0.1, 50),
                "usage_range": (1000, 1000000),
            },
            "Amazon CloudFront": {
                "service_name": "CloudFront",
                "usage_types": [
                    "DataTransfer-Out-Bytes",
                    "Requests-HTTP",
                    "Requests-HTTPS",
                ],
                "cost_range": (5, 300),
                "usage_range": (1000, 100000),
            },
            "Amazon DynamoDB": {
                "service_name": "DynamoDB",
                "usage_types": [
                    "StorageUsage",
                    "ReadCapacityUnit-Hrs",
                    "WriteCapacityUnit-Hrs",
                ],
                "cost_range": (2, 150),
                "usage_range": (100, 10000),
            },
        }

        environments = ["production", "staging", "development", "testing"]
        teams = ["team-alpha", "team-beta", "team-gamma", "team-delta", "team-epsilon"]
        regions = [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "ap-southeast-1",
            "ap-northeast-1",
        ]
        accounts = ["123456789012", "234567890123", "345678901234", "456789012345"]

        # Generate billing data
        billing_records = []
        inventory_records = []
        metrics_records = []

        for i in range(num_resources):
            # Pick random service
            service_name, service_config = random.choice(list(aws_services.items()))
            usage_type = random.choice(service_config["usage_types"])

            # Generate resource ID based on service
            if "EC2" in service_name:
                resource_id = (
                    f"i-{random.randint(100000000000000, 999999999999999):015x}"
                )
            elif "EBS" in service_name:
                resource_id = (
                    f"vol-{random.randint(100000000000000, 999999999999999):015x}"
                )
            elif "RDS" in service_name:
                resource_id = f"db-{random.choice(['mysql', 'postgres', 'aurora'])}-{random.randint(1000, 9999)}"
            elif "Lambda" in service_name:
                resource_id = f"function-{random.choice(['api', 'processor', 'handler'])}-{random.randint(100, 999)}"
            else:
                resource_id = f"{service_config['service_name'].lower()}-{random.randint(100000, 999999)}"

            # Generate realistic costs and usage
            cost = round(random.uniform(*service_config["cost_range"]), 2)
            usage = round(random.uniform(*service_config["usage_range"]), 2)

            # Random metadata
            environment = random.choice(environments)
            team = random.choice(teams)
            region = random.choice(regions)
            account = random.choice(accounts)

            # Billing record
            billing_record = {
                "bill/BillingPeriodStartDate": "2024-01-01",
                "bill/BillingPeriodEndDate": "2024-01-31",
                "lineItem/UsageAccountId": account,
                "product/ProductName": service_name,
                "lineItem/ResourceId": resource_id,
                "product/region": region,
                "lineItem/UsageType": usage_type,
                "lineItem/UsageAmount": usage,
                "lineItem/UsageUnit": "Hrs" if "Usage:" in usage_type else "GB-Mo",
                "lineItem/UnblendedCost": cost,
                "lineItem/NetAmortizedCost": cost
                * random.uniform(0.8, 1.0),  # Slight discount
                "resourceTags/Environment": environment,
                "resourceTags/Team": team,
                "resourceTags/CostCenter": f"CC-{random.randint(1000, 9999)}",
                "resourceTags/Project": f"project-{random.choice(['web', 'api', 'ml', 'data'])}",
            }
            billing_records.append(billing_record)

            # Inventory record
            inventory_record = {
                "resource_id": resource_id,
                "service": service_config["service_name"],
                "region": region,
                "availability_zone": f"{region}{'abc'[random.randint(0, 2)]}",
                "tags": {
                    "Environment": environment,
                    "Team": team,
                    "CostCenter": f"CC-{random.randint(1000, 9999)}",
                    "Project": f"project-{random.choice(['web', 'api', 'ml', 'data'])}",
                },
                "properties": self._generate_sample_properties(
                    service_config["service_name"], usage_type
                ),
                "created_at": (
                    datetime.now() - timedelta(days=random.randint(1, 365))
                ).isoformat(),
            }
            inventory_records.append(inventory_record)

            # Metrics record (for compute resources)
            if service_config["service_name"] in ["EC2", "RDS", "Lambda"]:
                metrics_record = {
                    "resource_id": resource_id,
                    "timestamp": "2024-01-31T23:59:59Z",
                    "period_days": 30,
                    "cpu_utilization_p50": round(random.uniform(5, 80), 1),
                    "cpu_utilization_p90": round(random.uniform(15, 95), 1),
                    "cpu_utilization_p95": round(random.uniform(20, 99), 1),
                    "memory_utilization_p50": round(random.uniform(10, 70), 1),
                    "memory_utilization_p90": round(random.uniform(20, 85), 1),
                    "memory_utilization_p95": round(random.uniform(25, 90), 1),
                    "network_in": random.randint(100000, 10000000),
                    "network_out": random.randint(200000, 20000000),
                    "is_idle": random.random() < 0.1,  # 10% chance of being idle
                }
                metrics_records.append(metrics_record)

        # Save to files
        billing_df = pd.DataFrame(billing_records)
        billing_df.to_csv(self.data_dir / "billing" / "sample_billing.csv", index=False)

        with open(self.data_dir / "inventory" / "sample_inventory.json", "w") as f:
            json.dump(inventory_records, f, indent=2)

        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.to_csv(self.data_dir / "metrics" / "sample_metrics.csv", index=False)

        logger.info(
            "Sample AWS data files created",
            billing_records=len(billing_records),
            inventory_records=len(inventory_records),
            metrics_records=len(metrics_records),
            total_cost=round(
                sum(r["lineItem/UnblendedCost"] for r in billing_records), 2
            ),
            billing_file=str(self.data_dir / "billing" / "sample_billing.csv"),
            inventory_file=str(self.data_dir / "inventory" / "sample_inventory.json"),
            metrics_file=str(self.data_dir / "metrics" / "sample_metrics.csv"),
        )

        return {
            "billing_file": str(self.data_dir / "billing" / "sample_billing.csv"),
            "inventory_file": str(
                self.data_dir / "inventory" / "sample_inventory.json"
            ),
            "metrics_file": str(self.data_dir / "metrics" / "sample_metrics.csv"),
            "total_records": len(billing_records),
            "total_cost": round(
                sum(r["lineItem/UnblendedCost"] for r in billing_records), 2
            ),
        }

    def _generate_sample_properties(
        self, service: str, usage_type: str
    ) -> Dict[str, Any]:
        """Generate realistic properties for different service types"""
        import random

        if service == "EC2":
            instance_type = (
                usage_type.split(":")[-1] if ":" in usage_type else "t3.micro"
            )
            return {
                "instance_type": instance_type,
                "state": random.choice(["running", "stopped"]),
                "cpu_count": random.choice([1, 2, 4, 8, 16]),
                "memory_gb": random.choice([1, 2, 4, 8, 16, 32]),
                "storage_gb": random.choice([8, 20, 50, 100]),
                "platform": random.choice(["Linux/UNIX", "Windows"]),
            }
        elif service == "EBS":
            return {
                "volume_type": (
                    usage_type.split(".")[-1] if "." in usage_type else "gp3"
                ),
                "size_gb": random.choice([8, 20, 50, 100, 500, 1000]),
                "iops": random.choice([3000, 6000, 12000]),
                "throughput": random.choice([125, 250, 500]),
                "state": random.choice(["in-use", "available"]),
                "encrypted": random.choice([True, False]),
            }
        elif service == "S3":
            return {
                "bucket_name": f"bucket-{random.randint(1000, 9999)}",
                "storage_class": random.choice(
                    ["STANDARD", "STANDARD_IA", "ONEZONE_IA", "GLACIER"]
                ),
                "versioning": random.choice([True, False]),
                "encryption": random.choice([True, False]),
            }
        elif service == "RDS":
            db_type = usage_type.split(":")[-1] if ":" in usage_type else "db.t3.micro"
            return {
                "db_instance_class": db_type,
                "engine": random.choice(
                    ["mysql", "postgres", "aurora-mysql", "aurora-postgresql"]
                ),
                "allocated_storage": random.choice([20, 50, 100, 500]),
                "multi_az": random.choice([True, False]),
                "backup_retention": random.choice([1, 7, 14, 30]),
            }
        elif service == "Lambda":
            return {
                "runtime": random.choice(
                    ["python3.9", "nodejs18.x", "java11", "dotnet6"]
                ),
                "memory_size": random.choice([128, 256, 512, 1024, 2048]),
                "timeout": random.choice([30, 60, 300, 900]),
                "architecture": random.choice(["x86_64", "arm64"]),
            }
        else:
            return {}
