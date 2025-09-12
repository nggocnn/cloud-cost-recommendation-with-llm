#!/usr/bin/env python3
"""
Transform AWS CLI JSON data into LLM cost recommendation system format.

This script converts raw AWS CLI outputs into the structured format expected by
the LLM cost recommendation system (inventory JSON, billing CSV, metrics CSV).
"""

import json
import csv
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import argparse


class AWSCLIDataTransformer:
    """Transform AWS CLI JSON outputs to system-expected formats"""
    
    def __init__(self, cli_data_dir: str, output_dir: str = "data"):
        self.cli_data_dir = Path(cli_data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        (self.output_dir / "inventory").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "billing").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        
        self.inventory_data = []
        self.metrics_data = []
        self.billing_data = []

    def transform_all_cli_data(self):
        """Transform all JSON files in the CLI data directory"""
        print(f"Scanning {self.cli_data_dir} for AWS CLI JSON files...")
        
        # Process all JSON files (exclude Zone.Identifier files)
        json_files = [f for f in self.cli_data_dir.glob("*.json") 
                     if not f.name.endswith(":Zone.Identifier")]
        
        # Also process text files that might contain metrics (like S3 bucket contents)
        text_files = [f for f in self.cli_data_dir.glob("*.txt") 
                     if not f.name.endswith(":Zone.Identifier")]
        
        all_files = json_files + text_files
        print(f"Found {len(json_files)} JSON files and {len(text_files)} text files to process")
        
        for file in all_files:
            try:
                self._process_cli_file(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        # Save transformed data
        self._save_inventory_data()
        self._save_metrics_data()
        self._save_billing_data()
        
        print("\nTransformation completed!")
        print(f"Generated files:")
        print(f"  - Inventory: {len(self.inventory_data)} resources")
        print(f"  - Metrics: {len(self.metrics_data)} metric records")
        print(f"  - Billing: {len(self.billing_data)} billing records")

    def _process_cli_file(self, file: Path):
        """Process a single CLI file (JSON or text)"""
        print(f"Processing: {file.name}")
        
        filename = file.stem.lower()
        
        # Handle text files (like S3 bucket contents)
        if file.suffix == '.txt':
            if "s3_bucket_contents" in filename:
                self._process_s3_bucket_contents(file)
            return
        
        # Handle JSON files
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Route to appropriate processor based on filename
        if "ec2_instance" in filename:
            self._process_ec2_instances(data, filename)
        elif "ec2_cpu_metrics" in filename:
            self._process_ec2_metrics(data, filename, "cpu")
        elif "ec2_network" in filename:
            self._process_ec2_metrics(data, filename, "network")
        elif "ebs_volume" in filename:
            self._process_ebs_volumes(data, filename)
        elif "ebs_" in filename and "metrics" in filename:
            self._process_ebs_metrics(data, filename)
        elif "lambda_function" in filename:
            self._process_lambda_functions(data, filename)
        elif "lambda_" in filename and "metrics" in filename:
            self._process_lambda_metrics(data, filename)
        elif "cloudfront" in filename:
            self._process_cloudfront(data, filename)
        elif "efs" in filename:
            self._process_efs(data, filename)
        elif "s3_bucket" in filename:
            self._process_s3(data, filename)
        # Add more processors as needed
        else:
            print(f"  - No specific processor for {filename}, skipping")

    def _process_ec2_instances(self, data: Dict, filename: str):
        """Process EC2 instance data"""
        if "Reservations" not in data:
            return
            
        for reservation in data["Reservations"]:
            for instance in reservation.get("Instances", []):
                instance_id = instance.get("InstanceId")
                if not instance_id:
                    continue
                
                # Extract basic info
                resource = {
                    "resource_id": instance_id,
                    "service": "EC2",
                    "region": self._extract_region_from_placement(instance.get("Placement", {})),
                    "availability_zone": instance.get("Placement", {}).get("AvailabilityZone"),
                    "account_id": "123456789012",  # Default - update with actual if available
                    "tags": self._extract_tags(instance.get("Tags", [])),
                    "properties": {
                        "instance_type": instance.get("InstanceType"),
                        "state": instance.get("State", {}).get("Name"),
                        "launch_time": instance.get("LaunchTime"),
                        "platform": instance.get("Platform", "Linux/UNIX"),
                        "vpc_id": instance.get("VpcId"),
                        "subnet_id": instance.get("SubnetId"),
                        "security_groups": [sg.get("GroupId") for sg in instance.get("SecurityGroups", [])],
                        "monitoring": instance.get("Monitoring", {}).get("State"),
                        "ebs_optimized": instance.get("EbsOptimized", False),
                    },
                    "created_at": instance.get("LaunchTime"),
                }
                self.inventory_data.append(resource)
                
                # Generate sample billing data
                self._generate_billing_record(
                    resource_id=instance_id,
                    service="Amazon Elastic Compute Cloud",
                    region=resource["region"],
                    usage_type=f"BoxUsage:{instance.get('InstanceType', 't3.micro')}",
                    usage_amount=720,  # Assume running full month
                    cost=self._estimate_ec2_cost(instance.get('InstanceType', 't3.micro')),
                    tags=resource["tags"]
                )

    def _process_ebs_volumes(self, data: Dict, filename: str):
        """Process EBS volume data"""
        volumes = data.get("Volumes", [])
        
        for volume in volumes:
            volume_id = volume.get("VolumeId")
            if not volume_id:
                continue
                
            resource = {
                "resource_id": volume_id,
                "service": "EBS",
                "region": self._extract_region_from_az(volume.get("AvailabilityZone", "")),
                "availability_zone": volume.get("AvailabilityZone"),
                "account_id": "123456789012",
                "tags": self._extract_tags(volume.get("Tags", [])),
                "properties": {
                    "volume_type": volume.get("VolumeType"),
                    "size_gb": volume.get("Size"),
                    "state": volume.get("State"),
                    "iops": volume.get("Iops"),
                    "throughput": volume.get("Throughput"),
                    "encrypted": volume.get("Encrypted", False),
                    "snapshot_id": volume.get("SnapshotId"),
                },
                "created_at": volume.get("CreateTime"),
            }
            self.inventory_data.append(resource)
            
            # Generate billing data
            self._generate_billing_record(
                resource_id=volume_id,
                service="Amazon Elastic Block Store",
                region=resource["region"],
                usage_type=f"VolumeUsage.{volume.get('VolumeType', 'gp3')}",
                usage_amount=volume.get("Size", 20),
                cost=self._estimate_ebs_cost(volume.get("VolumeType", "gp3"), volume.get("Size", 20)),
                tags=resource["tags"]
            )

    def _process_lambda_functions(self, data: Dict, filename: str):
        """Process Lambda function data"""
        config = data.get("Configuration", {})
        if not config:
            return
            
        function_name = config.get("FunctionName")
        if not function_name:
            return
            
        function_arn = config.get("FunctionArn", "")
        region = self._extract_region_from_arn(function_arn)
        
        resource = {
            "resource_id": function_name,
            "service": "Lambda",
            "region": region,
            "availability_zone": None,
            "account_id": "123456789012",
            "tags": self._extract_tags(config.get("Tags", {})),
            "properties": {
                "runtime": config.get("Runtime"),
                "memory_size": config.get("MemorySize"),
                "timeout": config.get("Timeout"),
                "code_size": config.get("CodeSize"),
                "handler": config.get("Handler"),
                "last_modified": config.get("LastModified"),
                "state": config.get("State"),
                "architecture": config.get("Architectures", ["x86_64"])[0] if config.get("Architectures") else "x86_64",
            },
            "created_at": config.get("LastModified"),
        }
        self.inventory_data.append(resource)
        
        # Generate billing data
        self._generate_billing_record(
            resource_id=function_name,
            service="AWS Lambda",
            region=region,
            usage_type="Request",
            usage_amount=10000,  # Estimated requests
            cost=5.0,  # Estimated cost
            tags=resource["tags"]
        )

    def _process_ec2_metrics(self, data: Dict, filename: str, metric_type: str):
        """Process EC2 CloudWatch metrics"""
        datapoints = data.get("Datapoints", [])
        
        # Extract instance ID from filename or other source
        # This is a simplified approach - you might need to adjust based on your naming
        instance_id = self._extract_resource_id_from_filename(filename, "i-")
        
        if not instance_id or not datapoints:
            return
            
        # Aggregate metrics over the period
        if metric_type == "cpu":
            avg_cpu = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
            max_cpu = max(dp.get("Maximum", 0) for dp in datapoints)
            
            metric_record = {
                "resource_id": instance_id,
                "timestamp": datetime.now().isoformat(),
                "period_days": 30,
                "cpu_utilization_p50": avg_cpu,
                "cpu_utilization_p90": max_cpu * 0.9,
                "cpu_utilization_p95": max_cpu * 0.95,
            }
            self._update_or_add_metric(metric_record)
            
        elif metric_type == "network":
            if "network_in" in filename:
                avg_network = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
                metric_record = {
                    "resource_id": instance_id,
                    "network_in": avg_network,
                }
            elif "network_out" in filename:
                avg_network = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
                metric_record = {
                    "resource_id": instance_id,
                    "network_out": avg_network,
                }
            else:
                return
                
            self._update_or_add_metric(metric_record)

    def _process_lambda_metrics(self, data: Dict, filename: str):
        """Process Lambda CloudWatch metrics"""
        datapoints = data.get("Datapoints", [])
        
        # Extract function name from filename
        function_name = self._extract_function_name_from_filename(filename)
        
        if not function_name or not datapoints:
            return
            
        # Aggregate metrics over the period
        if "duration" in filename:
            avg_duration = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
            metric_record = {
                "resource_id": function_name,
                "lambda_duration_avg": avg_duration,
            }
            self._update_or_add_metric(metric_record)
        elif "errors" in filename:
            error_count = sum(dp.get("Sum", 0) for dp in datapoints)
            metric_record = {
                "resource_id": function_name,
                "lambda_errors": error_count,
            }
            self._update_or_add_metric(metric_record)

    def _process_ebs_metrics(self, data: Dict, filename: str):
        """Process EBS CloudWatch metrics"""
        datapoints = data.get("Datapoints", [])
        
        # Extract volume ID from filename
        volume_id = self._extract_volume_id_from_filename(filename)
        
        if not volume_id or not datapoints:
            return
            
        # Aggregate metrics over the period
        if "read_ops" in filename:
            avg_iops = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
            # Estimate throughput from IOPS (assuming 16KB average I/O size)
            avg_throughput = avg_iops * 16 * 1024  # bytes per second
            metric_record = {
                "resource_id": volume_id,
                "iops_read": avg_iops,
                "throughput_read": avg_throughput,
            }
            self._update_or_add_metric(metric_record)
        elif "write_ops" in filename:
            avg_iops = sum(dp.get("Average", 0) for dp in datapoints) / len(datapoints)
            # Estimate throughput from IOPS (assuming 16KB average I/O size)
            avg_throughput = avg_iops * 16 * 1024  # bytes per second
            metric_record = {
                "resource_id": volume_id,
                "iops_write": avg_iops,
                "throughput_write": avg_throughput,
            }
            self._update_or_add_metric(metric_record)

    def _process_cloudfront(self, data: Dict, filename: str):
        """Process CloudFront distributions"""
        if "distributions_list" in filename:
            distributions = data.get("DistributionList", {}).get("Items", [])
            for dist in distributions:
                dist_id = dist.get("Id")
                if not dist_id:
                    continue
                    
                resource = {
                    "resource_id": dist_id,
                    "service": "CloudFront",
                    "region": "global",
                    "availability_zone": None,
                    "account_id": "123456789012",
                    "tags": {},
                    "properties": {
                        "domain_name": dist.get("DomainName"),
                        "status": dist.get("Status"),
                        "price_class": dist.get("PriceClass"),
                        "enabled": dist.get("Enabled", False),
                        "comment": dist.get("Comment", ""),
                    },
                    "created_at": dist.get("LastModifiedTime"),
                }
                self.inventory_data.append(resource)

    def _process_efs(self, data: Dict, filename: str):
        """Process EFS file systems"""
        if "file_system_details" in filename:
            fs = data
            fs_id = fs.get("FileSystemId")
            if not fs_id:
                return
                
            resource = {
                "resource_id": fs_id,
                "service": "EFS",
                "region": self._extract_region_from_arn(fs.get("FileSystemArn", "")),
                "availability_zone": None,
                "account_id": "123456789012",
                "tags": self._extract_tags(fs.get("Tags", [])),
                "properties": {
                    "performance_mode": fs.get("PerformanceMode"),
                    "throughput_mode": fs.get("ThroughputMode"),
                    "size_bytes": fs.get("SizeInBytes", {}).get("Value", 0),
                    "encrypted": fs.get("Encrypted", False),
                    "lifecycle_policy": fs.get("LifeCyclePolicy"),
                },
                "created_at": fs.get("CreationTime"),
            }
            self.inventory_data.append(resource)

    def _process_s3(self, data: Dict, filename: str):
        """Process S3 bucket data"""
        if "bucket_location" in filename:
            # Extract bucket name from other files or use a default
            bucket_name = self._extract_bucket_name_from_filename(filename)
            if not bucket_name:
                bucket_name = "eip-finops"  # Default based on the contents we saw
            
            region = data.get("LocationConstraint") or "us-east-1"
            
            resource = {
                "resource_id": bucket_name,
                "service": "S3",
                "region": region,
                "availability_zone": None,
                "account_id": "123456789012",
                "tags": {},
                "properties": {
                    "region": region,
                    "storage_class": "Standard",  # Default
                },
                "created_at": datetime.now().isoformat(),
            }
            self.inventory_data.append(resource)

    def _process_s3_bucket_contents(self, file: Path):
        """Process S3 bucket contents from text file"""
        bucket_name = "eip-finops"  # Extract from filename if needed
        
        with open(file, 'r') as f:
            content = f.read()
        
        # Parse the S3 ls output
        lines = content.strip().split('\n')
        total_objects = 0
        total_size_bytes = 0
        
        for line in lines:
            if line.startswith('Total Objects:'):
                total_objects = int(line.split(':')[1].strip())
            elif line.startswith('   Total Size:'):
                size_str = line.split(':')[1].strip()
                # Parse size string like "2.2 MiB"
                if 'MiB' in size_str:
                    size_val = float(size_str.split()[0])
                    total_size_bytes = size_val * 1024 * 1024
                elif 'KiB' in size_str:
                    size_val = float(size_str.split()[0])
                    total_size_bytes = size_val * 1024
                elif 'Bytes' in size_str:
                    total_size_bytes = float(size_str.split()[0])
        
        # Estimate requests based on object count (rough estimate)
        estimated_get_requests = total_objects * 10  # Assume 10 GET requests per object per month
        estimated_put_requests = total_objects * 1   # Assume 1 PUT request per object per month
        
        metric_record = {
            "resource_id": bucket_name,
            "timestamp": datetime.now().isoformat(),
            "period_days": 30,
            "storage_used": total_size_bytes,
            "requests_get": estimated_get_requests,
            "requests_put": estimated_put_requests,
        }
        self._update_or_add_metric(metric_record)

    def _extract_bucket_name_from_filename(self, filename: str) -> str:
        """Extract bucket name from filename"""
        # If the filename contains bucket name patterns
        if "eip-finops" in filename:
            return "eip-finops"
        # Could add more patterns as needed
        return None

    def _extract_tags(self, tags_data) -> Dict[str, str]:
        """Extract tags from AWS format to simple dict"""
        if isinstance(tags_data, list):
            return {tag.get("Key", ""): tag.get("Value", "") for tag in tags_data}
        elif isinstance(tags_data, dict):
            return tags_data
        return {}

    def _extract_region_from_placement(self, placement: Dict) -> str:
        """Extract region from placement data"""
        az = placement.get("AvailabilityZone", "")
        return az[:-1] if az else "us-east-1"

    def _extract_region_from_az(self, az: str) -> str:
        """Extract region from availability zone"""
        return az[:-1] if az else "us-east-1"

    def _extract_region_from_arn(self, arn: str) -> str:
        """Extract region from ARN"""
        if not arn:
            return "us-east-1"
        parts = arn.split(":")
        return parts[3] if len(parts) > 3 else "us-east-1"

    def _extract_resource_id_from_filename(self, filename: str, prefix: str) -> Optional[str]:
        """Extract resource ID from filename (improved approach)"""
        # Look for resource ID patterns in the filename
        import re
        
        if prefix == "i-":
            # Look for EC2 instance ID pattern: i-xxxxxxxxxxxxxxxxx
            match = re.search(r'(i-[a-f0-9]{17})', filename)
            if match:
                return match.group(1)
        elif prefix == "vol-":
            # Look for EBS volume ID pattern: vol-xxxxxxxxxxxxxxxxx
            match = re.search(r'(vol-[a-f0-9]{17})', filename)
            if match:
                return match.group(1)
        
        # Fallback: if the filename contains the prefix, try to extract it
        if prefix in filename:
            parts = filename.split('_')
            for part in parts:
                if part.startswith(prefix):
                    return part
        
        return None

    def _extract_function_name_from_filename(self, filename: str) -> Optional[str]:
        """Extract Lambda function name from filename"""
        # Try to extract from common patterns
        if "lambda_" in filename and "_" in filename:
            parts = filename.split("_")
            # Look for function name patterns
            for part in parts:
                if part and not part in ["lambda", "metrics", "duration", "errors", "invocations"]:
                    return part
        return None

    def _extract_volume_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract EBS volume ID from filename"""
        import re
        
        # Look for volume ID pattern: vol-xxxxxxxxxxxxxxxxx
        match = re.search(r'(vol-[a-f0-9]{17})', filename)
        if match:
            return match.group(1)
        
        # Fallback: if the filename contains vol-, try to extract it
        if "vol-" in filename:
            parts = filename.split('_')
            for part in parts:
                if part.startswith("vol-"):
                    return part
        
        return None

    def _estimate_ec2_cost(self, instance_type: str) -> float:
        """Estimate monthly EC2 cost based on instance type"""
        cost_map = {
            "t3.nano": 3.80, "t3.micro": 7.59, "t3.small": 15.18,
            "t3.medium": 30.37, "t3.large": 60.74, "t3.xlarge": 121.47,
            "m5.large": 70.08, "m5.xlarge": 140.16, "m5.2xlarge": 280.32,
            "c5.large": 62.56, "c5.xlarge": 125.12, "r5.large": 91.25,
        }
        return cost_map.get(instance_type, 50.0)  # Default cost

    def _estimate_ebs_cost(self, volume_type: str, size_gb: int) -> float:
        """Estimate monthly EBS cost"""
        cost_per_gb = {"gp3": 0.08, "gp2": 0.10, "io1": 0.125, "io2": 0.125}
        return cost_per_gb.get(volume_type, 0.08) * size_gb

    def _generate_billing_record(self, resource_id: str, service: str, region: str,
                                usage_type: str, usage_amount: float, cost: float,
                                tags: Dict[str, str]):
        """Generate a billing record"""
        billing_record = {
            "bill/BillingPeriodStartDate": "2024-01-01",
            "bill/BillingPeriodEndDate": "2024-01-31",
            "lineItem/UsageAccountId": "123456789012",
            "product/ProductName": service,
            "lineItem/ResourceId": resource_id,
            "product/region": region,
            "lineItem/UsageType": usage_type,
            "lineItem/UsageAmount": usage_amount,
            "lineItem/UsageUnit": "Hrs" if "Usage:" in usage_type or "BoxUsage:" in usage_type else "GB-Mo",
            "lineItem/UnblendedCost": cost,
            "lineItem/NetAmortizedCost": cost * 0.95,  # Slight discount
        }
        
        # Add resource tags
        for key, value in tags.items():
            billing_record[f"resourceTags/{key}"] = value
            
        self.billing_data.append(billing_record)

    def _update_or_add_metric(self, metric_data: Dict):
        """Update existing metric record or add new one"""
        resource_id = metric_data.get("resource_id")
        
        # Find existing metric record
        existing_record = None
        for i, record in enumerate(self.metrics_data):
            if record.get("resource_id") == resource_id:
                existing_record = i
                break
        
        if existing_record is not None:
            # Update existing record
            self.metrics_data[existing_record].update(metric_data)
        else:
            # Add default values for new record
            default_metric = {
                "resource_id": resource_id,
                "timestamp": datetime.now().isoformat(),
                "period_days": 30,
                "cpu_utilization_p50": None,
                "cpu_utilization_p90": None,
                "cpu_utilization_p95": None,
                "memory_utilization_p50": None,
                "memory_utilization_p90": None,
                "memory_utilization_p95": None,
                "network_in": None,
                "network_out": None,
                "is_idle": False,
            }
            default_metric.update(metric_data)
            self.metrics_data.append(default_metric)

    def _save_inventory_data(self):
        """Save inventory data to JSON file"""
        output_file = self.output_dir / "inventory" / "cli_inventory.json"
        with open(output_file, 'w') as f:
            json.dump(self.inventory_data, f, indent=2, default=str)
        print(f"Saved inventory data: {output_file}")

    def _save_metrics_data(self):
        """Save metrics data to CSV file"""
        if not self.metrics_data:
            print("No metrics data to save")
            return
            
        output_file = self.output_dir / "metrics" / "cli_metrics.csv"
        df = pd.DataFrame(self.metrics_data)
        df.to_csv(output_file, index=False)
        print(f"Saved metrics data: {output_file}")

    def _save_billing_data(self):
        """Save billing data to CSV file"""
        if not self.billing_data:
            print("No billing data to save")
            return
            
        output_file = self.output_dir / "billing" / "cli_billing.csv"
        df = pd.DataFrame(self.billing_data)
        df.to_csv(output_file, index=False)
        print(f"Saved billing data: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Transform AWS CLI JSON data")
    parser.add_argument("--cli-dir", default="data/cli", 
                       help="Directory containing AWS CLI JSON files")
    parser.add_argument("--output-dir", default="data",
                       help="Output directory for transformed data")
    
    args = parser.parse_args()
    
    transformer = AWSCLIDataTransformer(args.cli_dir, args.output_dir)
    transformer.transform_all_cli_data()
    
    print("\nTo analyze with LLM cost recommendation system, run:")
    print(f"python -m llm_cost_recommendation \\")
    print(f"  --billing-file {args.output_dir}/billing/cli_billing.csv \\")
    print(f"  --inventory-file {args.output_dir}/inventory/cli_inventory.json \\")
    print(f"  --metrics-file {args.output_dir}/metrics/cli_metrics.csv \\")
    print(f"  --output-file cli_analysis.json")


if __name__ == "__main__":
    main()
