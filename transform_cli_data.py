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
        
        # Process all JSON files recursively (exclude Zone.Identifier files)
        json_files = [f for f in self.cli_data_dir.rglob("*.json") 
                     if not f.name.endswith(":Zone.Identifier")]
        
        # Also process text files that might contain metrics (like S3 bucket contents)
        text_files = [f for f in self.cli_data_dir.rglob("*.txt") 
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
        elif "_cpu_utilization" in filename:
            self._process_ec2_instance_metrics(data, filename, "cpu")
        elif "_memory_utilization" in filename:
            self._process_ec2_instance_metrics(data, filename, "memory")
        elif "_uptime" in filename:
            self._process_ec2_instance_metrics(data, filename, "uptime")
        elif "monitoring_cpu" in filename:
            self._process_monitoring_metrics(data, filename, "cpu")
        elif "monitoring_mem" in filename:
            self._process_monitoring_metrics(data, filename, "memory")
        elif "monitoring_network" in filename:
            self._process_monitoring_metrics(data, filename, "network")
        elif "monitoring_readops" in filename or "monitoring_writeops" in filename:
            self._process_monitoring_metrics(data, filename, "iops")
        elif "monitoring_readbytes" in filename or "monitoring_writebytes" in filename:
            self._process_monitoring_metrics(data, filename, "throughput")
        elif "monitoring_burstbalance" in filename:
            self._process_monitoring_metrics(data, filename, "burst")
        elif "monitoring_freestorage" in filename:
            self._process_monitoring_metrics(data, filename, "storage")
        elif "monitoring_numberofobjects" in filename:
            self._process_monitoring_metrics(data, filename, "objects")
        elif "monitoring_bytesdownloaded" in filename:
            self._process_monitoring_metrics(data, filename, "downloads")
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
        instance_id = self._extract_resource_id_from_filename(filename)
        
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

    def _process_ec2_instance_metrics(self, data: Dict, filename: str, metric_type: str):
        """Process EC2 instance-specific metrics files"""
        # Extract instance ID from filename like: ec2_i-05ea7a5068995a703_cpu_utilization.json
        instance_id = None
        if "_i-" in filename:
            parts = filename.split("_")
            for part in parts:
                if part.startswith("i-"):
                    instance_id = part
                    break
        
        if not instance_id:
            return
            
        if metric_type == "cpu":
            # Handle CPU utilization data
            datapoints = data.get("Datapoints", [])
            if not datapoints:
                return
                
            # Extract CPU values with timestamps for time-series analysis
            cpu_timeseries = []
            for dp in datapoints:
                if dp.get("Average") is not None:
                    cpu_timeseries.append({
                        "timestamp": dp.get("Timestamp"),
                        "value": dp.get("Average")
                    })
            
            if not cpu_timeseries:
                return
                
            # Sort by timestamp for proper time-series analysis
            cpu_timeseries.sort(key=lambda x: x["timestamp"])
            cpu_values = [dp["value"] for dp in cpu_timeseries]
            
            # Calculate basic statistics
            cpu_values_sorted = sorted(cpu_values)
            n = len(cpu_values_sorted)
            
            avg_cpu = sum(cpu_values_sorted) / n
            p90_cpu = cpu_values_sorted[int(n * 0.9)] if n > 0 else avg_cpu
            p95_cpu = cpu_values_sorted[int(n * 0.95)] if n > 0 else avg_cpu
            min_cpu = min(cpu_values_sorted)
            max_cpu = max(cpu_values_sorted)
            
            # Time-series pattern analysis
            time_patterns = self._analyze_time_patterns(cpu_timeseries)
            
            metric_record = {
                "resource_id": instance_id,
                "timestamp": datetime.now().isoformat(),
                "period_days": 30,
                "cpu_utilization_p50": avg_cpu,
                "cpu_utilization_p90": p90_cpu,
                "cpu_utilization_p95": p95_cpu,
                "cpu_utilization_min": min_cpu,
                "cpu_utilization_max": max_cpu,
                "cpu_utilization_stddev": self._calculate_stddev(cpu_values),
                "cpu_utilization_trend": time_patterns.get("trend", "stable"),
                "cpu_utilization_volatility": time_patterns.get("volatility", "low"),
                "cpu_utilization_peak_hours": time_patterns.get("peak_hours", []),
                "cpu_utilization_patterns": time_patterns.get("patterns", {}),
                "cpu_timeseries_data": cpu_timeseries[-30:] if len(cpu_timeseries) > 30 else cpu_timeseries  # Last 30 data points
            }
            self._update_or_add_metric(metric_record)
            
        elif metric_type == "memory":
            # Handle memory utilization data
            datapoints = data.get("Datapoints", [])
            if not datapoints:
                return
                
            memory_values = [dp.get("Average", 0) for dp in datapoints if dp.get("Average") is not None]
            if not memory_values:
                return
                
            memory_values.sort()
            n = len(memory_values)
            
            avg_memory = sum(memory_values) / n
            p90_memory = memory_values[int(n * 0.9)] if n > 0 else avg_memory
            p95_memory = memory_values[int(n * 0.95)] if n > 0 else avg_memory
            
            metric_record = {
                "resource_id": instance_id,
                "memory_utilization_p50": avg_memory,
                "memory_utilization_p90": p90_memory,
                "memory_utilization_p95": p95_memory,
            }
            self._update_or_add_metric(metric_record)
            
        elif metric_type == "uptime":
            # Handle uptime/status data
            if isinstance(data, dict):
                status = data.get("State", {}).get("Name", "unknown")
                metric_record = {
                    "resource_id": instance_id,
                    "uptime_status": status,
                }
                self._update_or_add_metric(metric_record)

    def _analyze_time_patterns(self, timeseries_data):
        """Analyze time-series patterns for trends and volatility"""
        if len(timeseries_data) < 3:
            return {"trend": "insufficient_data", "volatility": "unknown", "patterns": {}}
        
        values = [dp["value"] for dp in timeseries_data]
        timestamps = [dp["timestamp"] for dp in timeseries_data]
        
        # Calculate trend using simple linear regression slope
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.1:
            trend = "stable"
        elif slope > 0.1:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Calculate volatility (coefficient of variation)
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_dev = variance ** 0.5
        cv = (std_dev / mean_val) if mean_val > 0 else 0
        
        if cv < 0.2:
            volatility = "low"
        elif cv < 0.5:
            volatility = "moderate"
        else:
            volatility = "high"
        
        # Extract peak hours (simplified - assumes daily patterns)
        peak_hours = []
        try:
            from datetime import datetime as dt
            for ts_data in timeseries_data:
                timestamp = ts_data["timestamp"]
                if isinstance(timestamp, str):
                    dt_obj = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt_obj.hour
                    if ts_data["value"] > mean_val * 1.2:  # Values 20% above mean
                        peak_hours.append(hour)
        except:
            pass
        
        # Remove duplicates and sort
        peak_hours = sorted(list(set(peak_hours)))
        
        patterns = {
            "slope": round(slope, 4),
            "coefficient_of_variation": round(cv, 3),
            "mean": round(mean_val, 2),
            "data_points": n,
            "time_span_days": (len(timeseries_data) * 1) if len(timeseries_data) > 0 else 0  # Assuming daily data
        }
        
        return {
            "trend": trend,
            "volatility": volatility,
            "peak_hours": peak_hours,
            "patterns": patterns
        }

    def _calculate_stddev(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

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

    def _process_monitoring_metrics(self, data: Dict, filename: str, metric_type: str):
        """Process CloudWatch monitoring metrics data"""
        
        # Extract resource ID from filename
        resource_id = self._extract_resource_id_from_filename(filename)
        if not resource_id:
            return
            
        # Handle both formats: MetricDataResults (new format) and Datapoints (old format)
        datapoints = []
        if "MetricDataResults" in data:
            for result in data["MetricDataResults"]:
                values = result.get("Values", [])
                timestamps = result.get("Timestamps", [])
                for i, value in enumerate(values):
                    timestamp = timestamps[i] if i < len(timestamps) else None
                    datapoints.append({"Average": value, "Timestamp": timestamp})
        elif "Datapoints" in data:
            datapoints = data["Datapoints"]
        else:
            return
            
        if not datapoints:
            return
            
        # Calculate statistics
        values = [dp.get("Average", 0) for dp in datapoints if dp.get("Average") is not None]
        if not values:
            return
            
        avg_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
        
        # Get timestamp from first datapoint
        timestamp = datapoints[0].get("Timestamp") if datapoints else datetime.now().isoformat()
        
        # Create metric record based on type
        metric_record = {
            "resource_id": resource_id,
            "timestamp": timestamp,
            "period_days": 30,
        }
        
        if metric_type == "cpu":
            metric_record.update({
                "cpu_utilization_p50": avg_value,
                "cpu_utilization_p90": avg_value * 1.2,  # Estimate
                "cpu_utilization_p95": max_value,
            })
        elif metric_type == "memory":
            metric_record.update({
                "memory_utilization_p50": avg_value,
                "memory_utilization_p90": avg_value * 1.2,  # Estimate
                "memory_utilization_p95": max_value,
            })
        elif metric_type == "network":
            if "receive" in filename or "in" in filename:
                metric_record["network_in"] = avg_value
            elif "transmit" in filename or "out" in filename:
                metric_record["network_out"] = avg_value
        elif metric_type == "iops":
            if "read" in filename:
                metric_record["iops_read"] = avg_value
            elif "write" in filename:
                metric_record["iops_write"] = avg_value
        elif metric_type == "throughput":
            if "read" in filename:
                metric_record["throughput_read"] = avg_value
            elif "write" in filename:
                metric_record["throughput_write"] = avg_value
        elif metric_type == "storage":
            metric_record["storage_used"] = avg_value
        elif metric_type == "objects":
            metric_record["object_count"] = avg_value
        elif metric_type == "downloads":
            metric_record["bytes_downloaded"] = avg_value
        elif metric_type == "burst":
            metric_record["burst_balance"] = avg_value
            
        self._update_or_add_metric(metric_record)

    def _extract_resource_id_from_filename(self, filename: str) -> str:
        """Extract resource ID from monitoring filename"""
        # Pattern: monitoring_metric_resourceid_date.json
        parts = filename.split('_')
        if len(parts) >= 3:
            # For patterns like monitoring_cpu_i-12345_202506
            if parts[2].startswith('i-'):  # EC2 instance
                return parts[2]
            elif parts[2].startswith('vol-'):  # EBS volume
                return parts[2]
            elif len(parts) >= 4 and parts[3].startswith('i-'):  # Some patterns have extra parts
                return parts[3]
            elif len(parts) >= 4 and parts[3].startswith('vol-'):
                return parts[3]
            else:
                # For other patterns, try to find resource ID-like strings
                for part in parts:
                    if part.startswith(('i-', 'vol-', 'fs-', 'lambda-')):
                        return part
                # For RDS and S3 patterns like monitoring_cpu_akaocr360-1_202506
                # Extract the database or bucket name
                if len(parts) >= 3:
                    return parts[2]
        return None

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
        import json
        
        resource_id = metric_data.get("resource_id")
        
        # Clean up None values and serialize complex data for CSV
        for key, value in metric_data.items():
            if value is None:
                metric_data[key] = ""
            elif isinstance(value, (list, dict)):
                metric_data[key] = json.dumps(value) if value else ""
        
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
