# Custom Conditions for LLM Cost Recommendation Agents

This document explains how to use the custom conditional rules system to adapt agent behavior to specific requirements.

## Overview

The custom conditions system allows you to:

- **Modify thresholds** based on resource tags, metrics, or properties
- **Skip or force specific recommendation types** for certain resources
- **Adjust risk levels** for recommendations
- **Add custom prompts** for specialized analysis
- **Create complex logic** using AND/OR operators

## Condition Structure

### Basic Condition

```yaml
conditions:
  - field: "tag.Environment"
    operator: "equals"
    value: "production"
    description: "Resource tagged as production"
```

### Available Fields

#### Resource Fields

- `resource_id` - The AWS resource ID
- `service` - AWS service type (EC2, S3, RDS, etc.)
- `region` - AWS region
- `availability_zone` - AZ for the resource
- `account_id` - AWS account ID (for reference, system processes multi-account data)

#### Tag Fields (prefix with `tag.`)

- `tag.Environment` - Environment tag
- `tag.Application` - Application tag
- `tag.Criticality` - Criticality level
- `tag.Owner` - Resource owner
- `tag.*` - Any custom tag

#### Metrics Fields

- `cpu_utilization_p50` - 50th percentile CPU utilization
- `cpu_utilization_p90` - 90th percentile CPU utilization
- `cpu_utilization_p95` - 95th percentile CPU utilization
- `memory_utilization_p50/p90/p95` - Memory utilization percentiles
- `network_in/network_out` - Network metrics
- `iops_read/iops_write` - IOPS metrics
- `throughput_read/throughput_write` - Throughput metrics
- `is_idle` - Boolean idle status

#### Cost Fields

- `monthly_cost` - Estimated monthly cost
- `daily_cost` - Estimated daily cost

#### Time Fields

- `created_at` - Resource creation timestamp
- `age_days` - Resource age in days

#### Property Fields (prefix with `property.`)

- `property.instance_type` - EC2 instance type
- `property.storage_class` - S3 storage class
- `property.multi_az` - RDS Multi-AZ status
- `property.*` - Any resource property

### Available Operators

#### Comparison Operators

- `equals` / `not_equals` - Exact string/number matching
- `greater_than` / `less_than` - Numeric comparison
- `greater_equal` / `less_equal` - Numeric comparison with equality

#### String Operators

- `contains` / `not_contains` - Substring matching
- `regex` - Regular expression matching

#### List Operators

- `in` / `not_in` - Value in list matching

#### Existence Operators

- `exists` / `not_exists` - Field presence checking

## Rule Actions

### Threshold Overrides

```yaml
threshold_overrides:
  cpu_low_threshold: 70.0      # Increase CPU threshold
  memory_low_threshold: 60.0   # Increase memory threshold
  iops_utilization_threshold: 50.0
```

### Recommendation Type Control

```yaml
# Skip certain recommendation types
skip_recommendation_types:
  - "rightsizing"
  - "idle_resource"

# Force certain recommendation types
force_recommendation_types:
  - "purchasing_option"
  - "lifecycle"
```

### Risk Level Adjustment

```yaml
risk_adjustment: "increase"  # or "decrease"
```

### Custom Prompts

```yaml
custom_prompt: "This is a critical production resource. Focus on purchasing optimizations only and avoid any performance-impacting changes."
```

## Example Rules

### 1. Production CPU Buffer Rule

```yaml
- name: "production_cpu_buffer"
  description: "Production instances need 30% CPU buffer"
  enabled: true
  priority: 100
  logic: "AND"
  conditions:
    - field: "tag.Environment"
      operator: "equals"
      value: "production"
    - field: "cpu_utilization_p95"
      operator: "greater_than"
      value: 70
  threshold_overrides:
    cpu_low_threshold: 70.0
  custom_prompt: "Maintain 30% CPU headroom for traffic spikes."
```

### 2. Critical Application Protection

```yaml
- name: "critical_no_rightsizing"
  description: "Critical applications avoid rightsizing"
  enabled: true
  priority: 90
  logic: "OR"
  conditions:
    - field: "tag.Criticality"
      operator: "equals"
      value: "critical"
    - field: "tag.Application"
      operator: "in"
      value: ["payment-gateway", "auth-service"]
  skip_recommendation_types:
    - "rightsizing"
  risk_adjustment: "increase"
```

### 3. Development Environment Aggressive Optimization

```yaml
- name: "dev_aggressive"
  description: "Development can be optimized aggressively"
  enabled: true
  priority: 50
  logic: "OR"
  conditions:
    - field: "tag.Environment"
      operator: "in"
      value: ["dev", "test", "staging"]
  threshold_overrides:
    cpu_low_threshold: 10.0
    memory_low_threshold: 20.0
  risk_adjustment: "decrease"
```

### 4. High-Cost Resource Prioritization

```yaml
- name: "high_cost_priority"
  description: "Prioritize high-cost resources"
  enabled: true
  priority: 80
  logic: "AND"
  conditions:
    - field: "monthly_cost"
      operator: "greater_than"
      value: 500
  force_recommendation_types:
    - "purchasing_option"
    - "rightsizing"
```

### 5. Compliance Data Handling

```yaml
- name: "compliance_special_handling"
  description: "Compliance data needs special handling"
  enabled: true
  priority: 100
  logic: "OR"
  conditions:
    - field: "tag.DataClass"
      operator: "in"
      value: ["PII", "PHI", "financial"]
    - field: "tag.Compliance"
      operator: "exists"
      value: true
  threshold_overrides:
    ia_access_threshold: 90  # Longer retention
  custom_prompt: "Ensure lifecycle policies meet regulatory requirements."
```

### 6. Database Performance Protection

```yaml
- name: "database_performance_buffer"
  description: "Database volumes need performance buffers"
  enabled: true
  priority: 100
  logic: "OR"
  conditions:
    - field: "tag.Application"
      operator: "regex"
      value: "(database|db|mysql|postgres)"
    - field: "property.mount_point"
      operator: "contains"
      value: "/var/lib/mysql"
  threshold_overrides:
    iops_utilization_threshold: 50.0
    throughput_utilization_threshold: 50.0
```

## Rule Priority and Logic

### Priority System

- **Higher priority rules** (higher numbers) are processed first
- **Same priority rules** are processed in configuration order
- **Multiple rule matches** can accumulate effects

### Logic Operators

- **AND**: All conditions must be true
- **OR**: Any condition can be true

### Rule Combination

When multiple rules match:

- **Threshold overrides** are merged (later rules override earlier ones)
- **Skip/Force lists** are accumulated
- **Risk adjustments** are accumulated
- **Custom prompts** are accumulated

## Service-Specific Examples

### EC2 Instances

```yaml
# Production instances with high CPU need buffers
- name: "ec2_production_buffer"
  conditions:
    - field: "tag.Environment"
      operator: "equals"
      value: "production"
    - field: "cpu_utilization_p95"
      operator: "greater_than"
      value: 70
  threshold_overrides:
    cpu_low_threshold: 70.0
```

### S3 Buckets

```yaml
# Log buckets for aggressive archival
- name: "s3_log_archival"
  conditions:
    - field: "tag.DataType"
      operator: "equals"
      value: "logs"
  threshold_overrides:
    ia_access_threshold: 7
    glacier_access_threshold: 30
```

### RDS Databases

```yaml
# Multi-AZ databases conservative approach
- name: "rds_multi_az_conservative"
  conditions:
    - field: "property.multi_az"
      operator: "equals"
      value: true
  skip_recommendation_types:
    - "rightsizing"
  risk_adjustment: "increase"
```

### Lambda Functions

```yaml
# High-frequency functions performance focus
- name: "lambda_high_frequency"
  conditions:
    - field: "invocation_count"
      operator: "greater_than"
      value: 1000000
  threshold_overrides:
    memory_utilization_threshold: 40.0
```

### EBS Volumes

```yaml
# Database volumes need IOPS buffers
- name: "ebs_database_performance"
  conditions:
    - field: "tag.Application"
      operator: "regex"
      value: "(database|db)"
  threshold_overrides:
    iops_utilization_threshold: 50.0
```

## Best Practices

### 1. Use Clear Naming

- Use descriptive rule names
- Include affected service in name
- Describe the business logic

### 2. Set Appropriate Priorities

- **100**: Critical business rules
- **90**: High-priority operational rules
- **80**: Cost optimization priorities
- **70**: Service-specific optimizations
- **50**: General optimization rules

### 3. Add Descriptions

- Document why the rule exists
- Explain the business context
- Include contact information if needed

### 4. Test Incrementally

- Start with conservative rules
- Monitor recommendation quality
- Gradually increase aggressiveness

### 5. Use Meaningful Tags

- Ensure resources are properly tagged
- Use consistent tag naming
- Document tag taxonomy

## Monitoring and Debugging

### Enable Debug Logging

```bash
python -m llm_cost_recommendation --log-format json --verbose
```

### Check Rule Application

The logs will show:

- Which rules are evaluated
- Which conditions match
- What overrides are applied
- How recommendations are filtered

### Common Issues

1. **No rules firing**: Check tag spelling and values
2. **Too aggressive**: Increase thresholds gradually
3. **Rules conflicting**: Review priority ordering
4. **Missing fields**: Ensure required data is available

## Advanced Examples

### Complex Multi-Condition Rules

```yaml
- name: "complex_optimization_rule"
  description: "Complex rule with multiple conditions"
  enabled: true
  priority: 75
  logic: "AND"
  conditions:
    # Must be high-cost resource
    - field: "monthly_cost"
      operator: "greater_than"
      value: 1000
    # In specific regions
    - field: "region"
      operator: "in"
      value: ["us-east-1", "us-west-2"]
    # Not critical applications
    - field: "tag.Criticality"
      operator: "not_equals"
      value: "critical"
    # Low utilization
    - field: "cpu_utilization_p95"
      operator: "less_than"
      value: 30
  force_recommendation_types:
    - "rightsizing"
    - "purchasing_option"
  custom_prompt: "High-cost, low-utilization resource in primary regions. Aggressive optimization recommended."
```

### Time-Based Rules

```yaml
- name: "old_resource_cleanup"
  description: "Resources older than 1 year need review"
  enabled: true
  priority: 60
  logic: "AND"
  conditions:
    - field: "age_days"
      operator: "greater_than"
      value: 365
    - field: "tag.Environment"
      operator: "not_equals"
      value: "production"
  custom_prompt: "This resource is over 1 year old. Consider if it's still needed or can be decommissioned."
```

This custom conditions system provides powerful flexibility to adapt the LLM cost recommendation system to your specific organizational requirements and policies.
