## EC2 (virtual machines)

* **What to record:** Instance type (vCPU/RAM), attached storage (how many volumes, type, size, throughput), and free disk space.
* **What to watch:** CPU and memory usage trends, uptime.
* **Main cost levers:** Purchase option (On-Demand, Reserved Instances, Spot).
* **Also keep:** Resource tags (owner, app, env, cost center).

## EBS (block storage for EC2)

* **What to record:** Volume type, provisioned size, provisioned throughput/IOPS, snapshot status.
* **What to watch:** Read/write operations, bytes read/written, burst credits/throughput usage.
* **Also keep:** Resource tags.

## S3 (object storage)

* **What to record:** Total data stored, storage class by bucket/prefix (e.g., Standard, IA, Glacier), request counts, retention/soft-delete settings, lifecycle rules.
* **What to watch:** Data retrieved/transferred, number of objects.
* **Also keep:** Resource tags.

## EFS (shared file storage)

* **What to record:** Storage class (Standard/IA), any provisioned throughput.
* **What to watch:** Storage used, throughput, read/write I/O bytes.
* **Also keep:** Resource tags.

## RDS (managed databases)

* **What to record:** Engine, instance class, storage type/size, region, backup configuration, replicas, tags.
* **What to watch:** CPU/memory, free storage, read/write IOPS, network throughput.
* **Main cost levers:** Purchase option (On-Demand vs. Reserved).
* **Also keep:** Resource tags.

## RDS snapshots

* **What to record:** Type (automated/manual), size, count, age, lifecycle policy.
* **Also keep:** Resource tags.

## DynamoDB (NoSQL)

* **What to record:** Capacity mode (On-Demand vs. Provisioned), provisioned RCU/WCU if used.
* **What to watch:** Consumed RCU/WCU and table storage.
* **Also keep:** Resource tags.

## Lambda (serverless)

* **What to record:** Memory setting (implies CPU), timeout.
* **What to watch:** Average/percentile execution time, invocation count.
* **Also keep:** Resource tags.

## CloudFront (CDN)

* **What to record:** Price class / edge coverage.
* **What to watch:** Data transfer out, HTTP/HTTPS requests.
* **Also keep:** Resource tags.

## Elastic Load Balancing

* **Shared:** Data transfer charges, tags.
* **ALB:** Hours running, Load Balancer Capacity Units (LCU) per hour, any reservations.
* **NLB:** Hours running, Network Load Balancer Capacity Units (NLCU) per hour, any reservations.
* **GWLB:** Hours running, Gateway Load Balancer Capacity Units (GLCU) per hour.

## Elastic IP

* **What to record:** Whether attached or idle; hours allocated/idle.
* **Also keep:** Resource tags.

## NAT Gateway

* **What to watch:** Hours running and data processed (GB).
* **Also keep:** Resource tags.

## VPC Interface/Gateway Endpoints

* **What to record:** Connected services.
* **What to watch:** Data processed/transferred.

## VPC Peering

* **What to watch:** Data in/out across peering links.

## SQS (queues)

* **What to watch:** API requests, FIFO vs. Standard usage, message payload sizes, features used that affect cost.
* **Also keep:** Resource tags.

## SNS (notifications)

* **What to watch:** Number of notifications by protocol (mobile push, email, HTTP/S, SQS, etc.), API requests, FIFO topic usage, payload sizes.
* **Also keep:** Resource tags.

## Data transfer (cross-cutting)

* **What to watch:** Inter-AZ, inter-region, and internet egress by source/destination.

---

# How to design your multi-LLM agent system (no code)

## 1) Overall architecture

* **Ingestion layer**

  * **Billing:** Start with CSV exports (e.g., AWS cost & usage). Required fields: account, service, resource ID, region/AZ, usage type, usage amount + unit, blended/unblended cost, discounts/credits, and tags.
  * **Inventory:** Periodic snapshots of live resources (EC2, EBS, S3, etc.) with their properties. (currently we can use mock data from json file)
  * **Metrics:** Rolling windows (e.g., 7/30/90 days) of utilization per service (CPU/mem/IOPS/latency/requests). (csv file)
* **Normalization**

  * Map all providers to a **common schema** (Resource, Metrics, Pricing, Tags). Keep provider-specific extras in an “extensions” field.

* **Recommendation engine** (base on LangChain framework, may use LangGraph as well, use OpenAI API configure with configuration from .env file)

  * **Coordinator agent** (orchestrator) that:

    * Routes each resource to the right **service agent** (EC2 agent, S3 agent, etc.).
    * Consolidates recommendations, deduplicates conflicts, and ranks by savings vs. risk.
  * **Service agents** (one per service/provider) that:

    * Read normalized inputs + service-specific metrics.
    * Propose rightsizing, tier changes, lifecycle rules, purchasing options, and architectural tweaks.
    * Service agents must be implemented in config-based entity so that when adding new service, new config file will be add, no code explosion happens
  * **Rules & guardrails layer** (deterministic checks) for “obvious” wins (e.g., idle EIP, unattached EBS, NAT with huge GB).
* **Outputs**

  * Human-readable report: itemized recommendations, estimated monthly saving, risk/impact, and exact “why”.
  * Machine-readable JSON for pipelines/dashboards.

## 2) Prevent “code explosion”

* **Plugin model**

  * Each service agent is a **plugin** discovered via a registry (name, provider, version, capabilities).
  * Shared **base agent contract**: inputs (schema), expected outputs (Recommendation list with cost delta), evaluation rubric.
* **Config-first**

  * Load agents, thresholds, and analysis windows from configuration, not hard-coded.
  * Service agents must be implemented in config-based entity so that when adding new service, new config file will be add, no code explosion happens
* **Shared prompt scaffolding**

  * **Base prompt**: global goals (minimize cost, preserve SLOs), constraints (no downtime unless flagged), and common data dictionary.
  * **Service prompt add-ons**: only the deltas (e.g., S3 storage classes, EBS IOPS semantics).
  * **Response schema**: require structured fields (action, rationale, before/after config, cost delta, risk, rollback).

## 3) Vision module (optional, additive)

* **Input:** Architecture diagrams (PNG/SVG/PDF) or exported graphs.
* **Processing:** A vision LLM extracts components (icons, labels), links (data flows), and annotations (regions/AZs).
* **Fusion:** Cross-check diagram entities with inventory; flag mismatches (e.g., undocumented NAT, overlooked peering).
* **Value:** Improves **data-transfer** and **topology-driven** recommendations (e.g., move workloads to reduce cross-AZ, add Gateway Endpoint to cut NAT egress).

## 4) Exact pricing with recommended configurations

* **For each recommendation**, compute:

  * Current monthly cost (from bill/metrics).
  * Proposed monthly cost (from pricing API using the recommended size/class/commitment).
  * Savings, break-even (if RI/SP), and sensitivity (load growth).
* **Include**: storage request tiers, retrieval charges, per-hour LCU/NLCU/GLCU, NAT GB processed, data transfer matrices, Lambda duration × memory × invokes.

## 5) Let the agents propose rightsizing

* **Inputs to give the agents:**

  * Utilization percentiles (P50/P90/P95), peak windows, sustained vs. spiky patterns.
  * Performance headroom target (e.g., keep P95 CPU < 60%).
  * Business constraints (prod vs. non-prod, maintenance windows).
* **Expected outputs from agents:**

  * Concrete new size/class/tier (e.g., `m7g.large` from `m5.xlarge`, or S3 IA from Standard).
  * Price comparison table and performance implications.
  * Rollout plan (test first, canary, revert path).

## 6) Starting with CSV billing data

* **Minimum columns to include:**

  * `bill_period_start`, `bill_period_end`, `account_id`, `service`, `region`, `availability_zone` (if any), `resource_id` (or line-item resource), `usage_type`, `usage_amount`, `usage_unit`, `unblended_cost`, `amortized_cost`, `credit`, `savings_plan_eligible`, `reservation_applied`, `tags_*`.
* **Transformations:**

  * Aggregate per resource per day.
  * Pivot key usage types (e.g., `DataTransfer-Out-Bytes`, `TimedStorage-ByteHrs`, `ReadRequests`, `WriteRequests`).
  * Join with inventory + metrics by `resource_id` and time.
* **Quality checks:**

  * Missing tags, missing metrics, anomalies (sudden spikes), orphaned spend (no matching resource).

## 7) Extending to new services and providers

* **Service plugin contract** (applies to AWS today; Azure/GCP later):

  * **Metadata:** `provider`, `service`, `version`.
  * **Inputs:** normalized resource doc + metrics + pricing accessor.
  * **Outputs:** list of Recommendations with structured fields.
  * **Capabilities:** which recommendation types it can emit (rightsizing, lifecycle, purchasing option, topology).
* **Provider adapter**

  * Implements: inventory fetch, metrics fetch, pricing lookup, and id mapping.
  * Keeps provider-specific quirks inside the adapter; the rest of the system stays the same.
* **Placeholders**

  * Create empty entries for Azure and GCP in the registry (disabled until configured), and a shared mapping table for “equivalent” services (e.g., EC2 ↔️ VM ↔️ Compute Engine; S3 ↔️ Blob ↔️ GCS).

---