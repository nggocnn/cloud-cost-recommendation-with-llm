# FastAPI REST API Documentation

## Overview

The LLM Cost Recommendation System provides a comprehensive REST API built with FastAPI, offering asynchronous processing, automatic documentation, and robust error handling for cloud cost optimization analysis.

## API Endpoints

### Health Check

**GET** `/health`

Returns system health status and component information.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "timestamp": "2025-10-04T07:22:21.539560",
  "components": {
    "coordinator": "healthy",
    "llm_service": "healthy",
    "configuration": "loaded"
  }
}
```

### File Analysis

**POST** `/analyze`

Analyzes cloud resources using uploaded CSV/JSON files.

**Request:** `multipart/form-data`

- `billing_file`: CSV file with billing data
- `inventory_file`: JSON file with resource inventory
- `metrics_file`: CSV file with performance metrics

**Response:**

```json
{
  "analysis_id": "uuid-string",
  "status": "completed",
  "recommendations": [...],
  "summary": {
    "total_resources": 150,
    "total_recommendations": 23,
    "potential_monthly_savings": 2850.45,
    "analysis_duration_ms": 12500
  }
}
```

### Structured Data Analysis

**POST** `/recommendations`

Processes structured data for cost recommendations.

**Request:** `application/json`

```json
{
  "resources": [...],
  "billing_data": [...],
  "metrics_data": [...],
  "options": {
    "analysis_window_days": 30,
    "min_cost_threshold": 1.0,
    "include_low_impact": false
  }
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "id": "rec-uuid",
      "resource_id": "i-1234567890abcdef0",
      "service": "AWS.EC2",
      "recommendation_type": "rightsizing",
      "current_cost_monthly": 150.50,
      "potential_savings_monthly": 75.25,
      "confidence": 0.85,
      "risk": "Low",
      "implementation_steps": [...],
      "analysis": "Detailed analysis text..."
    }
  ],
  "metadata": {
    "analysis_timestamp": "2025-10-04T07:22:21.539560",
    "total_resources_analyzed": 150,
    "agent_coverage": {
      "AWS.EC2": 45,
      "AWS.S3": 20,
      "default": 5
    }
  }
}
```

## Starting the API Server

### Starting the Server

```bash
# Start with default settings
python -m llm_cost_recommendation serve

# Or using uvicorn directly
uvicorn llm_cost_recommendation.api:app --reload

# With custom host and port
python -m llm_cost_recommendation serve --host 127.0.0.1 --port 8080

# With multiple workers for better performance
uvicorn llm_cost_recommendation.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2
```

## Interactive Documentation

The API provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Security Features

### Rate Limiting

The API includes built-in rate limiting to prevent abuse:

```python
# Default rate limits
@limiter.limit("100/minute")  # 100 requests per minute
@limiter.limit("1000/hour")   # 1000 requests per hour
```

### Input Validation

All inputs are validated using Pydantic models:

```python
class AnalysisRequest(BaseModel):
    resources: List[Resource]
    billing_data: Optional[List[BillingRecord]] = None
    metrics_data: Optional[List[MetricsRecord]] = None
    options: Optional[AnalysisOptions] = None
    
    @validator('resources')
    def validate_resources(cls, v):
        if not v:
            raise ValueError("At least one resource is required")
        return v
```

### Error Handling

Comprehensive error handling with structured responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "resources",
      "issue": "Required field missing"
    },
    "timestamp": "2025-10-04T07:22:21.539560"
  }
}
```

## API Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Upload files for analysis
curl -X POST "http://localhost:8000/analyze" \
  -F "billing_file=@data/billing/cli_billing.csv" \
  -F "inventory_file=@data/inventory/cli_inventory.json" \
  -F "metrics_file=@data/metrics/cli_metrics.csv"

# Structured data analysis
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# File upload
files = {
    'billing_file': open('data/billing/cli_billing.csv', 'rb'),
    'inventory_file': open('data/inventory/cli_inventory.json', 'rb'),
    'metrics_file': open('data/metrics/cli_metrics.csv', 'rb')
}
response = requests.post("http://localhost:8000/analyze", files=files)
print(response.json())
```

## Performance Features

### Async Processing

All endpoints use async/await for non-blocking operations:

```python
@app.post("/recommendations")
async def get_recommendations(request: AnalysisRequest):
    # Async processing for better performance
    result = await coordinator.analyze_resources_async(request.resources)
    return result
```

### Concurrent Request Handling

The API supports concurrent request processing with proper resource management:

- **Connection pooling** for external services
- **Request queuing** to prevent resource exhaustion  
- **Memory management** for large datasets
- **Timeout handling** for LLM requests

### Caching

Strategic caching for improved performance:

- **Configuration caching** for agent settings
- **Model validation caching** for repeated requests
- **LLM response caching** for identical analyses

## Monitoring and Observability

### Health Monitoring

The `/health` endpoint provides detailed system status:

```json
{
  "status": "healthy|degraded|unhealthy",
  "components": {
    "coordinator": "healthy",
    "llm_service": "healthy|degraded",
    "configuration": "loaded|error",
    "database": "connected|disconnected"
  },
  "metrics": {
    "uptime_seconds": 3600,
    "requests_total": 1500,
    "active_analyses": 3
  }
}
```

### Logging

Structured logging for all API operations:

```json
{
  "timestamp": "2025-10-04T07:22:21.539560",
  "level": "info",
  "event": "analysis_completed",
  "analysis_id": "uuid-string",
  "duration_ms": 12500,
  "resources_count": 150,
  "recommendations_count": 23
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input data or format |
| `FILE_PROCESSING_ERROR` | Error processing uploaded files |
| `LLM_SERVICE_ERROR` | LLM service unavailable or timeout |
| `CONFIGURATION_ERROR` | Invalid agent or system configuration |
| `RESOURCE_LIMIT_ERROR` | Request exceeds system limits |
| `INTERNAL_ERROR` | Unexpected system error |

## API Metrics

The API tracks comprehensive metrics:

- **Request volume** and **response times**
- **Error rates** by endpoint and error type
- **Resource usage** (memory, CPU)
- **LLM service** performance and costs
- **Analysis quality** metrics (confidence, savings)

Access metrics via the health endpoint or integrate with monitoring tools like Prometheus/Grafana.