"""
FastAPI application for LLM cost recommendation service.
Auto-reload enabled for development.
"""

import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .services.config import ConfigManager
from .services.llm import LLMService
from .services.ingestion import DataIngestionService
from .agents.coordinator import CoordinatorAgent
from .models.types import RiskLevel
from .utils.logging import configure_logging, get_logger
from .models.api_models import (
    APIError,
    HealthResponse,
    StatusResponse,
    AnalysisRequest,
    AnalysisResponse,
    SystemMetrics,
)
from .models.recommendations import RecommendationReport

# Global application state
logger = get_logger(__name__)
app_state = {}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            "Request started",
            method=request.method,
            path=str(request.url.path),
            query_params=str(request.query_params),
            client_host=request.client.host,
            request_id=request_id,
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log successful request
            logger.info(
                "Request completed",
                status_code=response.status_code,
                process_time=round(process_time, 4),
                request_id=request_id,
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                process_time=round(process_time, 4),
            )
            
            # Return structured error response
            error_response = APIError(
                error="Internal server error",
                code="INTERNAL_ERROR",
                details={"request_id": request_id}
            )
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(mode="json"),
                headers={"X-Request-ID": request_id}
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting LLM Cost Recommendation API")
    
    # Initialize services
    try:
        config_dir = app_state.get("config_dir", "config")
        data_dir = app_state.get("data_dir", "data")
        
        config_manager = ConfigManager(config_dir)
        llm_service = LLMService(config_manager.llm_config)
        data_service = DataIngestionService(data_dir)
        coordinator = CoordinatorAgent(config_manager, llm_service)
        
        # Store in app state
        app_state.update({
            "config_manager": config_manager,
            "llm_service": llm_service,
            "data_service": data_service,
            "coordinator": coordinator,
        })
        
        logger.info(
            "Application initialized successfully",
            total_agents=len(coordinator.service_agents),
            enabled_agents=len([a for a in coordinator.service_agents.values() if getattr(a, 'enabled', True)]),
            llm_model=llm_service.model_name if hasattr(llm_service, 'model_name') else 'GPT-4o-mini'
        )
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Cost Recommendation API")


# Create FastAPI app
app = FastAPI(
    title="LLM Cost Recommendation API",
    description="AI-powered cloud cost optimization recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response"""
    error_response = APIError(
        error=exc.detail,
        code=f"HTTP_{exc.status_code}",
        details={"status_code": exc.status_code}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode="json")
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    error_response = APIError(
        error=str(exc),
        code="VALIDATION_ERROR",
        details={"type": "ValueError"}
    )
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error("Unhandled exception", error=str(exc), type=type(exc).__name__)
    error_response = APIError(
        error="An unexpected error occurred",
        code="INTERNAL_ERROR",
        details={"type": type(exc).__name__}
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode="json")
    )


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    components = {}
    
    try:
        # Check coordinator
        coordinator = app_state.get("coordinator")
        if coordinator:
            components["coordinator"] = "healthy"
            components["agents"] = str(len(coordinator.service_agents))
        else:
            components["coordinator"] = "not_initialized"
            
        # Check LLM service
        llm_service = app_state.get("llm_service") 
        if llm_service:
            components["llm_service"] = "healthy"
        else:
            components["llm_service"] = "not_initialized"
            
    except Exception as e:
        components["error"] = str(e)
    
    return HealthResponse(
        status="healthy",
        components=components
    )


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe"""
    coordinator = app_state.get("coordinator")
    if not coordinator:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status"""
    config_manager = app_state.get("config_manager")
    coordinator = app_state.get("coordinator")
    
    if not config_manager or not coordinator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return StatusResponse(
        config={
            "config_dir": str(config_manager.config_dir),
            "enabled_services": config_manager.global_config.enabled_services,
        },
        agents={
            "total": len(coordinator.service_agents),
            "enabled": len([a for a in coordinator.service_agents.values() if getattr(a, 'enabled', True)]),
            "agent_list": [str(key) for key in coordinator.service_agents.keys()]
        }
    )


# System monitoring endpoints
@app.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get system metrics"""
    import psutil
    import platform
    
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage('/').percent,
        load_average=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
        python_version=platform.python_version(),
        platform=platform.system()
    )


# Analysis endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_cost_optimization(request: AnalysisRequest):
    """Run cost optimization analysis"""
    coordinator = app_state.get("coordinator")
    if not coordinator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(
            "Starting cost analysis", 
            resources=len(request.resources),
            billing_records=len(request.billing),
            metrics_records=len(request.metrics),
            individual_processing=request.individual_processing
        )
        
        # Process individual resources or batch
        if request.individual_processing:
            # Process each resource individually
            all_recommendations = []
            
            for resource in request.resources:
                # Get metrics and billing for this resource
                resource_metrics = [m for m in request.metrics if m.resource_id == resource.resource_id]
                resource_billing = [b for b in request.billing if b.resource_id == resource.resource_id]
                
                # Analyze single resource
                single_report = await coordinator.analyze_resources_and_generate_report(
                    [resource],
                    {resource.resource_id: resource_metrics[0] if resource_metrics else None},
                    {resource.resource_id: resource_billing}
                )
                all_recommendations.extend(single_report.recommendations)
                
            final_recommendations = all_recommendations
        else:
            # Batch processing
            # Group billing data by resource_id
            if request.billing:
                from collections import defaultdict
                billing_grouped = defaultdict(list)
                for billing in request.billing:
                    billing_grouped[billing.resource_id].append(billing)
                billing_data = dict(billing_grouped)
            else:
                billing_data = {}
            
            # Group metrics data by resource_id
            metrics_data = {m.resource_id: m for m in request.metrics}
            
            # Run analysis
            report = await coordinator.analyze_resources_and_generate_report(
                request.resources,
                metrics_data,
                billing_data
            )
            final_recommendations = report.recommendations
        
        # Use the report from coordinator (for batch) or create new one (for individual)
        if request.individual_processing:
            # Create a simple report from individual recommendations
            total_monthly_savings = sum(r.estimated_monthly_savings for r in final_recommendations)
            total_annual_savings = sum(r.annual_savings for r in final_recommendations)
            
            # Count risk levels
            low_risk_count = len([r for r in final_recommendations if r.risk_level == RiskLevel.LOW])
            medium_risk_count = len([r for r in final_recommendations if r.risk_level == RiskLevel.MEDIUM])
            high_risk_count = len([r for r in final_recommendations if r.risk_level == RiskLevel.HIGH])
            
            # Group savings by service
            savings_by_service = {}
            for rec in final_recommendations:
                service_name = rec.service.value
                if service_name not in savings_by_service:
                    savings_by_service[service_name] = 0
                savings_by_service[service_name] += rec.estimated_monthly_savings
            
            final_report = RecommendationReport(
                id=f"individual-{uuid.uuid4().hex[:8]}",
                total_monthly_savings=total_monthly_savings,
                total_annual_savings=total_annual_savings,
                total_recommendations=len(final_recommendations),
                recommendations=final_recommendations,
                low_risk_count=low_risk_count,
                medium_risk_count=medium_risk_count,
                high_risk_count=high_risk_count,
                savings_by_service=savings_by_service
            )
        else:
            final_report = report
        
        logger.info(
            "Analysis completed", 
            recommendations=len(final_recommendations),
            potential_savings=f"${final_report.total_monthly_savings:.2f}",
            high_impact_recommendations=len([r for r in final_recommendations if r.risk_level == RiskLevel.HIGH])
        )
        
        return AnalysisResponse(
            request_id=getattr(request, 'state', {}).request_id if hasattr(request, 'state') else str(uuid.uuid4())[:8],
            resources_analyzed=len(request.resources),
            processing_time_seconds=0.0,  # We could track this if needed
            individual_processing=request.individual_processing,
            report=final_report
        )
        
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Recommendation endpoints (placeholder for future implementation)
@app.get("/recommendations/{recommendation_id}")
async def get_recommendation_detail(recommendation_id: str):
    """Get detailed recommendation information (placeholder)"""
    # This would fetch from database in real implementation
    raise HTTPException(status_code=501, detail="Recommendation detail endpoint not implemented")


def get_dependencies():
    """Get service dependencies"""
    return app_state


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
    config_dir: str = "config",
    data_dir: str = "data",
    reload: bool = False,
    log_format: str = "human"
):
    """Run the FastAPI server"""
    # Configure logging
    configure_logging(level=log_level, format_type=log_format, component="api")
    
    logger.info(
        "Starting LLM Cost Recommendation API server",
        host=host,
        port=port,
        workers=workers,
        config_dir=config_dir,
        data_dir=data_dir,
        reload=reload
    )
    
    # Store config in app state
    app_state.update({
        "config_dir": config_dir,
        "data_dir": data_dir,
    })
    
    # Configure uvicorn
    uvicorn_config = {
        "host": host,
        "port": port,
        "log_level": log_level.lower(),
        "access_log": False,  # We handle access logging in middleware
    }
    
    # Add reload for development
    if reload:
        uvicorn_config["reload"] = True
    else:
        uvicorn_config["workers"] = workers
    
    # Run server
    uvicorn.run("llm_cost_recommendation.api:app", **uvicorn_config)


if __name__ == "__main__":
    run_server()