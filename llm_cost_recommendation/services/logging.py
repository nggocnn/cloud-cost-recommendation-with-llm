"""
Enhanced logging configuration for better readability and debugging.
"""
import sys
import structlog
from typing import Dict, Any
import os
from datetime import datetime


class ColorFormatter:
    """ANSI color codes for terminal output"""
    
    # Color codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def colorize_level(level: str) -> str:
    """Add colors to log levels"""
    level_colors = {
        'debug': f"{ColorFormatter.DIM}{ColorFormatter.CYAN}DEBUG{ColorFormatter.RESET}",
        'info': f"{ColorFormatter.GREEN}INFO{ColorFormatter.RESET}",
        'warning': f"{ColorFormatter.YELLOW}WARN{ColorFormatter.RESET}",
        'error': f"{ColorFormatter.BRIGHT_RED}ERROR{ColorFormatter.RESET}",
        'critical': f"{ColorFormatter.BG_RED}{ColorFormatter.WHITE}CRIT{ColorFormatter.RESET}",
    }
    return level_colors.get(level.lower(), level.upper())


def colorize_service(service: str) -> str:
    """Add colors to service names"""
    service_colors = {
        'EC2': f"{ColorFormatter.BRIGHT_BLUE}EC2{ColorFormatter.RESET}",
        'EBS': f"{ColorFormatter.BRIGHT_MAGENTA}EBS{ColorFormatter.RESET}",
        'S3': f"{ColorFormatter.BRIGHT_GREEN}S3{ColorFormatter.RESET}",
        'RDS': f"{ColorFormatter.BRIGHT_CYAN}RDS{ColorFormatter.RESET}",
        'Lambda': f"{ColorFormatter.BRIGHT_YELLOW}Lambda{ColorFormatter.RESET}",
    }
    return service_colors.get(service, f"{ColorFormatter.CYAN}{service}{ColorFormatter.RESET}")


def human_readable_processor(logger, method_name, event_dict):
    """
    Custom processor for human-readable console output
    """
    # Skip if not for console output
    if not should_use_human_readable():
        return event_dict
    
    # Extract key fields
    timestamp = event_dict.get('timestamp', datetime.now().isoformat())
    level = event_dict.get('level', 'info')
    event = event_dict.get('event', '')
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_str = dt.strftime('%H:%M:%S')
    except:
        time_str = timestamp[:8] if len(timestamp) > 8 else timestamp
    
    # Build the main message
    parts = []
    
    # Add timestamp with dim color
    parts.append(f"{ColorFormatter.DIM}[{time_str}]{ColorFormatter.RESET}")
    
    # Add colored level
    parts.append(f"[{colorize_level(level)}]")
    
    # Add service/agent info if available
    service = event_dict.get('service')
    agent_id = event_dict.get('agent_id')
    
    if service:
        parts.append(f"[{colorize_service(service)}]")
    elif agent_id:
        parts.append(f"[{ColorFormatter.BRIGHT_CYAN}{agent_id}{ColorFormatter.RESET}]")
    
    # Add the main event message
    if event:
        parts.append(f"{ColorFormatter.BOLD}{event}{ColorFormatter.RESET}")
    
    # Build context string from remaining fields
    context_fields = []
    skip_fields = {'timestamp', 'level', 'event', 'service', 'agent_id'}
    
    for key, value in event_dict.items():
        if key not in skip_fields and value is not None:
            if isinstance(value, (int, float)):
                if key.endswith('_count') or key.endswith('_savings') or key == 'count':
                    context_fields.append(f"{ColorFormatter.GREEN}{key}={value}{ColorFormatter.RESET}")
                else:
                    context_fields.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) < 50:
                context_fields.append(f"{key}={ColorFormatter.CYAN}{value}{ColorFormatter.RESET}")
            elif isinstance(value, list):
                context_fields.append(f"{key}=[{len(value)} items]")
            elif isinstance(value, dict):
                context_fields.append(f"{key}={{{len(value)} keys}}")
            else:
                # Truncate long strings
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:97] + "..."
                context_fields.append(f"{key}={str_value}")
    
    # Join all parts
    message = " ".join(parts)
    if context_fields:
        context = " ".join(context_fields)
        message += f" {ColorFormatter.DIM}({context}){ColorFormatter.RESET}"
    
    # Replace the event_dict with our formatted message
    return {"message": message}


def should_use_human_readable() -> bool:
    """Determine if we should use human-readable output"""
    # Use human readable if:
    # 1. Running in a terminal (TTY)
    # 2. Not explicitly disabled via environment variable
    # 3. Not running in JSON mode
    
    if os.getenv('LOG_FORMAT') == 'json':
        return False
    
    if os.getenv('LOG_FORMAT') == 'human':
        return True
    
    # Auto-detect: use human readable if stdout is a TTY
    return sys.stdout.isatty()


def configure_logging(level: str = "INFO", format_type: str = "auto") -> None:
    """
    Configure structured logging with enhanced readability
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format ("auto", "json", "human")
    """
    # Set format type from environment or parameter
    if format_type == "auto":
        use_human = should_use_human_readable()
    else:
        use_human = format_type == "human"
    
    # Configure processors based on output format
    if use_human:
        # Human-readable format for interactive use
        processors = [
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # JSON format for logging systems
        processors = [
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer()
        ]
    
    # Convert level string to number
    level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    log_level = level_map.get(level.upper(), 20)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """Get a configured logger instance"""
    return structlog.get_logger(name)


# Convenience function for quick setup
def setup_logging(verbose: bool = False, json_format: bool = False) -> None:
    """
    Quick setup function for common use cases
    
    Args:
        verbose: Enable DEBUG level logging
        json_format: Force JSON output format
    """
    level = "DEBUG" if verbose else "INFO"
    format_type = "json" if json_format else "auto"
    configure_logging(level=level, format_type=format_type)
