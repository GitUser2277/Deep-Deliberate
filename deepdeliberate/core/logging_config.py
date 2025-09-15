"""
Centralized logging configuration for the DeepDeliberate framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the DeepDeliberate framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Framework-specific logger
    framework_logger = logging.getLogger("deepdeliberate")
    framework_logger.info(f"Logging configured - Level: {level}, File: {log_file}")
    
    return framework_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Structured logging for better observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_api_request(
        self, 
        endpoint: str, 
        method: str = "POST",
        tokens_used: Optional[int] = None,
        response_time: Optional[float] = None,
        status: str = "success"
    ) -> None:
        """Log API request with structured data."""
        self.logger.info(
            f"API_REQUEST - {method} {endpoint} - "
            f"Status: {status} - "
            f"Tokens: {tokens_used or 'N/A'} - "
            f"Time: {response_time:.3f}s" if response_time else "Time: N/A"
        )
    
    def log_agent_execution(
        self,
        agent_name: str,
        query: str,
        response_length: int,
        execution_time: float,
        success: bool = True
    ) -> None:
        """Log agent execution with metrics."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"AGENT_EXEC - {agent_name} - "
            f"Status: {status} - "
            f"Query: {query[:50]}... - "
            f"Response: {response_length} chars - "
            f"Time: {execution_time:.3f}s"
        )
    
    def log_session_event(
        self,
        session_id: str,
        event_type: str,
        details: Optional[str] = None
    ) -> None:
        """Log session-level events."""
        self.logger.info(
            f"SESSION - {session_id} - "
            f"Event: {event_type} - "
            f"Details: {details or 'N/A'}"
        )