"""
Exception hierarchy for the DeepDeliberate framework.

This module defines a comprehensive exception hierarchy that enables
granular error handling and better debugging capabilities.
"""

from typing import Optional, Dict, Any


class DeepDeliberateError(Exception):
    """Base exception for all DeepDeliberate framework errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class FrameworkError(DeepDeliberateError):
    """General framework error for core operations."""
    pass


class ConfigurationError(DeepDeliberateError):
    """Raised when configuration is invalid or missing."""
    pass


class AgentDiscoveryError(DeepDeliberateError):
    """Raised when agent discovery fails."""
    pass


class RecoverableError(DeepDeliberateError):
    """Base class for errors that can be recovered from."""
    pass


class FatalError(DeepDeliberateError):
    """Base class for errors that require termination."""
    pass


# Query Generation Errors
class QueryGenerationError(RecoverableError):
    """Raised when query generation fails."""
    pass


class PersonaError(ConfigurationError):
    """Raised when persona configuration is invalid."""
    pass


# Agent Execution Errors
class AgentExecutionError(RecoverableError):
    """Raised when agent execution fails."""
    pass


class AgentTimeoutError(AgentExecutionError):
    """Raised when agent execution times out."""
    pass


class AgentValidationError(FatalError):
    """Raised when agent validation fails."""
    pass


# Evaluation Errors
class EvaluationError(RecoverableError):
    """Raised when response evaluation fails."""
    pass


class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation times out."""
    pass


# API Errors
class APIError(RecoverableError):
    """Base class for API-related errors."""
    pass


class APITimeoutError(APIError):
    """Raised when API requests timeout."""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    pass


class APIAuthenticationError(FatalError):
    """Raised when API authentication fails."""
    pass


# Session and Logging Errors
class SessionError(DeepDeliberateError):
    """Base class for session-related errors."""
    pass


class CheckpointError(SessionError):
    """Raised when checkpoint operations fail."""
    pass


class LoggingError(RecoverableError):
    """Raised when logging operations fail."""
    pass


# Security Errors
class SecurityError(FatalError):
    """Base class for security-related errors."""
    pass


class ValidationError(SecurityError):
    """Raised when input validation fails."""
    pass


class SandboxError(SecurityError):
    """Raised when sandbox execution fails."""
    pass


def should_continue_on_error(error: Exception) -> bool:
    """
    Determine if execution should continue after an error.
    
    Args:
        error: The exception that occurred
        
    Returns:
        True if execution can continue, False if it should stop
    """
    # Never continue on fatal errors
    if isinstance(error, FatalError):
        return False
    
    # Continue on recoverable errors
    if isinstance(error, RecoverableError):
        return True
    
    # For unexpected errors, be conservative and stop
    return False


def categorize_error(error: Exception) -> str:
    """
    Categorize an error for logging and metrics.
    
    Args:
        error: The exception to categorize
        
    Returns:
        String category for the error
    """
    if isinstance(error, ConfigurationError):
        return "configuration"
    elif isinstance(error, AgentExecutionError):
        return "agent_execution"
    elif isinstance(error, QueryGenerationError):
        return "query_generation"
    elif isinstance(error, EvaluationError):
        return "evaluation"
    elif isinstance(error, APIError):
        return "api"
    elif isinstance(error, SessionError):
        return "session"
    elif isinstance(error, SecurityError):
        return "security"
    else:
        return "unexpected"