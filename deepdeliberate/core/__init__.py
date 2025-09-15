"""
Core framework components for DeepDeliberate.

This module contains the essential components for agent testing:
- Agent interface and discovery
- Query generation and evaluation
- Session management and logging
- Configuration management
"""

from .models import (
    ExecutionMode,
    UserDecision,
    Persona,
    FrameworkConfig,
    CLIConfig,
    TestInteraction,
    SessionState,
    GeneratedQuery,
    AgentResponse,
    EvaluationResult
)

from .interfaces import (
    AgentInterface,
    QueryGeneratorInterface,
    AgentExecutorInterface,
    EvaluatorInterface,
    SessionLoggerInterface
)

from .exceptions import (
    DeepDeliberateError,
    ConfigurationError,
    AgentDiscoveryError,
    RecoverableError,
    FatalError,
    QueryGenerationError,
    AgentExecutionError,
    EvaluationError
)

from .execution_modes import (
    ExecutionModeHandler,
    AutoModeHandler,
    ApproveModeHandler,
    ExecutionModeFactory
)

from .session_logger import SessionLogger
from .framework_core import FrameworkCore
from .response_parser import (
    DeepSeekResponseParser,
    ParsedResponse,
    ReasoningContent,
    ParseError,
    ParseErrorType
)

__all__ = [
    # Models
    "ExecutionMode",
    "UserDecision", 
    "Persona",
    "FrameworkConfig",
    "CLIConfig",
    "TestInteraction",
    "SessionState",
    "GeneratedQuery",
    "AgentResponse",
    "EvaluationResult",
    
    # Interfaces
    "AgentInterface",
    "QueryGeneratorInterface",
    "AgentExecutorInterface", 
    "EvaluatorInterface",
    "SessionLoggerInterface",
    
    # Exceptions
    "DeepDeliberateError",
    "ConfigurationError",
    "AgentDiscoveryError",
    "RecoverableError",
    "FatalError",
    "QueryGenerationError",
    "AgentExecutionError",
    "EvaluationError",
    
    # Core Components
    "ExecutionModeHandler",
    "AutoModeHandler",
    "ApproveModeHandler",
    "ExecutionModeFactory",
    "SessionLogger",
    "FrameworkCore",
    
    # Response Parsing
    "DeepSeekResponseParser",
    "ParsedResponse", 
    "ReasoningContent",
    "ParseError",
    "ParseErrorType"
]