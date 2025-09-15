"""
DeepDeliberate - A lightweight AI agent testing framework.

This package provides comprehensive testing capabilities for Pydantic-based AI agents
through natural language interactions with automated evaluation and logging.

Example:
    >>> from deepdeliberate import run_framework
    >>> from deepdeliberate.core.models import CLIConfig, ExecutionMode
    >>> 
    >>> config = CLIConfig(
    ...     agent_file="my_agent.py",
    ...     mode=ExecutionMode.AUTO,
    ...     count=10
    ... )
    >>> result = await run_framework(config)
"""

__version__ = "0.1.0"
__author__ = "DeepDeliberate Team"
__description__ = "Lightweight AI agent testing framework"

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    # Core functionality exports will be added when main entry point is ready
    # "run_framework",
    # "CLIConfig", 
    # "ExecutionMode"
]