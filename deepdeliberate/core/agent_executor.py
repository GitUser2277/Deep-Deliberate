"""
Agent execution utilities for the DeepDeliberate framework.

This module provides standardized execution of PydanticAI agents
with proper error handling and response formatting.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Optional

from pydantic_ai import Agent

from .models import AgentResponse
from .exceptions import AgentExecutionError, AgentTimeoutError
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "AgentExecutor",
    "execute_agent_sync",
    "execute_agent_async"
]


class AgentExecutor:
    """Standardized executor for PydanticAI agents."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
    
    async def execute_async(
        self, 
        agent: Agent, 
        query: str, 
        deps: Any = None
    ) -> AgentResponse:
        """Execute agent query asynchronously."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.run(query, deps=deps),
                timeout=self.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                text=str(result.data),
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "model": getattr(agent.model, 'name', 'unknown'),
                    "success": True,
                    "deps_provided": deps is not None
                }
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            raise AgentTimeoutError(
                f"Agent execution timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent execution failed: {e}")
            
            return AgentResponse(
                text="",
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "success": False,
                    "error": str(e)
                },
                error=str(e)
            )
    
    def execute_sync(
        self, 
        agent: Agent, 
        query: str, 
        deps: Any = None
    ) -> AgentResponse:
        """Execute agent query synchronously."""
        start_time = time.time()
        
        try:
            # Use run_sync for synchronous execution
            result = agent.run_sync(query, deps=deps)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                text=str(result.data),
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "model": getattr(agent.model, 'name', 'unknown'),
                    "success": True,
                    "deps_provided": deps is not None
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent execution failed: {e}")
            
            return AgentResponse(
                text="",
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "success": False,
                    "error": str(e)
                },
                error=str(e)
            )


# Global executor instance
_default_executor = AgentExecutor()


def execute_agent_sync(agent: Agent, query: str, deps: Any = None) -> AgentResponse:
    """
    Convenience function for synchronous agent execution.
    
    Args:
        agent: PydanticAI Agent instance
        query: Query string to execute
        deps: Optional dependencies for the agent
        
    Returns:
        AgentResponse with execution results
    """
    return _default_executor.execute_sync(agent, query, deps)


async def execute_agent_async(agent: Agent, query: str, deps: Any = None) -> AgentResponse:
    """
    Convenience function for asynchronous agent execution.
    
    Args:
        agent: PydanticAI Agent instance
        query: Query string to execute
        deps: Optional dependencies for the agent
        
    Returns:
        AgentResponse with execution results
    """
    return await _default_executor.execute_async(agent, query, deps)