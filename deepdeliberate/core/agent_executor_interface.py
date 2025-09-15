"""
Agent executor interface for the DeepDeliberate framework.

This module provides the interface for executing queries against PydanticAI agents
with proper error handling and response formatting.
"""

import asyncio
import importlib.util
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic_ai import Agent

from .interfaces import AgentExecutorInterface
from .models import AgentResponse, AgentMetadata
from .exceptions import AgentExecutionError, AgentValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


class FrameworkAgentExecutor(AgentExecutorInterface):
    """
    Agent executor that can discover and execute PydanticAI agents.
    
    This class provides the capability to:
    - Discover PydanticAI agents in Python files
    - Execute queries against discovered agents
    - Handle both sync and async agent execution
    - Provide proper error handling and response formatting
    """
    
    def __init__(self):
        self.discovered_agents: Dict[str, Agent] = {}
        self.agent_metadata: Dict[str, AgentMetadata] = {}
        self.current_agent: Optional[Agent] = None
    
    async def initialize(self, agent_file_path: str) -> None:
        """
        Initialize the executor by discovering agents in the specified file.
        
        Args:
            agent_file_path: Path to Python file containing PydanticAI agents
        """
        try:
            agents = await self._discover_agents(agent_file_path)
            
            if not agents:
                raise AgentValidationError(f"No PydanticAI agents found in {agent_file_path}")
            
            # Use the first agent found as the current agent
            self.current_agent = list(agents.values())[0]
            
            logger.info(f"Initialized agent executor with {len(agents)} agent(s)")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent executor: {e}")
            raise AgentExecutionError(f"Agent executor initialization failed: {e}")
    
    async def execute_query(self, query: str) -> AgentResponse:
        """
        Execute a query against the current agent.
        
        Args:
            query: Query string to execute
            
        Returns:
            AgentResponse with execution results
        """
        if not self.current_agent:
            raise AgentExecutionError("No agent initialized. Call initialize() first.")
        
        start_time = datetime.now()
        
        try:
            # Execute the agent query
            result = await self.current_agent.run(query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                text=str(result.data),
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "model": getattr(self.current_agent.model, 'name', 'unknown'),
                    "success": True
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Agent execution failed: {e}")
            
            return AgentResponse(
                text=f"ERROR: {str(e)}",
                execution_time=execution_time,
                metadata={
                    "agent_type": "pydantic_ai",
                    "success": False,
                    "error": str(e)
                }
            )
    
    async def _discover_agents(self, file_path: str) -> Dict[str, Agent]:
        """
        Discover PydanticAI agents in a Python file.
        
        Args:
            file_path: Path to Python file to scan
            
        Returns:
            Dictionary of discovered agents
        """
        agents = {}
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("agent_module", file_path)
            if not spec or not spec.loader:
                raise AgentValidationError(f"Could not load module from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find Agent instances
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, Agent):
                    agents[name] = obj
                    
                    # Create metadata
                    metadata = AgentMetadata(
                        name=name,
                        model=getattr(obj.model, 'name', 'unknown'),
                        output_type=str(getattr(obj, '_result_type', str)),
                        deps_type=str(getattr(obj, '_deps_type', None)),
                        file_path=file_path
                    )
                    
                    self.agent_metadata[name] = metadata
                    logger.info(f"Discovered agent: {name}")
            
            return agents
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            raise AgentValidationError(f"Failed to discover agents in {file_path}: {e}")
    
    def get_agent_metadata(self) -> Dict[str, AgentMetadata]:
        """Get metadata for all discovered agents."""
        return self.agent_metadata.copy()
    
    def set_current_agent(self, agent_name: str) -> None:
        """
        Set the current agent to use for execution.
        
        Args:
            agent_name: Name of the agent to use
        """
        if agent_name not in self.discovered_agents:
            available = list(self.discovered_agents.keys())
            raise AgentValidationError(
                f"Agent '{agent_name}' not found. Available agents: {available}"
            )
        
        self.current_agent = self.discovered_agents[agent_name]
        logger.info(f"Set current agent to: {agent_name}")