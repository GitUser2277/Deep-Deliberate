"""
PydanticAI agent discovery module.

This module provides functionality to discover PydanticAI Agent instances
in Python files, extract their metadata, and validate compatibility with
the testing framework.
"""

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, List, Optional, Type

from pydantic_ai import Agent

from .interfaces import AgentInterface
from .models import AgentMetadata, AgentResponse

__all__ = [
    "PydanticAgentDiscovery",
    "discover_agents_in_file",
    "get_agent_info"
]


class PydanticAgentDiscovery(AgentInterface):
    """Implementation of agent discovery for PydanticAI agents."""
    
    def __init__(self):
        """Initialize the agent discovery system."""
        self._discovered_agents = {}
        self._module_cache = {}
        self._agent_file_mapping = {}  # Maps agent objects to their file paths
    
    def discover_pydantic_agents(self, agent_path: str) -> List[Agent]:
        """
        Discover PydanticAI Agent instances in the specified Python file.
        
        Args:
            agent_path: Path to the Python file containing agents
            
        Returns:
            List of discovered PydanticAI Agent instances
            
        Raises:
            FileNotFoundError: If the agent file doesn't exist
            ImportError: If the module cannot be imported
            ValueError: If no valid agents are found
        """
        agent_file = Path(agent_path)
        if not agent_file.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")
        
        if not agent_file.suffix == '.py':
            raise ValueError(f"Agent file must be a Python file: {agent_path}")
        
        # Load the module
        module = self._load_module(agent_path)
        
        # Find all Agent instances
        agents = []
        for name, obj in inspect.getmembers(module):
            if self._is_pydantic_agent(obj):
                agents.append(obj)
                self._discovered_agents[f"{module.__name__}.{name}"] = obj
                # Store agent-to-file mapping for efficient lookup
                self._agent_file_mapping[obj] = agent_path
        
        if not agents:
            raise ValueError(f"No PydanticAI Agent instances found in {agent_path}")
        
        return agents
    
    def _load_module(self, file_path: str) -> Any:
        """
        Load a Python module from file path with cache invalidation.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Loaded module object
        """
        file_path_obj = Path(file_path).resolve()
        file_path_str = str(file_path_obj)
        
        # Check if module needs reloading based on file modification time
        if file_path_str in self._module_cache:
            cached_module, cached_mtime = self._module_cache[file_path_str]
            current_mtime = file_path_obj.stat().st_mtime
            
            if current_mtime == cached_mtime:
                return cached_module
            else:
                # File has been modified, remove from cache
                module_name = file_path_obj.stem
                if module_name in sys.modules:
                    del sys.modules[module_name]
        
        module_name = file_path_obj.stem
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules to handle relative imports
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean up sys.modules on failure
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise ImportError(f"Failed to execute module {file_path}: {e}")
        
        # Cache module with modification time
        current_mtime = file_path_obj.stat().st_mtime
        self._module_cache[file_path_str] = (module, current_mtime)
        return module
    
    def _is_pydantic_agent(self, obj: Any) -> bool:
        """
        Check if an object is a PydanticAI Agent instance.
        
        Args:
            obj: Object to check
            
        Returns:
            True if object is a PydanticAI Agent instance
        """
        try:
            return isinstance(obj, Agent)
        except Exception:
            return False
    
    def get_agent_metadata(self, agent: Agent) -> AgentMetadata:
        """
        Extract metadata from a PydanticAI agent.
        
        Args:
            agent: PydanticAI Agent instance
            
        Returns:
            AgentMetadata with extracted information
        """
        if not self._is_pydantic_agent(agent):
            raise ValueError("Object is not a PydanticAI Agent instance")
        
        # Extract agent name (try to find it in discovered agents)
        agent_name = "unknown"
        for name, cached_agent in self._discovered_agents.items():
            if cached_agent is agent:
                agent_name = name.split('.')[-1]
                break
        
        # Extract model information
        model_name = "unknown"
        if hasattr(agent, 'model') and agent.model:
            if hasattr(agent.model, 'name'):
                model_name = agent.model.name
            elif hasattr(agent.model, '__class__'):
                model_name = agent.model.__class__.__name__
            else:
                model_name = str(agent.model)
        
        # Extract output type
        output_type = "Any"
        if hasattr(agent, '_result_type') and agent._result_type:
            output_type = self._get_type_name(agent._result_type)
        
        # Extract deps type
        deps_type = None
        if hasattr(agent, '_deps_type') and agent._deps_type:
            deps_type = self._get_type_name(agent._deps_type)
        
        # Find file path using efficient mapping
        file_path = self._agent_file_mapping.get(agent, "unknown")
        
        return AgentMetadata(
            name=agent_name,
            model=model_name,
            output_type=output_type,
            deps_type=deps_type,
            file_path=file_path
        )
    
    def _get_type_name(self, type_obj: Type) -> str:
        """
        Get a readable name for a type object.
        
        Args:
            type_obj: Type object
            
        Returns:
            String representation of the type
        """
        if hasattr(type_obj, '__name__'):
            return type_obj.__name__
        elif hasattr(type_obj, '_name'):
            return type_obj._name
        else:
            return str(type_obj)
    
    def validate_agent_compatibility(self, agent: Agent) -> bool:
        """
        Validate that an agent is compatible with the testing framework.
        
        Args:
            agent: PydanticAI Agent instance to validate
            
        Returns:
            True if agent is compatible, False otherwise
        """
        if not self._is_pydantic_agent(agent):
            return False
        
        # Check if agent has required methods
        required_methods = ['run_sync', 'run']
        for method in required_methods:
            if not hasattr(agent, method):
                return False
            if not callable(getattr(agent, method)):
                return False
        
        return True
    
    def execute_query(self, agent: Agent, query: str, deps: Any = None) -> AgentResponse:
        """
        Execute a query against a PydanticAI agent.
        
        This method delegates to the AgentExecutor for standardized execution.
        
        Args:
            agent: PydanticAI Agent instance
            query: Query string to execute
            deps: Optional dependencies for the agent
            
        Returns:
            AgentResponse with execution results
        """
        from .agent_executor import execute_agent_sync
        
        if not self.validate_agent_compatibility(agent):
            return AgentResponse(
                text="",
                execution_time=0.0,
                metadata={"error": "Agent not compatible"},
                error="Agent is not compatible with the framework"
            )
        
        return execute_agent_sync(agent, query, deps)


def discover_agents_in_file(file_path: str) -> List[Agent]:
    """
    Convenience function to discover agents in a file.
    
    Args:
        file_path: Path to Python file containing agents
        
    Returns:
        List of discovered PydanticAI Agent instances
    """
    discovery = PydanticAgentDiscovery()
    return discovery.discover_pydantic_agents(file_path)


def get_agent_info(agent: Agent) -> AgentMetadata:
    """
    Convenience function to get agent metadata.
    
    Args:
        agent: PydanticAI Agent instance
        
    Returns:
        AgentMetadata with agent information
    """
    discovery = PydanticAgentDiscovery()
    return discovery.get_agent_metadata(agent)