"""
Core interfaces for the DeepDeliberate framework.

This module defines the abstract interfaces that all framework components
must implement to ensure consistent behavior and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    GeneratedQuery, AgentResponse, EvaluationResult, TestInteraction,
    SessionState, UserDecision, Persona
)

__all__ = [
    "QueryGeneratorInterface",
    "AgentExecutorInterface", 
    "EvaluatorInterface",
    "SessionLoggerInterface",
    "LLMClientInterface",
    "AgentInterface"
]


class QueryGeneratorInterface(ABC):
    """Interface for query generation components."""
    
    @abstractmethod
    async def generate_query(
        self,
        persona: Persona,
        context: Dict[str, Any]
    ) -> GeneratedQuery:
        """Generate a query based on persona and context."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the generator."""
        pass


class AgentExecutorInterface(ABC):
    """Interface for agent execution components."""
    
    @abstractmethod
    async def initialize(self, agent_file_path: str) -> None:
        """Initialize the executor with agent discovery."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str) -> AgentResponse:
        """Execute a query against the current agent."""
        pass


class EvaluatorInterface(ABC):
    """Interface for response evaluation components."""
    
    @abstractmethod
    async def evaluate_response(
        self,
        query: str,
        response: str,
        persona: Persona
    ) -> EvaluationResult:
        """Evaluate an agent response using persona-specific criteria."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the evaluator."""
        pass


class SessionLoggerInterface(ABC):
    """Interface for session logging components."""
    
    @abstractmethod
    async def initialize_session(self, session_id: str) -> None:
        """Initialize logging for a new session."""
        pass
    
    @abstractmethod
    async def log_interaction(self, interaction: TestInteraction) -> None:
        """Log a test interaction."""
        pass
    
    @abstractmethod
    async def log_user_decision(
        self,
        session_id: str,
        iteration: int,
        decision: UserDecision,
        query: str
    ) -> None:
        """Log a user decision in approve mode."""
        pass
    
    @abstractmethod
    async def create_checkpoint(self, session_state: SessionState) -> str:
        """Create a checkpoint and return checkpoint ID."""
        pass
    
    @abstractmethod
    async def load_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a session."""
        pass
    
    @abstractmethod
    async def finalize_session(self, session_state: SessionState) -> None:
        """Finalize session logging."""
        pass
    
    @abstractmethod
    async def recover_from_checkpoint(self, session_id: str) -> Optional[SessionState]:
        """Recover session from checkpoint."""
        pass


class LLMClientInterface(ABC):
    """Interface for LLM client implementations."""
    
    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from LLM."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close client resources."""
        pass


class AgentInterface(ABC):
    """Interface for agent discovery and execution components."""
    
    @abstractmethod
    def discover_pydantic_agents(self, agent_path: str) -> List[Any]:
        """Discover PydanticAI agents in a Python file."""
        pass
    
    @abstractmethod
    def get_agent_metadata(self, agent: Any) -> 'AgentMetadata':
        """Extract metadata from an agent."""
        pass
    
    @abstractmethod
    def validate_agent_compatibility(self, agent: Any) -> bool:
        """Validate agent compatibility with framework."""
        pass
    
    @abstractmethod
    def execute_query(self, agent: Any, query: str, deps: Any = None) -> AgentResponse:
        """Execute a query against an agent."""
        pass