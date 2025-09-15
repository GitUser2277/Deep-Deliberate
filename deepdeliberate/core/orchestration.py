"""
Test session orchestration for the DeepDeliberate framework.

This module provides the main orchestrator that coordinates all components
of the testing pipeline with clear responsibility separation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .interfaces import (
    QueryGeneratorInterface, AgentExecutorInterface, EvaluatorInterface, 
    SessionLoggerInterface
)
from .models import (
    ExecutionMode, Persona, SessionState, TestInteraction, CLIConfig
)
from .execution_modes import ExecutionModeFactory
from .exceptions import ConfigurationError, SessionError
from .settings import FrameworkSettings

logger = logging.getLogger(__name__)


class TestSessionOrchestrator:
    """
    Main orchestrator for test sessions with clear responsibility separation.
    
    This class coordinates all components of the testing pipeline:
    - Query generation using persona specifications
    - Agent execution with proper error handling
    - Response evaluation with configurable criteria
    - Session logging and checkpointing
    
    Features:
    - Clean separation of concerns
    - Comprehensive error handling
    - Session recovery capabilities
    - Progress tracking and metrics
    """
    
    def __init__(
        self,
        settings: FrameworkSettings,
        query_generator: QueryGeneratorInterface,
        agent_executor: AgentExecutorInterface,
        evaluator: EvaluatorInterface,
        session_logger: SessionLoggerInterface
    ):
        self.settings = settings
        self.query_generator = query_generator
        self.agent_executor = agent_executor
        self.evaluator = evaluator
        self.session_logger = session_logger
        
        # Session tracking
        self.active_sessions: Dict[str, SessionState] = {}
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def create_session(
        self,
        cli_config: CLIConfig,
        persona_name: Optional[str] = None
    ) -> SessionState:
        """
        Create a new test session with proper initialization.
        
        Args:
            cli_config: CLI configuration parameters
            persona_name: Optional specific persona to use
            
        Returns:
            Initialized SessionState
            
        Raises:
            ConfigurationError: If configuration is invalid
            SessionError: If session creation fails
        """
        try:
            # Resolve persona
            persona = await self._resolve_persona(persona_name or cli_config.persona)
            
            # Generate session ID
            session_id = self._generate_session_id()
            
            # Initialize session logger
            await self.session_logger.initialize_session(session_id)
            
            # Create session state
            session_state = SessionState(
                session_id=session_id,
                current_iteration=0,
                total_iterations=cli_config.count,
                mode=cli_config.mode,
                persona=persona,
                completed_interactions=[],
                checkpoint_data={}
            )
            
            # Track active session
            self.active_sessions[session_id] = session_state
            self.session_metrics[session_id] = {
                "start_time": datetime.now(),
                "total_queries": cli_config.count,
                "completed_queries": 0,
                "success_rate": 0.0,
                "average_score": 0.0
            }
            
            logger.info(f"Created session {session_id} with persona '{persona.name}'")
            return session_state
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionError(f"Session creation failed: {e}")
    
    async def execute_session(self, session_state: SessionState) -> SessionState:
        """
        Execute a complete test session using appropriate execution mode.
        
        Args:
            session_state: Session to execute
            
        Returns:
            Updated session state with results
        """
        try:
            # Create execution mode handler
            handler = ExecutionModeFactory.create_handler(
                session_state.mode,
                self.query_generator,
                self.agent_executor,
                self.evaluator,
                self.session_logger
            )
            
            # Execute session
            logger.info(f"Starting execution of session {session_state.session_id}")
            updated_state = await handler.execute_session(session_state, session_state.persona)
            
            # Update metrics
            await self._update_session_metrics(updated_state)
            
            # Final checkpoint
            await self.session_logger.create_checkpoint(updated_state)
            
            logger.info(f"Completed session {session_state.session_id}")
            return updated_state
            
        except Exception as e:
            logger.error(f"Session execution failed: {e}")
            # Create emergency checkpoint
            await self.session_logger.create_checkpoint(session_state)
            raise
    
    async def resume_session(self, session_id: str) -> Optional[SessionState]:
        """
        Resume a session from the latest checkpoint.
        
        Args:
            session_id: ID of session to resume
            
        Returns:
            Resumed session state or None if not found
        """
        try:
            checkpoint = await self.session_logger.load_latest_checkpoint(session_id)
            if not checkpoint:
                logger.warning(f"No checkpoint found for session {session_id}")
                return None
            
            # Reconstruct session state from checkpoint
            # This would need to be implemented based on checkpoint format
            logger.info(f"Resumed session {session_id} from checkpoint")
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return None
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a session.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            Session status information or None if not found
        """
        if session_id not in self.active_sessions:
            return None
        
        session_state = self.active_sessions[session_id]
        metrics = self.session_metrics.get(session_id, {})
        
        return {
            "session_id": session_id,
            "mode": session_state.mode.value,
            "persona": session_state.persona.name,
            "progress": {
                "current": session_state.current_iteration,
                "total": session_state.total_iterations,
                "percentage": (session_state.current_iteration / session_state.total_iterations) * 100
            },
            "metrics": metrics,
            "status": "active" if session_state.current_iteration < session_state.total_iterations else "completed"
        }
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with their status."""
        sessions = []
        for session_id in self.active_sessions:
            status = await self.get_session_status(session_id)
            if status:
                sessions.append(status)
        return sessions
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up resources for a completed session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.session_metrics:
            del self.session_metrics[session_id]
        
        logger.info(f"Cleaned up session {session_id}")
    
    async def _resolve_persona(self, persona_name: Optional[str]) -> Persona:
        """Resolve persona from configuration."""
        if not persona_name:
            # Use first persona as default
            resolved_personas = self.settings.resolve_persona_inheritance()
            if not resolved_personas:
                raise ConfigurationError("No personas defined in configuration")
            return resolved_personas[0]
        
        persona = self.settings.get_persona_by_name(persona_name)
        if not persona:
            available_personas = [p.name for p in self.settings.personas]
            raise ConfigurationError(
                f"Persona '{persona_name}' not found. "
                f"Available personas: {available_personas}"
            )
        
        return persona
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{len(self.active_sessions)}"
    
    async def _update_session_metrics(self, session_state: SessionState) -> None:
        """Update session metrics based on completed interactions."""
        if session_state.session_id not in self.session_metrics:
            return
        
        metrics = self.session_metrics[session_state.session_id]
        interactions = session_state.completed_interactions
        
        if interactions:
            # Calculate success rate (non-error responses)
            successful_interactions = [
                i for i in interactions 
                if not i.agent_response.startswith("ERROR:")
            ]
            
            metrics["completed_queries"] = len(interactions)
            metrics["success_rate"] = len(successful_interactions) / len(interactions)
            
            if successful_interactions:
                avg_score = sum(i.evaluation_score for i in successful_interactions) / len(successful_interactions)
                metrics["average_score"] = avg_score
        
        metrics["end_time"] = datetime.now()
        if "start_time" in metrics:
            duration = metrics["end_time"] - metrics["start_time"]
            metrics["duration_seconds"] = duration.total_seconds()


class PersonaBasedQueryGenerator:
    """
    Query generator that creates realistic scenarios based on persona specifications.
    
    This class generates contextually appropriate queries by:
    - Using persona behavioral patterns
    - Applying tone specifications
    - Utilizing scenario templates with parameter injection
    - Maintaining context across query generation
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.generation_cache: Dict[str, List[str]] = {}
    
    async def generate_query(self, persona: Persona, context: Dict[str, Any]) -> 'GeneratedQuery':
        """Generate a query based on persona and context."""
        # Implementation would go here
        # This is a placeholder for the actual query generation logic
        pass


class ComprehensiveResponseEvaluator:
    """
    Response evaluator that assesses agent responses using persona-specific criteria.
    
    This class evaluates responses by:
    - Applying persona-specific evaluation criteria weights
    - Using multi-dimensional scoring (accuracy, relevance, completion, safety)
    - Providing detailed reasoning for scores
    - Maintaining evaluation consistency
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.evaluation_cache: Dict[str, 'EvaluationResult'] = {}
    
    async def evaluate_response(
        self, 
        query: str, 
        response: str, 
        persona: Persona
    ) -> 'EvaluationResult':
        """Evaluate agent response using persona-specific criteria."""
        # Implementation would go here
        # This is a placeholder for the actual evaluation logic
        pass