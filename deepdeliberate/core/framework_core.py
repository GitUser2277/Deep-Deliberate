"""
Core framework orchestration for the DeepDeliberate framework.

This module implements the main framework controller with state machine pattern
for mode handling, testing session orchestration, and session management.
"""

import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import click

from .models import CLIConfig, ExecutionMode, SessionState, Persona, FrameworkConfig
from .execution_modes import ExecutionModeFactory
from .interfaces import (
    QueryGeneratorInterface, AgentExecutorInterface, 
    EvaluatorInterface, SessionLoggerInterface
)
from .query_generator import DeepSeekQueryGenerator
from .evaluator import DeepSeekEvaluator
from .agent_executor_interface import FrameworkAgentExecutor
from .session_logger import SessionLogger
from .agent_discovery import PydanticAgentDiscovery
from .deepseek_client import DeepSeekConfig
from .exceptions import FrameworkError, ConfigurationError, SessionError
from .logging_config import get_logger

logger = get_logger(__name__)


class FrameworkState(str, Enum):
    """States in the framework state machine."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    SESSION_ACTIVE = "session_active"
    SESSION_PAUSED = "session_paused"
    SESSION_COMPLETED = "session_completed"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class FrameworkCore:
    """
    Main framework controller that orchestrates all components.
    
    Implements a state machine pattern for mode handling and provides
    comprehensive session orchestration with initialization and shutdown procedures.
    
    State Machine:
    UNINITIALIZED -> INITIALIZED -> SESSION_ACTIVE -> SESSION_COMPLETED
                                 -> SESSION_PAUSED -> SESSION_ACTIVE
                                 -> ERROR -> INITIALIZED (recovery)
                                 -> SHUTDOWN
    
    Features:
    - State machine pattern for robust mode handling
    - Component lifecycle management
    - Session initialization and shutdown procedures
    - Error recovery and graceful degradation
    - Resource cleanup and connection pooling
    """
    
    def __init__(
        self,
        query_generator: QueryGeneratorInterface,
        agent_executor: AgentExecutorInterface,
        evaluator: EvaluatorInterface,
        session_logger: SessionLoggerInterface
    ):
        self.query_generator = query_generator
        self.agent_executor = agent_executor
        self.evaluator = evaluator
        self.session_logger = session_logger
        self.config: Optional[FrameworkConfig] = None
        self.current_session: Optional[SessionState] = None
        
        # State machine
        self._state = FrameworkState.UNINITIALIZED
        self._state_history: List[FrameworkState] = []
        self._state_data: Dict[str, Any] = {}
        
        # Component status tracking
        self._component_status: Dict[str, bool] = {
            "query_generator": False,
            "agent_executor": False,
            "evaluator": False,
            "session_logger": False
        }
        
        # Resource management
        self._resources: Dict[str, Any] = {}
        self._cleanup_tasks: List[asyncio.Task] = []
    
    @property
    def state(self) -> FrameworkState:
        """Get current framework state."""
        return self._state
    
    @property
    def state_history(self) -> List[FrameworkState]:
        """Get framework state history."""
        return self._state_history.copy()
    
    def _transition_state(self, new_state: FrameworkState, data: Optional[Dict[str, Any]] = None) -> None:
        """Transition to a new state with validation."""
        valid_transitions = {
            FrameworkState.UNINITIALIZED: [FrameworkState.INITIALIZED, FrameworkState.ERROR],
            FrameworkState.INITIALIZED: [FrameworkState.SESSION_ACTIVE, FrameworkState.ERROR, FrameworkState.SHUTDOWN],
            FrameworkState.SESSION_ACTIVE: [FrameworkState.SESSION_PAUSED, FrameworkState.SESSION_COMPLETED, FrameworkState.ERROR],
            FrameworkState.SESSION_PAUSED: [FrameworkState.SESSION_ACTIVE, FrameworkState.SESSION_COMPLETED, FrameworkState.ERROR],
            FrameworkState.SESSION_COMPLETED: [FrameworkState.SESSION_ACTIVE, FrameworkState.SHUTDOWN],
            FrameworkState.ERROR: [FrameworkState.INITIALIZED, FrameworkState.SHUTDOWN],
            FrameworkState.SHUTDOWN: []  # Terminal state
        }
        
        if new_state not in valid_transitions.get(self._state, []):
            raise FrameworkError(
                f"Invalid state transition from {self._state} to {new_state}. "
                f"Valid transitions: {valid_transitions.get(self._state, [])}"
            )
        
        logger.info(f"State transition: {self._state} -> {new_state}")
        self._state_history.append(self._state)
        self._state = new_state
        
        if data:
            self._state_data.update(data)
    
    async def initialize(self, config: FrameworkConfig) -> None:
        """
        Initialize the framework with configuration and validate all components.
        
        This method performs comprehensive initialization including:
        - Configuration validation
        - Component initialization and health checks
        - Resource allocation
        - State machine initialization
        """
        try:
            logger.info("Starting framework initialization")
            
            # Validate configuration
            if not config:
                raise ConfigurationError("Configuration cannot be None")
            
            self.config = config
            
            # Initialize components with proper async context
            async with asyncio.TaskGroup() as tg:
                init_task = tg.create_task(self._initialize_components())
                health_task = tg.create_task(self._validate_component_health())
                resource_task = tg.create_task(self._initialize_resources())
            
            # Transition to initialized state
            self._transition_state(FrameworkState.INITIALIZED, {
                "initialization_time": datetime.now(),
                "config_hash": hash(str(config.model_dump()))
            })
            
            click.echo("✅ Framework initialized successfully")
            logger.info("Framework initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            self._transition_state(FrameworkState.ERROR, {"error": str(e)})
            raise FrameworkError(f"Framework initialization failed: {e}") from e
    
    async def _initialize_components(self) -> None:
        """Initialize all framework components with proper async handling."""
        components = [
            ("query_generator", self.query_generator),
            ("evaluator", self.evaluator),
            ("session_logger", self.session_logger)
        ]
        
        for name, component in components:
            try:
                # Initialize component if it has an initialize method
                if hasattr(component, 'initialize'):
                    if asyncio.iscoroutinefunction(component.initialize):
                        await component.initialize()
                    else:
                        component.initialize()
                
                self._component_status[name] = True
                logger.debug(f"Component {name} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize component {name}: {e}")
                self._component_status[name] = False
                raise FrameworkError(f"Component initialization failed: {name}") from e
        
        # Note: agent_executor is not initialized here because it requires
        # the agent_file_path parameter. It will be initialized separately
        # by calling initialize_agent_executor() method.
    
    async def initialize_agent_executor(self, agent_file_path: str) -> None:
        """
        Initialize the agent executor with the specified agent file path.
        
        This method must be called after the main framework initialization
        and before starting any testing sessions.
        
        Args:
            agent_file_path: Path to the Python file containing PydanticAI agents
        """
        try:
            if hasattr(self.agent_executor, 'initialize'):
                if asyncio.iscoroutinefunction(self.agent_executor.initialize):
                    await self.agent_executor.initialize(agent_file_path)
                else:
                    self.agent_executor.initialize(agent_file_path)
            
            self._component_status["agent_executor"] = True
            logger.debug("Agent executor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent executor: {e}")
            self._component_status["agent_executor"] = False
            raise FrameworkError(f"Agent executor initialization failed: {e}") from e
    
    async def _validate_component_health(self) -> None:
        """Validate health of all components."""
        # Skip agent_executor validation during main initialization
        # It will be validated after initialize_agent_executor() is called
        required_components = ["query_generator", "evaluator", "session_logger"]
        
        for name in required_components:
            if not self._component_status.get(name, False):
                raise FrameworkError(f"Component {name} is not healthy")
        
        # Additional health checks
        health_checks = [
            ("API connectivity", self._check_api_connectivity),
            ("File system access", self._check_filesystem_access),
            ("Memory availability", self._check_memory_availability)
        ]
        
        for check_name, check_func in health_checks:
            try:
                await check_func()
                logger.debug(f"Health check passed: {check_name}")
            except Exception as e:
                logger.warning(f"Health check failed: {check_name} - {e}")
                # Continue with warnings for non-critical checks
    
    async def _check_api_connectivity(self) -> None:
        """Check API connectivity."""
        # This would be implemented based on the specific API client
        pass
    
    async def _check_filesystem_access(self) -> None:
        """Check filesystem access for logging."""
        import os
        import tempfile
        
        output_dir = self.config.logging_settings.output_directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write access
        test_file = os.path.join(output_dir, ".framework_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise FrameworkError(f"Cannot write to output directory {output_dir}: {e}")
    
    async def _check_memory_availability(self) -> None:
        """Check available memory."""
        import psutil
        
        available_gb = psutil.virtual_memory().available / (1024**3)
        required_gb = self.config.performance_settings.memory_limit_gb
        
        if available_gb < required_gb:
            logger.warning(
                f"Available memory ({available_gb:.1f}GB) is less than "
                f"configured limit ({required_gb}GB)"
            )
    
    async def _initialize_resources(self) -> None:
        """Initialize framework resources."""
        # Connection pool for API requests
        self._resources["connection_pool"] = await self._create_connection_pool()
        
        # Cache for responses
        if self.config.performance_settings.cache_enabled:
            self._resources["response_cache"] = await self._create_response_cache()
        
        # Semaphore for concurrent requests
        max_concurrent = self.config.performance_settings.max_concurrent_requests
        self._resources["request_semaphore"] = asyncio.Semaphore(max_concurrent)
    
    async def _create_connection_pool(self) -> Any:
        """Create connection pool for API requests."""
        # This would create an actual connection pool
        # For now, return a placeholder
        return {"type": "connection_pool", "max_connections": 10}
    
    async def _create_response_cache(self) -> Dict[str, Any]:
        """Create response cache."""
        return {
            "type": "memory_cache",
            "max_size": 1000,
            "ttl": self.config.performance_settings.cache_ttl_seconds,
            "data": {}
        }
    
    async def run_testing_session(self, cli_config: CLIConfig) -> SessionState:
        """
        Run a complete testing session based on CLI configuration.
        
        This method orchestrates the entire testing session with proper state management:
        - Session initialization
        - Component coordination
        - Error handling and recovery
        - Session completion and cleanup
        """
        if self._state != FrameworkState.INITIALIZED:
            raise SessionError(f"Cannot start session in state {self._state}. Framework must be initialized.")
        
        try:
            # Transition to active session state
            self._transition_state(FrameworkState.SESSION_ACTIVE, {
                "session_start_time": datetime.now(),
                "cli_config": cli_config.model_dump()
            })
            
            # Resolve personas with inheritance
            resolved_personas = self.config.resolve_persona_inheritance()
            
            # Select persona
            persona = self._select_persona(resolved_personas, cli_config.persona)
            
            # Create session with proper initialization
            session_state = await self._initialize_session(cli_config, persona)
            self.current_session = session_state
            
            # Create execution mode handler with resource management
            handler = ExecutionModeFactory.create_handler(
                cli_config.mode,
                self.query_generator,
                self.agent_executor,
                self.evaluator,
                self.session_logger
            )
            
            # Execute session with monitoring
            completed_session = await self._execute_session_with_monitoring(handler, session_state, persona)
            
            # Complete session
            await self._complete_session(completed_session)
            
            return completed_session
            
        except KeyboardInterrupt:
            click.echo("\n⚠️  Session interrupted by user")
            await self._handle_session_interruption()
            raise
        except Exception as e:
            logger.error(f"Session execution failed: {e}")
            await self._handle_session_error(e)
            raise
    
    async def _initialize_session(self, cli_config: CLIConfig, persona: Persona) -> SessionState:
        """Initialize a new testing session with comprehensive setup."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create session state
        session_state = SessionState(
            session_id=session_id,
            current_iteration=0,
            total_iterations=cli_config.count,
            mode=cli_config.mode,
            persona=persona,
            completed_interactions=[],
            checkpoint_data={
                "initialization_time": datetime.now().isoformat(),
                "framework_state": self._state.value,
                "resource_status": self._get_resource_status()
            }
        )
        
        # Initialize session logger for this session
        await self.session_logger.initialize_session(session_state.session_id)
        
        # Create initial checkpoint
        await self.session_logger.create_checkpoint(session_state)
        
        click.echo(f"📝 Initialized session: {session_id}")
        click.echo(f"   Mode: {cli_config.mode.value}")
        click.echo(f"   Persona: {persona.name}")
        click.echo(f"   Iterations: {cli_config.count}")
        
        logger.info(f"Session initialized: {session_id}")
        
        return session_state
    
    async def _execute_session_with_monitoring(
        self, 
        handler, 
        session_state: SessionState, 
        persona: Persona
    ) -> SessionState:
        """Execute session with comprehensive monitoring and resource management."""
        start_time = datetime.now()
        
        try:
            # Monitor resource usage
            resource_monitor = asyncio.create_task(self._monitor_resources(session_state))
            
            # Execute the session
            completed_session = await handler.execute_session(session_state, persona)
            
            # Cancel monitoring
            resource_monitor.cancel()
            
            # Log session statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._log_session_statistics(completed_session, execution_time)
            
            return completed_session
            
        except Exception as e:
            # Cancel monitoring on error
            if 'resource_monitor' in locals():
                resource_monitor.cancel()
            raise
    
    async def _monitor_resources(self, session_state: SessionState) -> None:
        """Monitor resource usage during session execution."""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check memory usage
                import psutil
                memory_usage = psutil.virtual_memory().percent
                
                if memory_usage > 90:
                    logger.warning(f"High memory usage: {memory_usage}%")
                    
                # Update session checkpoint data
                session_state.checkpoint_data["last_resource_check"] = {
                    "timestamp": datetime.now().isoformat(),
                    "memory_usage_percent": memory_usage
                }
                
        except asyncio.CancelledError:
            logger.debug("Resource monitoring cancelled")
    
    async def _complete_session(self, session_state: SessionState) -> None:
        """Complete session with proper cleanup and state transition."""
        try:
            # Finalize session logging
            await self.session_logger.finalize_session(session_state)
            
            # Update session state
            session_state.checkpoint_data["completion_time"] = datetime.now().isoformat()
            
            # Create final checkpoint
            await self.session_logger.create_checkpoint(session_state)
            
            # Transition state
            self._transition_state(FrameworkState.SESSION_COMPLETED, {
                "session_id": session_state.session_id,
                "completion_time": datetime.now(),
                "total_interactions": len(session_state.completed_interactions)
            })
            
            click.echo(f"✅ Session completed: {session_state.session_id}")
            logger.info(f"Session completed successfully: {session_state.session_id}")
            
        except Exception as e:
            logger.error(f"Session completion failed: {e}")
            raise SessionError(f"Failed to complete session: {e}") from e
    
    async def _handle_session_interruption(self) -> None:
        """Handle session interruption with proper state management."""
        try:
            if self.current_session:
                # Transition to paused state
                self._transition_state(FrameworkState.SESSION_PAUSED, {
                    "interruption_time": datetime.now(),
                    "reason": "user_interruption"
                })
                
                # Save session state
                await self.session_logger.create_checkpoint(self.current_session)
                click.echo("💾 Session state saved for recovery")
                
                logger.info(f"Session interrupted and saved: {self.current_session.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle session interruption: {e}")
    
    async def _handle_session_error(self, error: Exception) -> None:
        """Handle session errors with proper state management."""
        try:
            # Transition to error state
            self._transition_state(FrameworkState.ERROR, {
                "error_time": datetime.now(),
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
            
            # Save session state if available
            if self.current_session:
                self.current_session.checkpoint_data["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": datetime.now().isoformat()
                }
                await self.session_logger.create_checkpoint(self.current_session)
                
            logger.error(f"Session error handled: {error}")
            
        except Exception as e:
            logger.critical(f"Failed to handle session error: {e}")
    
    async def _log_session_statistics(self, session_state: SessionState, execution_time: float) -> None:
        """Log comprehensive session statistics."""
        stats = {
            "session_id": session_state.session_id,
            "execution_time": execution_time,
            "total_interactions": len(session_state.completed_interactions),
            "successful_interactions": len([
                i for i in session_state.completed_interactions 
                if not i.agent_response.startswith("ERROR:")
            ]),
            "average_score": 0.0,
            "mode": session_state.mode.value,
            "persona": session_state.persona.name
        }
        
        if stats["successful_interactions"] > 0:
            successful_interactions = [
                i for i in session_state.completed_interactions 
                if not i.agent_response.startswith("ERROR:")
            ]
            stats["average_score"] = sum(i.evaluation_score for i in successful_interactions) / len(successful_interactions)
        
        logger.info(f"Session statistics: {stats}")
        
        # Store statistics in session data
        session_state.checkpoint_data["final_statistics"] = stats
    
    def _get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            "connection_pool": bool(self._resources.get("connection_pool")),
            "response_cache": bool(self._resources.get("response_cache")),
            "request_semaphore": bool(self._resources.get("request_semaphore")),
            "active_tasks": len(self._cleanup_tasks)
        }
    
    def _select_persona(self, personas: list[Persona], requested_persona: Optional[str]) -> Persona:
        """Select a persona based on CLI configuration."""
        if requested_persona:
            for persona in personas:
                if persona.name == requested_persona:
                    return persona
            available = [p.name for p in personas]
            raise ValueError(f"Persona '{requested_persona}' not found. Available: {available}")
        else:
            if not personas:
                raise ValueError("No personas defined in configuration")
            selected = personas[0]
            click.echo(f"ℹ️  No persona specified, using: {selected.name}")
            return selected
    
    async def _create_session(self, cli_config: CLIConfig, persona: Persona) -> SessionState:
        """Create a new session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        session_state = SessionState(
            session_id=session_id,
            current_iteration=0,
            total_iterations=cli_config.count,
            mode=cli_config.mode,
            persona=persona,
            completed_interactions=[],
            checkpoint_data={}
        )
        
        click.echo(f"📝 Created new session: {session_id}")
        return session_state
    
    async def recover_session(self, session_id: str) -> SessionState:
        """
        Recover a previously interrupted session.
        
        This method handles session recovery by:
        - Loading session state from checkpoint
        - Validating session integrity
        - Restoring framework state
        - Resuming execution from last checkpoint
        """
        try:
            logger.info(f"Attempting to recover session: {session_id}")
            
            # Load session from checkpoint
            session_state = await self.session_logger.recover_from_checkpoint(session_id)
            if not session_state:
                raise SessionError(f"No checkpoint found for session: {session_id}")
            
            # Validate session integrity
            await self._validate_session_integrity(session_state)
            
            # Restore framework state
            if self._state == FrameworkState.ERROR:
                self._transition_state(FrameworkState.INITIALIZED)
            elif self._state == FrameworkState.SESSION_PAUSED:
                self._transition_state(FrameworkState.SESSION_ACTIVE)
            
            # Update current session
            self.current_session = session_state
            
            click.echo(f"🔄 Session recovered: {session_id}")
            click.echo(f"   Progress: {session_state.current_iteration}/{session_state.total_iterations}")
            click.echo(f"   Completed interactions: {len(session_state.completed_interactions)}")
            
            logger.info(f"Session recovered successfully: {session_id}")
            return session_state
            
        except Exception as e:
            logger.error(f"Session recovery failed: {e}")
            raise SessionError(f"Failed to recover session {session_id}: {e}") from e
    
    async def _validate_session_integrity(self, session_state: SessionState) -> None:
        """Validate the integrity of a recovered session."""
        # Check required fields
        required_fields = ['session_id', 'current_iteration', 'total_iterations', 'mode', 'persona']
        for field in required_fields:
            if not hasattr(session_state, field) or getattr(session_state, field) is None:
                raise SessionError(f"Session integrity check failed: missing {field}")
        
        # Validate iteration bounds
        if session_state.current_iteration < 0 or session_state.current_iteration > session_state.total_iterations:
            raise SessionError(f"Invalid iteration state: {session_state.current_iteration}/{session_state.total_iterations}")
        
        # Validate completed interactions count
        expected_interactions = session_state.current_iteration
        actual_interactions = len(session_state.completed_interactions)
        if actual_interactions > expected_interactions:
            logger.warning(f"More interactions than expected: {actual_interactions} > {expected_interactions}")
    
    async def shutdown(self) -> None:
        """
        Shutdown the framework with proper cleanup.
        
        This method performs comprehensive shutdown including:
        - Session finalization
        - Resource cleanup
        - Component shutdown
        - State transition to terminal state
        """
        try:
            logger.info("Starting framework shutdown")
            
            # Save current session if active
            if self.current_session and self._state in [FrameworkState.SESSION_ACTIVE, FrameworkState.SESSION_PAUSED]:
                click.echo("💾 Saving current session before shutdown...")
                await self.session_logger.create_checkpoint(self.current_session)
            
            # Cancel all cleanup tasks
            for task in self._cleanup_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            
            # Cleanup resources
            await self._cleanup_resources()
            
            # Shutdown components
            await self._shutdown_components()
            
            # Transition to shutdown state
            self._transition_state(FrameworkState.SHUTDOWN, {
                "shutdown_time": datetime.now(),
                "final_session_id": self.current_session.session_id if self.current_session else None
            })
            
            click.echo("✅ Framework shutdown completed")
            logger.info("Framework shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Framework shutdown failed: {e}")
            # Force transition to shutdown state even on error
            self._state = FrameworkState.SHUTDOWN
            raise FrameworkError(f"Framework shutdown failed: {e}") from e
    
    async def _cleanup_resources(self) -> None:
        """Cleanup framework resources."""
        try:
            # Close connection pool
            if "connection_pool" in self._resources:
                connection_pool = self._resources["connection_pool"]
                if hasattr(connection_pool, 'close'):
                    await connection_pool.close()
                logger.debug("Connection pool closed")
            
            # Clear response cache
            if "response_cache" in self._resources:
                self._resources["response_cache"]["data"].clear()
                logger.debug("Response cache cleared")
            
            # Clear all resources
            self._resources.clear()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            raise
    
    async def _shutdown_components(self) -> None:
        """Shutdown all framework components."""
        components = [
            ("session_logger", self.session_logger),
            ("evaluator", self.evaluator),
            ("query_generator", self.query_generator),
            ("agent_executor", self.agent_executor)
        ]
        
        for name, component in components:
            try:
                # Shutdown component if it has a close method
                if hasattr(component, 'close'):
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                
                self._component_status[name] = False
                logger.debug(f"Component {name} shutdown successfully")
                
            except Exception as e:
                logger.error(f"Failed to shutdown component {name}: {e}")
                self._component_status[name] = False
            try:
                # Shutdown component if it has a close method
                if hasattr(component, 'close'):
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                
                self._component_status[name] = False
                logger.debug(f"Component {name} shutdown successfully")
                
            except Exception as e:
                logger.error(f"Failed to shutdown component {name}: {e}")
                # Continue with other components even if one fails
        
        logger.info("Component shutdown completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.shutdown()
        return False
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status information."""
        status = {
            "framework_state": self._state.value,
            "current_session": None,
            "component_status": self._component_status.copy(),
            "resource_status": self._get_resource_status(),
            "state_history": [state.value for state in self._state_history[-5:]]  # Last 5 states
        }
        
        if self.current_session:
            status["current_session"] = {
                "session_id": self.current_session.session_id,
                "mode": self.current_session.mode.value,
                "persona": self.current_session.persona.name,
                "progress": f"{self.current_session.current_iteration}/{self.current_session.total_iterations}",
                "completed_interactions": len(self.current_session.completed_interactions)
            }
        
        return status
    
    async def pause_session(self) -> None:
        """Pause the current active session."""
        if self._state != FrameworkState.SESSION_ACTIVE:
            raise SessionError(f"Cannot pause session in state {self._state}")
        
        if not self.current_session:
            raise SessionError("No active session to pause")
        
        # Transition to paused state
        self._transition_state(FrameworkState.SESSION_PAUSED, {
            "pause_time": datetime.now(),
            "reason": "manual_pause"
        })
        
        # Create checkpoint
        await self.session_logger.create_checkpoint(self.current_session)
        
        click.echo(f"⏸️  Session paused: {self.current_session.session_id}")
        logger.info(f"Session paused: {self.current_session.session_id}")
    
    async def resume_session(self) -> None:
        """Resume a paused session."""
        if self._state != FrameworkState.SESSION_PAUSED:
            raise SessionError(f"Cannot resume session in state {self._state}")
        
        if not self.current_session:
            raise SessionError("No paused session to resume")
        
        # Transition to active state
        self._transition_state(FrameworkState.SESSION_ACTIVE, {
            "resume_time": datetime.now()
        })
        
        click.echo(f"▶️  Session resumed: {self.current_session.session_id}")
        logger.info(f"Session resumed: {self.current_session.session_id}")
    
    @classmethod
    async def create_from_config(
        cls, 
        framework_config: FrameworkConfig, 
        agent_file: str,
        deepseek_config: DeepSeekConfig
    ) -> 'FrameworkCore':
        """
        Factory method to create framework with all components.
        
        Args:
            framework_config: Framework configuration
            agent_file: Path to agent file
            deepseek_config: DeepSeek API configuration
            
        Returns:
            Configured FrameworkCore instance
        """
        # Discover agent
        discovery = PydanticAgentDiscovery()
        agents = discovery.discover_pydantic_agents(agent_file)
        if not agents:
            raise ValueError(f"No agents found in {agent_file}")
        
        agent = agents[0]  # Use first agent found
        
        # Create components
        query_generator = DeepSeekQueryGenerator(deepseek_config)
        evaluator = DeepSeekEvaluator(deepseek_config)
        agent_executor = FrameworkAgentExecutor(agent)
        session_logger = SessionLogger(framework_config.logging_settings.output_directory)
        
        # Create framework
        framework = cls(query_generator, agent_executor, evaluator, session_logger)
        await framework.initialize(framework_config)
        
        return framework
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.shutdown()
        return False
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status information."""
        status = {
            "framework_state": self._state.value,
            "current_session": None,
            "component_status": self._component_status.copy(),
            "resource_status": self._get_resource_status(),
            "state_history": [state.value for state in self._state_history[-5:]]  # Last 5 states
        }
        
        if self.current_session:
            status["current_session"] = {
                "session_id": self.current_session.session_id,
                "current_iteration": self.current_session.current_iteration,
                "total_iterations": self.current_session.total_iterations,
                "mode": self.current_session.mode.value,
                "persona": self.current_session.persona.name
            }
        
        return status