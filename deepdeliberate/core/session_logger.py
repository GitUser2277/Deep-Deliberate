"""
Session logging implementation for the DeepDeliberate framework.

This module provides comprehensive session logging with CSV output,
checkpointing, and recovery capabilities.
"""

# Standard library imports
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
import aiofiles
import aiocsv

# Local imports
from .interfaces import SessionLoggerInterface
from .models import TestInteraction, SessionState, UserDecision
from .logging_config import get_logger

__all__ = [
    "SessionLogger"
]

logger = get_logger(__name__)


class SessionLogger(SessionLoggerInterface):
    """
    Comprehensive session logger with CSV output and checkpointing.
    
    Features:
    - Immediate CSV writing to prevent data loss
    - Session checkpointing for recovery
    - User decision audit trail
    - Structured metadata logging
    """
    
    def __init__(self, output_directory: str = "test_results"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.interactions_file = None
        self.decisions_file = None
        self.checkpoints_file = None
        
        # In-memory buffers for batch writing
        self._interaction_buffer: List[Dict[str, Any]] = []
        self._decision_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        
    async def initialize_session(self, session_id: str) -> None:
        """
        Initialize logging files for a new session.
        
        Args:
            session_id: Unique identifier for the session
            
        Raises:
            RuntimeError: If file initialization fails
            
        Example:
            >>> logger = SessionLogger()
            >>> await logger.initialize_session("test_session_123")
        """
        # Use the provided session_id directly for filenames to avoid duplication
        # The session_id already contains a timestamp and a unique suffix
        base_name = session_id
        
        self.interactions_file = self.output_directory / f"{base_name}_interactions.csv"
        self.decisions_file = self.output_directory / f"{base_name}_decisions.csv"
        self.checkpoints_file = self.output_directory / f"{base_name}_checkpoints.json"
        
        # Initialize CSV files with headers
        await self._initialize_interactions_csv()
        await self._initialize_decisions_csv()
    
    async def _initialize_interactions_csv(self) -> None:
        """Initialize the interactions CSV file with headers."""
        self.CSV_HEADERS = [
            "timestamp", "session_id", "persona_name", "query", 
            "agent_response", "evaluation_score", "evaluation_reasoning",
            "execution_time", "metadata"
        ]
        
        # Use proper async CSV operations with error handling
        try:
            async with aiofiles.open(self.interactions_file, 'w', encoding='utf-8', newline='') as f:
                writer = aiocsv.AsyncDictWriter(f, fieldnames=self.CSV_HEADERS)
                await writer.writeheader()
        except Exception as e:
            logger.error(f"Failed to initialize interactions CSV: {e}")
            raise
    
    async def _initialize_decisions_csv(self) -> None:
        """Initialize the decisions CSV file with headers."""
        self.DECISION_HEADERS = [
            "timestamp", "session_id", "iteration", "decision", 
            "query", "metadata"
        ]
        
        # Use proper async CSV operations
        async with aiofiles.open(self.decisions_file, 'w', encoding='utf-8', newline='') as f:
            writer = aiocsv.AsyncDictWriter(f, fieldnames=self.DECISION_HEADERS)
            await writer.writeheader()
    
    async def log_interaction(self, interaction: TestInteraction) -> None:
        """Log a test interaction immediately to CSV."""
        if not self.interactions_file:
            raise RuntimeError("Session not initialized. Call initialize_session() first.")
        
        # Prepare row data
        row_data = {
            "timestamp": interaction.timestamp.isoformat(),
            "session_id": interaction.session_id,
            "persona_name": interaction.persona_name,
            "query": interaction.query,
            "agent_response": interaction.agent_response,
            "evaluation_score": interaction.evaluation_score,
            "evaluation_reasoning": interaction.evaluation_reasoning,
            "execution_time": interaction.execution_time,
            "metadata": json.dumps(interaction.metadata, default=str)
        }
        
        # Write immediately to prevent data loss using async CSV
        async with aiofiles.open(self.interactions_file, 'a', encoding='utf-8', newline='') as f:
            writer = aiocsv.AsyncDictWriter(f, fieldnames=self.CSV_HEADERS)
            await writer.writerow(row_data)
    
    async def log_user_decision(
        self,
        session_id: str,
        iteration: int,
        decision: UserDecision,
        query: str
    ) -> None:
        """Log a user decision in approve mode."""
        if not self.decisions_file:
            raise RuntimeError("Session not initialized. Call initialize_session() first.")
        
        # Prepare row data
        row_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "iteration": iteration,
            "decision": decision.value,
            "query": query,
            "metadata": json.dumps({"decision_type": "user_approval"})
        }
        
        # Write immediately using async CSV
        async with aiofiles.open(self.decisions_file, 'a', encoding='utf-8', newline='') as f:
            writer = aiocsv.AsyncDictWriter(f, fieldnames=self.DECISION_HEADERS)
            await writer.writerow(row_data)
    
    async def create_checkpoint(self, session_state: SessionState) -> str:
        """Create a checkpoint and return checkpoint ID."""
        if not self.checkpoints_file:
            raise RuntimeError("Session not initialized. Call initialize_session() first.")
        
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_state.session_id,
            "current_iteration": session_state.current_iteration,
            "total_iterations": session_state.total_iterations,
            "mode": session_state.mode.value,
            "persona_name": session_state.persona.name,
            "completed_interactions_count": len(session_state.completed_interactions),
            "checkpoint_data": session_state.checkpoint_data
        }
        
        # Load existing checkpoints
        checkpoints = []
        if self.checkpoints_file.exists():
            async with aiofiles.open(self.checkpoints_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                if content.strip():
                    checkpoints = json.loads(content)
        
        # Add new checkpoint
        checkpoints.append(checkpoint_data)
        
        # Write back to file
        async with aiofiles.open(self.checkpoints_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(checkpoints, indent=2, default=str))
        
        return checkpoint_id
    
    async def load_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a session."""
        if not self.checkpoints_file or not self.checkpoints_file.exists():
            return None
        
        try:
            async with aiofiles.open(self.checkpoints_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                if not content.strip():
                    return None
                
                checkpoints = json.loads(content)
                
                # Find latest checkpoint for this session
                session_checkpoints = [
                    cp for cp in checkpoints 
                    if cp.get("session_id") == session_id
                ]
                
                if not session_checkpoints:
                    return None
                
                # Return most recent checkpoint
                return max(session_checkpoints, key=lambda x: x["timestamp"])
                
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    async def finalize_session(self, session_state: SessionState) -> None:
        """Finalize session logging with summary and cleanup."""
        if not self.interactions_file:
            raise RuntimeError("Session not initialized. Call initialize_session() first.")
        
        try:
            # Create final checkpoint
            await self.create_checkpoint(session_state)
            
            # Log session summary
            summary = {
                "session_id": session_state.session_id,
                "total_interactions": len(session_state.completed_interactions),
                "successful_interactions": len([
                    i for i in session_state.completed_interactions 
                    if not i.agent_response.startswith("ERROR:")
                ]),
                "mode": session_state.mode.value,
                "persona": session_state.persona.name,
                "completion_time": datetime.now().isoformat()
            }
            
            # Write session summary to separate file with proper async handling
            summary_file = self.output_directory / f"{session_state.session_id}_summary.json"
            async with aiofiles.open(summary_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(summary, indent=2, default=str))
            
            logger.info(f"Session finalized: {summary}")
            
        except Exception as e:
            logger.error(f"Failed to finalize session: {e}")
            raise
    
    async def recover_from_checkpoint(self, session_id: str) -> Optional[SessionState]:
        """
        Recover session from the latest checkpoint.
        
        Args:
            session_id: ID of session to recover
            
        Returns:
            Recovered SessionState or None if not found
            
        Note:
            This is a placeholder implementation. Full session reconstruction
            would require rebuilding the complete SessionState object from
            checkpoint data including persona, completed interactions, etc.
        """
        try:
            checkpoint_data = await self.load_latest_checkpoint(session_id)
            if not checkpoint_data:
                return None
            
            # TODO: Implement full session state reconstruction
            # This would involve:
            # 1. Recreating Persona object from checkpoint data
            # 2. Rebuilding completed_interactions list
            # 3. Restoring session metadata and state
            logger.info(f"Session recovery not fully implemented for: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return None

    def get_session_files(self, session_id: str) -> Dict[str, Path]:
        """Get file paths for a session."""
        return {
            "interactions": self.interactions_file,
            "decisions": self.decisions_file,
            "checkpoints": self.checkpoints_file
        }
