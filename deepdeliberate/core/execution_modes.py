"""
Execution mode handlers for the DeepDeliberate framework.

This module implements the auto and approve execution modes,
handling user interaction and session management.
"""

# Add imports for loading animations and better model info
import asyncio
import click
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from .interfaces import (
    QueryGeneratorInterface, AgentExecutorInterface, 
    EvaluatorInterface, SessionLoggerInterface
)
from .models import (
    ExecutionMode, Persona, SessionState, TestInteraction, 
    GeneratedQuery, UserDecision
)
from .exceptions import RecoverableError, FatalError, categorize_error, should_continue_on_error
from .logging_config import get_logger
# Import moved to function level to avoid circular dependency issues

logger = get_logger(__name__)


class LoadingSpinner:
    """A simple loading spinner for CLI operations."""
    
    def __init__(self, message: str = "Loading", model_name: str = ""):
        self.message = message
        self.model_name = model_name
        self.spinning = False
        self.thread = None
    
    def start(self):
        """Start the loading spinner."""
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the loading spinner."""
        self.spinning = False
        if self.thread:
            self.thread.join()
        click.echo("\r" + " " * 80 + "\r", nl=False)  # Clear the line
    
    def _spin(self):
        """The spinner animation logic."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.spinning:
            model_info = f" [{self.model_name}]" if self.model_name else ""
            click.echo(f"\r{chars[i]} {self.message}{model_info}...", nl=False)
            i = (i + 1) % len(chars)
            time.sleep(0.1)


class ExecutionModeHandler:
    """
    Base class for execution mode handlers.
    
    This abstract base class defines the interface for handling different
    execution modes (auto and approve) in the DeepDeliberate framework.
    
    Attributes:
        query_generator: Interface for generating test queries
        agent_executor: Interface for executing queries against agents
        evaluator: Interface for evaluating agent responses
        session_logger: Interface for logging session data
        
    Example:
        >>> handler = AutoModeHandler(query_gen, agent_exec, evaluator, logger)
        >>> session_state = await handler.execute_session(state, persona)
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
    
    async def execute_session(
        self,
        session_state: SessionState,
        persona: Persona
    ) -> SessionState:
        """Execute a testing session. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute_session")
    
    async def _handle_recoverable_error(
        self,
        error: RecoverableError,
        session_state: SessionState
    ) -> None:
        """Handle recoverable errors with appropriate logging and state updates."""
        # Log the error for debugging
        logger.warning(f"Handling recoverable error: {error}")
        
        # Update session state with error information
        session_state.checkpoint_data["last_error"] = {
            "type": type(error).__name__,
            "message": error.message,
            "details": error.details,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create checkpoint to preserve state
        await self.session_logger.create_checkpoint(session_state)
    
    def _display_interaction_result(self, interaction: TestInteraction) -> None:
        """Display the result of a completed interaction."""
        # Use standardized renderer (which includes its own header)
        from .deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        renderer.display_interaction_summary(
            interaction.query,
            interaction.agent_response,
            interaction.evaluation_score,
            interaction.evaluation_reasoning,
            interaction.execution_time
        )
    
    def _get_model_info(self, component_name: str) -> str:
        """Get model information for display purposes."""
        if component_name == "query_generator":
            return "DeepSeek R1"
        elif component_name == "agent_executor":
            return "Azure OpenAI o4-mini"
        elif component_name == "evaluator":
            return "DeepSeek R1"
        else:
            return "Unknown"
    
    def _categorize_error_source(self, error: Exception) -> str:
        """Categorize error by its likely source (DeepSeek vs Azure OpenAI)."""
        error_str = str(error).lower()
        if "deepseek" in error_str:
            return "DeepSeek API"
        elif "azure" in error_str and "openai" in error_str:
            return "Azure OpenAI API"
        elif "429" in error_str:
            return "Rate Limit (Provider)"
        else:
            return "API"
    
    async def _execute_single_interaction(
        self,
        session_state: SessionState,
        persona: Persona,
        query: GeneratedQuery
    ) -> TestInteraction:
        """Execute a single test interaction."""
        start_time = datetime.now()
        
        try:
            # Step 1: Execute query against agent (Azure OpenAI o4-mini)
            agent_spinner = LoadingSpinner("Executing agent query", self._get_model_info("agent_executor"))
            agent_spinner.start()
            
            try:
                agent_response = await self.agent_executor.execute_query(query.text)
                agent_spinner.stop()
                
                if agent_response.error:
                    error_source = self._categorize_error_source(Exception(agent_response.error))
                    click.echo(f"🚫 {error_source} Error: {agent_response.error}")
                else:
                    click.echo(f"✅ Agent response received from {self._get_model_info('agent_executor')}")
                    
                    # Show the agent response using standardized renderer
                    from .deepseek_display import get_display_renderer
                    renderer = get_display_renderer()
                    renderer.display_agent_response(
                        agent_response.text, 
                        f"{self._get_model_info('agent_executor')} Response"
                    )
                    
            except Exception as e:
                agent_spinner.stop()
                error_source = self._categorize_error_source(e)
                click.echo(f"🚫 {error_source} Error during agent execution: {str(e)}")
                raise
            
            # Step 2: Evaluate the response (DeepSeek R1)
            eval_spinner = LoadingSpinner("Evaluating response", self._get_model_info("evaluator"))
            eval_spinner.start()
            
            try:
                evaluation = await self.evaluator.evaluate_response(
                    query.text,
                    agent_response.text,
                    persona
                )
                eval_spinner.stop()
                click.echo(f"✅ Evaluation completed using {self._get_model_info('evaluator')}")
                
            except Exception as e:
                eval_spinner.stop()
                error_source = self._categorize_error_source(e)
                click.echo(f"🚫 {error_source} Error during evaluation: {str(e)}")
                
                # Create evaluation with error info
                from .models import EvaluationResult
                evaluation = EvaluationResult(
                    overall_score=0.0,
                    dimension_scores={},
                    reasoning=f"Evaluation failed: {str(e)}",
                    confidence=0.0,
                    evaluation_time=0.0
                )
            
            # Create interaction record
            interaction = TestInteraction(
                timestamp=start_time,
                session_id=session_state.session_id,
                persona_name=persona.name,
                query=query.text,
                agent_response=agent_response.text,
                evaluation_score=evaluation.overall_score,
                evaluation_reasoning=evaluation.reasoning,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "query_metadata": query.metadata,
                    "agent_metadata": agent_response.metadata,
                    "evaluation_metadata": {
                        "dimension_scores": evaluation.dimension_scores,
                        "confidence": evaluation.confidence,
                        "evaluation_time": evaluation.evaluation_time
                    }
                }
            )
            
            # Log interaction immediately
            await self.session_logger.log_interaction(interaction)
            
            return interaction
            
        except Exception as e:
            # Create error interaction record
            error_source = self._categorize_error_source(e)
            interaction = TestInteraction(
                timestamp=start_time,
                session_id=session_state.session_id,
                persona_name=persona.name,
                query=query.text,
                agent_response=f"ERROR: {str(e)}",
                evaluation_score=0.0,
                evaluation_reasoning=f"Execution failed: {error_source} - {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "error": str(e),
                    "error_source": error_source,
                    "query_metadata": query.metadata
                }
            )
            
            # Log error interaction
            await self.session_logger.log_interaction(interaction)
            
            return interaction


class AutoModeHandler(ExecutionModeHandler):
    """
    Handler for fully automated execution mode.
    
    This handler executes all queries sequentially without user intervention,
    providing progress feedback and comprehensive error handling.
    
    Features:
    - Progress bar with ETA estimation
    - Automatic checkpointing every 5 interactions
    - Graceful error handling with continuation
    - Summary statistics calculation
    
    Example:
        >>> handler = AutoModeHandler(query_gen, agent_exec, evaluator, logger)
        >>> result = await handler.execute_session(session_state, persona)
        >>> print(f"Completed {len(result.completed_interactions)} interactions")
    """
    
    async def execute_session(
        self,
        session_state: SessionState,
        persona: Persona
    ) -> SessionState:
        """Execute all queries sequentially without user intervention."""
        click.echo(f"\n🚀 Starting AUTO mode execution...")
        click.echo(f"   Persona: {persona.name}")
        click.echo(f"   Total queries: {session_state.total_iterations}")
        click.echo(f"   Session ID: {session_state.session_id}")
        click.echo(f"\n📋 Model Configuration:")
        click.echo(f"   • Query Generation: {self._get_model_info('query_generator')}")
        click.echo(f"   • Agent Execution: {self._get_model_info('agent_executor')}")
        click.echo(f"   • Response Evaluation: {self._get_model_info('evaluator')}")
        
        # Create progress bar
        with click.progressbar(
            range(session_state.current_iteration, session_state.total_iterations),
            label="Executing queries",
            show_eta=True,
            show_percent=True
        ) as progress_bar:
            
            for iteration in progress_bar:
                try:
                    # Generate query with loading indicator
                    query_spinner = LoadingSpinner("Generating query", self._get_model_info("query_generator"))
                    query_spinner.start()
                    
                    query = await self.query_generator.generate_query(
                        persona,
                        {"iteration": iteration + 1, "total": session_state.total_iterations}
                    )
                    query_spinner.stop()
                    click.echo(f"✅ Query generated using {self._get_model_info('query_generator')}")
                    
                    # Display generated query using standardized renderer (same as approve mode)
                    click.echo("\n" + "="*80)
                    click.echo(f"Query {iteration + 1} | Persona: {persona.name}")
                    click.echo("="*80)
                    from .deepseek_display import get_display_renderer
                    renderer = get_display_renderer()
                    renderer.display_query_generation(
                        query.text, 
                        query.metadata or {}, 
                        thinking_content=query.thinking_content
                    )
                    
                    # Execute interaction
                    interaction = await self._execute_single_interaction(
                        session_state, persona, query
                    )
                    
                    # Update session state
                    session_state.completed_interactions.append(interaction)
                    session_state.current_iteration = iteration + 1
                    
                    # Display interaction results (same as approve mode)
                    self._display_interaction_result(interaction)
                    
                    # Create checkpoint every 5 interactions
                    if (iteration + 1) % 5 == 0:
                        await self.session_logger.create_checkpoint(session_state)
                    
                except RecoverableError as e:
                    logger.warning(f"Recoverable error in iteration {iteration + 1}: {e}")
                    click.echo(f"\n⚠️  Recoverable error in iteration {iteration + 1}: {e.message}")
                    await self._handle_recoverable_error(e, session_state)
                    continue
                except FatalError as e:
                    logger.error(f"Fatal error in iteration {iteration + 1}: {e}")
                    click.echo(f"\n❌ Fatal error in iteration {iteration + 1}: {e.message}")
                    await self.session_logger.create_checkpoint(session_state)
                    raise
                except Exception as e:
                    error_category = categorize_error(e)
                    error_source = self._categorize_error_source(e)
                    logger.error(f"Unexpected {error_category} error in iteration {iteration + 1}: {e}")
                    click.echo(f"\n❌ {error_source} error in iteration {iteration + 1}: {str(e)}")
                    
                    # Only continue for truly unexpected errors after logging
                    if should_continue_on_error(e):
                        await self._handle_recoverable_error(RecoverableError(str(e)), session_state)
                        continue
                    raise
        
        click.echo(f"\n✅ AUTO mode execution completed!")
        click.echo(f"   Completed interactions: {len(session_state.completed_interactions)}")
        
        # Calculate summary statistics
        successful_interactions = [
            i for i in session_state.completed_interactions 
            if not i.agent_response.startswith("ERROR:")
        ]
        
        if successful_interactions:
            avg_score = sum(i.evaluation_score for i in successful_interactions) / len(successful_interactions)
            click.echo(f"   Average evaluation score: {avg_score:.2f}")
        
        return session_state


class ApproveModeHandler(ExecutionModeHandler):
    """
    Handler for step-by-step approval execution mode.
    
    This handler executes queries with user approval for each step,
    providing full control over the testing process.
    
    Features:
    - Interactive query approval with continue/retry/exit options
    - Real-time result display after each interaction
    - User decision audit logging
    - Session checkpointing after each approved query
    - Graceful error handling with user choice
    
    Example:
        >>> handler = ApproveModeHandler(query_gen, agent_exec, evaluator, logger)
        >>> result = await handler.execute_session(session_state, persona)
        >>> # User will be prompted for each query: (c)ontinue, (r)etry, (e)xit
    """
    
    async def execute_session(
        self,
        session_state: SessionState,
        persona: Persona
    ) -> SessionState:
        """Execute queries with user approval for each step."""
        click.echo(f"\n👤 Starting APPROVE mode execution...")
        click.echo(f"   Persona: {persona.name}")
        click.echo(f"   Total queries: {session_state.total_iterations}")
        click.echo(f"   Session ID: {session_state.session_id}")
        click.echo(f"\n📋 Model Configuration:")
        click.echo(f"   • Query Generation: {self._get_model_info('query_generator')}")
        click.echo(f"   • Agent Execution: {self._get_model_info('agent_executor')}")
        click.echo(f"   • Response Evaluation: {self._get_model_info('evaluator')}")
        click.echo(f"\n   Options for each query:")
        click.echo(f"   • (c)ontinue - Execute this query and proceed")
        click.echo(f"   • (r)etry - Generate a new query with same persona")
        click.echo(f"   • (e)exit - Save progress and exit session")
        
        iteration = session_state.current_iteration
        
        while iteration < session_state.total_iterations:
            try:
                # Generate query with loading indicator
                query_spinner = LoadingSpinner("Generating query", self._get_model_info("query_generator"))
                query_spinner.start()
                
                query = await self.query_generator.generate_query(
                    persona,
                    {"iteration": iteration + 1, "total": session_state.total_iterations}
                )
                query_spinner.stop()
                click.echo(f"✅ Query generated using {self._get_model_info('query_generator')}")
                
                # Display query for approval
                decision = self._display_query_approval(query, persona, iteration + 1)
                
                # Log user decision
                await self.session_logger.log_user_decision(
                    session_state.session_id,
                    iteration + 1,
                    decision,
                    query.text
                )
                
                if decision == UserDecision.CONTINUE:
                    # Execute agent step by step with approval
                    interaction = await self._execute_interaction_with_approval(
                        session_state, persona, query
                    )
                    
                    if interaction:  # Only proceed if interaction was completed
                        # Update session state
                        session_state.completed_interactions.append(interaction)
                        session_state.current_iteration = iteration + 1
                        
                        # Display final results
                        self._display_interaction_result(interaction)
                        
                        # Ask user for approval to continue to next iteration
                        next_decision = self._display_continue_approval(iteration + 1, session_state.total_iterations)
                        
                        if next_decision == UserDecision.EXIT:
                            click.echo("💾 Saving progress and exiting...")
                            await self.session_logger.create_checkpoint(session_state)
                            break
                        elif next_decision == UserDecision.RETRY:
                            click.echo("🔄 Retrying current iteration...")
                            # Stay on same iteration, will regenerate everything
                            continue
                        elif next_decision == UserDecision.CONTINUE:
                            # Continue to next iteration
                            # Create checkpoint
                            await self.session_logger.create_checkpoint(session_state)
                            iteration += 1
                    else:
                        # If interaction was None, user chose to exit during the process
                        click.echo("💾 Saving progress and exiting...")
                        await self.session_logger.create_checkpoint(session_state)
                        break
                    
                elif decision == UserDecision.RETRY:
                    click.echo("🔄 Regenerating query with same persona...")
                    # Stay on same iteration, will regenerate query
                    continue
                    
                elif decision == UserDecision.EXIT:
                    click.echo("💾 Saving progress and exiting...")
                    await self.session_logger.create_checkpoint(session_state)
                    break
                    
            except RecoverableError as e:
                logger.warning(f"Recoverable error in iteration {iteration + 1}: {e}")
                click.echo(f"\n⚠️  Recoverable error in iteration {iteration + 1}: {e.message}")
                
                user_choice = click.prompt(
                    "Error occurred. Choose action",
                    type=click.Choice(['c', 'r', 'e']),
                    show_choices=True,
                    show_default=False
                ).lower()
                
                if user_choice == 'c':
                    await self._handle_recoverable_error(e, session_state)
                    continue
                elif user_choice == 'r':
                    continue  # Retry current iteration
                elif user_choice == 'e':
                    await self.session_logger.create_checkpoint(session_state)
                    break
                    
            except FatalError as e:
                logger.error(f"Fatal error in iteration {iteration + 1}: {e}")
                click.echo(f"\n❌ Fatal error in iteration {iteration + 1}: {e.message}")
                await self.session_logger.create_checkpoint(session_state)
                raise
                
            except Exception as e:
                error_category = categorize_error(e)
                error_source = self._categorize_error_source(e)
                logger.error(f"Unexpected {error_category} error in iteration {iteration + 1}: {e}")
                click.echo(f"\n❌ {error_source} error in iteration {iteration + 1}: {str(e)}")
                
                user_choice = click.prompt(
                    "Error occurred. Choose action",
                    type=click.Choice(['c', 'r', 'e']),
                    show_choices=True,
                    show_default=False
                ).lower()
                
                if user_choice == 'c':
                    # Try to continue
                    if should_continue_on_error(e):
                        await self._handle_recoverable_error(RecoverableError(str(e)), session_state)
                        continue
                    else:
                        click.echo("Cannot continue after this error type.")
                        break
                elif user_choice == 'r':
                    continue  # Retry current iteration
                elif user_choice == 'e':
                    await self.session_logger.create_checkpoint(session_state)
                    break
        
        click.echo(f"\n✅ APPROVE mode execution completed!")
        click.echo(f"   Completed interactions: {len(session_state.completed_interactions)}")
        
        # Calculate summary statistics
        successful_interactions = [
            i for i in session_state.completed_interactions 
            if not i.agent_response.startswith("ERROR:")
        ]
        
        if successful_interactions:
            avg_score = sum(i.evaluation_score for i in successful_interactions) / len(successful_interactions)
            click.echo(f"   Average evaluation score: {avg_score:.2f}")
        
        return session_state
    
    async def _execute_interaction_with_approval(
        self,
        session_state: SessionState,
        persona: Persona,
        query: GeneratedQuery
    ) -> Optional[TestInteraction]:
        """Execute a single interaction with user approval between steps."""
        start_time = datetime.now()
        
        # Loop for agent response retry
        while True:
            try:
                # Step 1: Execute query against agent (Azure OpenAI o4-mini)
                agent_spinner = LoadingSpinner("Executing agent query", self._get_model_info("agent_executor"))
                agent_spinner.start()
                
                agent_response = await self.agent_executor.execute_query(query.text)
                agent_spinner.stop()
                
                if agent_response.error:
                    error_source = self._categorize_error_source(Exception(agent_response.error))
                    click.echo(f"🚫 {error_source} Error: {agent_response.error}")
                    return None
                else:
                    click.echo(f"✅ Agent response received from {self._get_model_info('agent_executor')}")
                    
                    # Show the agent response using standardized renderer
                    from .deepseek_display import get_display_renderer
                    renderer = get_display_renderer()
                    renderer.display_agent_response(
                        agent_response.text, 
                        f"{self._get_model_info('agent_executor')} Response"
                    )
                    
                    # Ask user for approval to proceed with evaluation  
                    eval_decision = self._display_evaluation_approval()
                    
                    if eval_decision == UserDecision.EXIT:
                        click.echo("💾 Exiting without evaluation...")
                        return None
                    elif eval_decision == UserDecision.RETRY:
                        click.echo("🔄 Regenerating agent response...")
                        # Continue the outer loop to regenerate agent response
                        continue
                    elif eval_decision == UserDecision.CONTINUE:
                        # Break out of agent retry loop to continue with evaluation
                        break
                    
            except Exception as e:
                agent_spinner.stop()
                error_source = self._categorize_error_source(e)
                click.echo(f"🚫 {error_source} Error during agent execution: {str(e)}")
                return None
        
        # Step 2: Evaluate the response (DeepSeek R1) - now outside agent retry loop
        # Loop for evaluation retry
        while True:
            eval_spinner = LoadingSpinner("Evaluating response", self._get_model_info("evaluator"))
            eval_spinner.start()
            
            try:
                evaluation = await self.evaluator.evaluate_response(
                    query.text,
                    agent_response.text,
                    persona
                )
                eval_spinner.stop()
                click.echo(f"✅ Evaluation completed using {self._get_model_info('evaluator')}")
                break  # Exit evaluation retry loop on success
                
            except Exception as e:
                eval_spinner.stop()
                error_source = self._categorize_error_source(e)
                click.echo(f"🚫 {error_source} Error during evaluation: {str(e)}")
                
                # Create evaluation with error info
                from .models import EvaluationResult
                evaluation = EvaluationResult(
                    overall_score=0.0,
                    dimension_scores={},
                    reasoning=f"Evaluation failed: {str(e)}",
                    confidence=0.0,
                    evaluation_time=0.0
                )
                break  # Exit evaluation retry loop even on error
        
        # Create interaction record (outside all retry loops)
        interaction = TestInteraction(
            timestamp=start_time,
            session_id=session_state.session_id,
            persona_name=persona.name,
            query=query.text,
            agent_response=agent_response.text,
            evaluation_score=evaluation.overall_score,
            evaluation_reasoning=evaluation.reasoning,
            execution_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "query_metadata": query.metadata,
                "agent_metadata": agent_response.metadata,
                "evaluation_metadata": {
                    "dimension_scores": evaluation.dimension_scores,
                    "confidence": evaluation.confidence,
                    "evaluation_time": evaluation.evaluation_time
                }
            }
        )
        
        # Log interaction immediately
        await self.session_logger.log_interaction(interaction)
        
        return interaction
    
    def _display_continue_approval(self, next_iteration: int, total_iterations: int) -> UserDecision:
        """Display approval prompt for continuing to next iteration."""
        if next_iteration <= total_iterations:
            click.echo(f"\n📋 Interaction completed. Next: Query {next_iteration} of {total_iterations}. Choose next action:")
            click.echo(f"   • (c)ontinue - Proceed to next query")
            click.echo(f"   • (r)etry - Regenerate current query and response")
            click.echo(f"   • (e)exit - Save progress and exit session")
        else:
            click.echo(f"\n📋 All {total_iterations} interactions completed! Choose next action:")
            click.echo(f"   • (c)ontinue - Complete session")
            click.echo(f"   • (r)etry - Regenerate last query and response")
            click.echo(f"   • (e)exit - Save progress and exit session")
        
        while True:
            choice = click.prompt(
                "\nChoose action",
                type=click.Choice(['c', 'r', 'e']),
                show_choices=True,
                show_default=False
            ).lower()
            
            if choice == 'c':
                return UserDecision.CONTINUE
            elif choice == 'r':
                return UserDecision.RETRY
            elif choice == 'e':
                return UserDecision.EXIT
    
    def _display_evaluation_approval(self) -> UserDecision:
        """Display approval prompt for evaluation step."""
        click.echo(f"\n📋 Response shown above. Choose next action:")
        click.echo(f"   • (c)ontinue - Proceed with evaluation")
        click.echo(f"   • (r)etry - Regenerate the agent response")
        click.echo(f"   • (e)exit - Save progress and exit session")
        
        while True:
            choice = click.prompt(
                "\nChoose action",
                type=click.Choice(['c', 'r', 'e']),
                show_choices=True,
                show_default=False
            ).lower()
            
            if choice == 'c':
                return UserDecision.CONTINUE
            elif choice == 'r':
                return UserDecision.RETRY
            elif choice == 'e':
                return UserDecision.EXIT
    
    def _display_query_approval(
        self,
        query: GeneratedQuery,
        persona: Persona,
        iteration: int
    ) -> UserDecision:
        """Display a query for user approval and get their decision."""
        click.echo("\n" + "="*80)
        click.echo(f"Query {iteration} | Persona: {persona.name}")
        click.echo("="*80)
        # Show generated query using standardized renderer with thinking content
        from .deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        renderer.display_query_generation(
            query.text, 
            query.metadata or {}, 
            thinking_content=query.thinking_content
        )
        
        while True:
            choice = click.prompt(
                "\nChoose action",
                type=click.Choice(['c', 'r', 'e']),
                show_choices=True,
                show_default=False
            ).lower()
            
            if choice == 'c':
                return UserDecision.CONTINUE
            elif choice == 'r':
                return UserDecision.RETRY
            elif choice == 'e':
                return UserDecision.EXIT
    
class ExecutionModeFactory:
    """Factory for creating execution mode handlers."""
    
    @staticmethod
    def create_handler(
        mode: ExecutionMode,
        query_generator: QueryGeneratorInterface,
        agent_executor: AgentExecutorInterface,
        evaluator: EvaluatorInterface,
        session_logger: SessionLoggerInterface
    ) -> ExecutionModeHandler:
        """Create appropriate execution mode handler."""
        if mode == ExecutionMode.AUTO:
            return AutoModeHandler(
                query_generator, agent_executor, evaluator, session_logger
            )
        elif mode == ExecutionMode.APPROVE:
            return ApproveModeHandler(
                query_generator, agent_executor, evaluator, session_logger
            )
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")