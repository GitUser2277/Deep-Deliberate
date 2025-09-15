"""
Command-line interface for DeepDeliberate framework.

This module provides the main CLI entry point for the framework,
handling argument parsing and user interaction.
"""

import click
from pathlib import Path
from typing import Optional

from .core.models import CLIConfig, ExecutionMode, UserDecision


@click.command()
@click.option(
    "-file", "--file", "agent_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to Python file containing PydanticAI agents"
)
@click.option(
    "-mode", "--mode",
    type=click.Choice([ExecutionMode.AUTO, ExecutionMode.APPROVE], case_sensitive=False),
    required=True,
    help="Execution mode: auto (fully automated) or approve (step-by-step)"
)
@click.option(
    "-count", "--count",
    default=10,
    type=int,
    help="Number of test queries to generate (default: 10)"
)
@click.option(
    "-persona", "--persona",
    type=str,
    help="Specific persona to use for testing (optional)"
)
@click.option(
    "-output", "--output",
    type=click.Path(path_type=Path),
    help="Output directory for test results (optional)"
)
@click.option(
    "-config", "--config",
    default="config.json",
    type=click.Path(path_type=Path),
    help="Configuration file path (default: config.json)"
)
def main(
    agent_file: Path,
    mode: ExecutionMode,
    count: int,
    persona: Optional[str],
    output: Optional[Path],
    config: Path
) -> None:
    """
    DeepDeliberate - AI Agent Testing Framework
    
    Test your PydanticAI agents with automated query generation and evaluation.
    
    Examples:
        deepdeliberate -file src/agent.py -mode auto -count 25
        deepdeliberate -file src/agent.py -mode approve -persona angry_customer
    """
    # Create CLI configuration
    cli_config = CLIConfig(
        agent_file=str(agent_file),
        mode=mode,
        count=count,
        persona=persona,
        output_dir=str(output) if output else None,
        config_file=str(config)
    )
    
    click.echo(">> Starting DeepDeliberate framework...")
    click.echo(f"   Agent file: {cli_config.agent_file}")
    click.echo(f"   Mode: {cli_config.mode}")
    click.echo(f"   Count: {cli_config.count}")
    
    if cli_config.persona:
        click.echo(f"   Persona: {cli_config.persona}")
    
    if cli_config.output_dir:
        click.echo(f"   Output: {cli_config.output_dir}")
    
    # Import and run framework core with proper async handling
    try:
        import asyncio
        from .core.framework_core import FrameworkCore
        
        # Run the async framework
        asyncio.run(run_framework_async(cli_config))
        
    except ImportError as e:
        click.echo(f"   Framework core not yet fully integrated: {e}")
        click.echo("   This will be completed when all components are implemented")


async def run_framework_async(cli_config: CLIConfig) -> None:
    """Run the framework with proper async handling."""
    from .core.framework_core import FrameworkCore
    from .core.query_generator import DeepSeekQueryGenerator
    from .core.evaluator import DeepSeekEvaluator
    from .core.agent_executor_interface import FrameworkAgentExecutor
    from .core.session_logger import SessionLogger
    from .config.manager import load_config
    
    # Load configuration
    config = load_config(cli_config.config_file)
    
    # Create framework components
    query_generator = DeepSeekQueryGenerator()
    agent_executor = FrameworkAgentExecutor()
    evaluator = DeepSeekEvaluator()
    session_logger = SessionLogger()
    
    # Create and run framework
    async with FrameworkCore(
        query_generator=query_generator,
        agent_executor=agent_executor,
        evaluator=evaluator,
        session_logger=session_logger
    ) as framework:
        await framework.initialize(config)
        result = await framework.run_testing_session(cli_config)
        click.echo(f"✅ Session completed with {len(result.completed_interactions)} interactions")


def display_query_approval(query: str, persona: str) -> UserDecision:
    """Display a query for user approval and get their decision."""
    click.echo("\n" + "="*60)
    click.echo(f"Persona: {persona}")
    click.echo("Generated Query:")
    click.echo("-" * 40)
    click.echo(query)
    click.echo("-" * 40)
    
    while True:
        choice = click.prompt(
            "Choose action",
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


if __name__ == "__main__":
    main()