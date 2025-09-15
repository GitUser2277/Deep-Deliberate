#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for DeepDeliberate framework.

This script provides the command-line interface for synthetic data generation
with automated query generation and evaluation.
"""

import asyncio
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Framework components
from deepdeliberate.config.manager import load_config
from deepdeliberate.core.agent_executor_interface import FrameworkAgentExecutor
from deepdeliberate.core.deepseek_client import DeepSeekConfig
from deepdeliberate.core.evaluator import DeepSeekEvaluator
from deepdeliberate.core.enhanced_evaluator import EnhancedDeepSeekEvaluator
from deepdeliberate.core.exceptions import FrameworkError
from deepdeliberate.core.framework_core import FrameworkCore
from deepdeliberate.core.models import CLIConfig, ExecutionMode
from deepdeliberate.core.query_generator import DeepSeekQueryGenerator
from deepdeliberate.core.session_logger import SessionLogger


def _setup_environment() -> None:
    """Configure console encoding (Windows) and load environment variables.

    Ensures `.env` is loaded before runtime checks and user-facing output.
    """
    # Fix Windows console encoding issues
    if sys.platform.startswith("win"):
        try:
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
        except Exception:
            # Fallback to default encoding
            pass

    # Load environment variables from project `.env` if present
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
    else:
        print(f"Environment file not found at: {env_path}")
        print("   Create a .env file with your API keys and configuration")


def display_model_configuration(enhanced_eval: bool = False):
    """Display the model configuration for transparency."""
    click.echo("\n📋 Model Configuration:")
    click.echo(
        "   • Query Generation: DeepSeek R1 "
        "(for diverse query generation)"
    )
    click.echo(
        "   • Agent Execution: Azure OpenAI o4-mini "
        "(your configured agent model)"
    )
    if enhanced_eval:
        click.echo(
            "   • Response Evaluation: DeepSeek R1 Enhanced "
            "(5-dimension comprehensive analysis)"
        )
    else:
        click.echo(
            "   • Response Evaluation: DeepSeek R1 Standard "
            "(4-dimension evaluation)"
        )

    # Show environment variable status
    click.echo("\n🔑 Environment Configuration:")
    deepseek_key = "✅ Set" if os.getenv("DEEPSEEK_API_KEY") else "❌ Missing"
    azure_key = "✅ Set" if os.getenv("AZURE_OPENAI_API_KEY") else "❌ Missing"
    azure_endpoint = (
        "✅ Set" if os.getenv("AZURE_OPENAI_ENDPOINT") else "❌ Missing"
    )
    azure_api_version = (
        "✅ Set" if os.getenv("AZURE_OPENAI_API_VERSION") else "❌ Missing"
    )
    azure_deployment = (
        "✅ Set" if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") else "❌ Missing"
    )

    click.echo(f"   • DEEPSEEK_API_KEY: {deepseek_key}")
    click.echo(f"   • AZURE_OPENAI_API_KEY: {azure_key}")
    click.echo(f"   • AZURE_OPENAI_ENDPOINT: {azure_endpoint}")
    click.echo(f"   • AZURE_OPENAI_API_VERSION: {azure_api_version}")
    click.echo(f"   • AZURE_OPENAI_DEPLOYMENT_NAME: {azure_deployment}")

    # Show rate limiting configuration
    rate_limit = os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")
    click.echo(f"   • API_RATE_LIMIT_PER_MINUTE: {rate_limit}")

    # Provide rate limiting guidance
    try:
        rate_limit_int = int(rate_limit.split("#")[0].strip())
        if rate_limit_int > 20:
            click.echo(
                "   ⚠️  Tip: If you hit rate limits, try reducing "
                "API_RATE_LIMIT_PER_MINUTE to 10-15"
            )
    except (ValueError, AttributeError):
        pass


@click.command()
@click.option(
    "-file", "--file", "agent_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to Python file containing PydanticAI agents"
)
@click.option(
    "-mode", "--mode",
    type=click.Choice([e.value for e in ExecutionMode], case_sensitive=False),
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
    help=(
        "Configuration file path (default: config.json, or use "
        "setup/configs/customer_service_config.json)"
    )
)
@click.option(
    "-enhanced", "--enhanced",
    is_flag=True,
    default=False,
    help="Use enhanced 5-dimension evaluator for comprehensive analysis"
)
def main(
    agent_file: Path,
    mode: ExecutionMode,
    count: int,
    persona: str,
    output: Path,
    config: Path,
    enhanced: bool
):
    """
    DeepDeliberate - AI Agent Testing Framework

    Test your PydanticAI agents with automated query generation and
    evaluation.

    Examples:
        python deepdeliberate.py -file setup/agents/customer_service_agent.py
            -mode auto -count 25
        python deepdeliberate.py -file setup/agents/customer_service_agent.py
            -mode approve -persona angry_customer
            -config setup/configs/customer_service_config.json
        python deepdeliberate.py -file setup/agents/education_agent.py
            -mode auto -count 10 --enhanced
            -config setup/configs/education_config.json
    """
    # Prepare environment for user-facing output and runtime
    _setup_environment()

    click.echo("DeepDeliberate Framework v1.0")
    click.echo("AI Agent Testing with Automated Query Generation")

    # Display configuration
    click.echo("\nConfiguration:")
    click.echo(f"  Agent File: {agent_file}")
    click.echo(f"  Mode: {mode}")
    click.echo(f"  Count: {count}")
    click.echo(f"  Persona: {persona}")
    click.echo(f"  Config: {config}")
    click.echo(f"  Enhanced Evaluator: {'Yes' if enhanced else 'No'}")

    # Display model configuration
    display_model_configuration(enhanced)

    # Create CLI configuration
    cli_config = CLIConfig(
        agent_file=str(agent_file),
        mode=mode,
        count=count,
        persona=persona,
        output_dir=str(output) if output else None,
        config_file=str(config),
        enhanced_evaluator=enhanced
    )

    # Run framework
    try:
        asyncio.run(run_framework_async(cli_config, enhanced))
    except KeyboardInterrupt:
        click.echo("\n⚠️  Framework interrupted by user")
    except Exception as e:
        click.echo(f"\n❌ Framework error: {e}")
        raise


async def run_framework_async(cli_config: CLIConfig, enhanced: bool):
    """Run the framework with proper async handling."""
    try:
        click.echo("\n🚀 Starting DeepDeliberate Framework")
        click.echo("=" * 50)

        # Load configuration
        click.echo("⚙️  Loading configuration...")
        config_path = Path(cli_config.config_file)

        # Normalize path for OS compatibility
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path

        if not config_path.exists():
            click.echo(f"❌ Configuration file not found: {config_path}")
            click.echo("   Available config files:")
            click.echo("   • config.json (root directory)")
            click.echo("   • setup/configs/customer_service_config.json")
            click.echo("   • setup/configs/education_config.json")
            click.echo("   • setup/configs/healthcare_config.json")
            raise FrameworkError(
                f"Configuration file not found: {config_path}"
            )

        click.echo(f"   Loading from: {config_path}")
        config = load_config(str(config_path))

        # Create DeepSeek configuration
        deepseek_config = DeepSeekConfig.from_environment()

        # Create components
        click.echo("📦 Creating framework components...")
        query_generator = DeepSeekQueryGenerator(deepseek_config)
        agent_executor = FrameworkAgentExecutor()

        # Choose evaluator based on enhanced flag
        if enhanced:
            click.echo(
                "   • Creating Enhanced DeepSeek Evaluator "
                "(5-dimension analysis)..."
            )
            evaluator = EnhancedDeepSeekEvaluator(deepseek_config)
        else:
            click.echo(
                "   • Creating Standard DeepSeek Evaluator "
                "(4-dimension analysis)..."
            )
            evaluator = DeepSeekEvaluator(deepseek_config)

        session_logger = SessionLogger(
            config.logging_settings.output_directory
        )

        # Create framework
        click.echo("🔧 Initializing framework...")
        framework = FrameworkCore(
            query_generator=query_generator,
            agent_executor=agent_executor,
            evaluator=evaluator,
            session_logger=session_logger
        )

        # Initialize framework and agent executor
        await framework.initialize(config)
        await framework.initialize_agent_executor(cli_config.agent_file)

        click.echo("✅ Framework initialized successfully")
        click.echo("🤖 Discovering and loading agents...")

        # Start testing session
        click.echo("🎯 Starting new testing session...")
        result = await framework.run_testing_session(cli_config)

        # Display results using standardized renderer
        successful = (
            [
                i
                for i in result.completed_interactions
                if not i.agent_response.startswith("ERROR")
            ]
            if result.completed_interactions
            else []
        )
        avg_score = (
            sum(i.evaluation_score for i in successful) / len(successful)
            if successful
            else None
        )

        from deepdeliberate.core.deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        renderer.display_session_summary(
            total_interactions=len(result.completed_interactions),
            successful_interactions=len(successful),
            average_score=avg_score
        )

    except FrameworkError as e:
        click.echo(f"❌ Framework error: {e}")
        raise
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
