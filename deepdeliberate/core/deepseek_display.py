"""
Standardized display renderer for DeepSeek R1 outputs in CLI.

This module provides a unified way to display DeepSeek R1 responses, thinking content,
evaluation results, and error messages with consistent formatting across the CLI.
"""

import click
from typing import Optional, Dict, Any
from .response_parser import ParsedResponse
from .models import EvaluationResult


class DeepSeekDisplayRenderer:
    """Standardized renderer for all DeepSeek R1 outputs in the CLI."""
    
    def __init__(self, show_thinking: bool = True, max_content_length: Optional[int] = None):
        """
        Initialize the display renderer.
        
        Args:
            show_thinking: Whether to display thinking content from <think> tags
            max_content_length: Maximum length for content before truncation (None = no limit)
        """
        self.show_thinking = show_thinking
        self.max_content_length = max_content_length
    
    def display_query_generation(self, query: str, metadata: Dict[str, Any], 
                                thinking_content: Optional[str] = None,
                                parsed_response: Optional[ParsedResponse] = None) -> None:
        """Display generated query with metadata and optional thinking."""
        click.echo("Generated Query:")
        click.echo("-" * 40)
        click.echo(query)
        click.echo("-" * 40)
        
        # Show thinking content if available (direct content takes precedence)
        if self.show_thinking:
            if thinking_content and thinking_content.strip():
                self._display_thinking_section(thinking_content, "DeepSeek Query Generation Thinking")
            elif parsed_response and parsed_response.has_reasoning():
                self._display_thinking_section(parsed_response.reasoning.raw_reasoning, "DeepSeek Query Generation Thinking")
        
        if metadata:
            metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
            click.echo(f"Metadata: {{{metadata_str}}}")
    
    def display_agent_response(self, response: str, title: str = "Agent Response") -> None:
        """Display agent response with consistent formatting."""
        click.echo(f"\n🤖 {title}:")
        click.echo("=" * 60)
        
        display_text = self._apply_content_truncation(response)
        click.echo(display_text)
        click.echo("=" * 60)
    
    def display_evaluation_result(self, result: EvaluationResult, 
                                 parsed_response: Optional[ParsedResponse] = None,
                                 raw_response: Optional[str] = None) -> None:
        """Display evaluation results with thinking content when available."""
        click.echo(f"\n📊 Evaluation Results:")
        
        # Show thinking content if available and enabled
        if parsed_response and self.show_thinking and parsed_response.has_reasoning():
            self._display_thinking_section(parsed_response.reasoning.raw_reasoning, 
                                         title="Evaluation Thinking")
        
        # Show evaluation scores and reasoning
        click.echo(f"   Overall Score: {result.overall_score:.2f}")
        click.echo(f"   Confidence: {result.confidence:.2f}")
        
        if result.dimension_scores:
            click.echo("   Dimension Scores:")
            for dimension, score in result.dimension_scores.items():
                click.echo(f"     • {dimension.title()}: {score:.2f}")
        
        if result.reasoning:
            click.echo("   Reasoning:")
            reasoning_display = self._apply_content_truncation(result.reasoning, max_length=300)
            click.echo(f"     {reasoning_display}")
        
        # If JSON parsing failed but we have raw response, show it in full
        if (result.reasoning and "JSON parsing failed" in result.reasoning and 
            raw_response and self.show_thinking):
            click.echo("\n   📋 Full Raw Evaluation Response (No Truncation):")
            click.echo("   " + "-" * 50)
            click.echo(f"     {raw_response}")
            click.echo("   " + "-" * 50)
    
    def display_json_parsing_failure(self, error_message: str, raw_content: str, 
                                   parsed_response: Optional[ParsedResponse] = None) -> None:
        """Display JSON parsing failure with full context."""
        click.echo(f"\n⚠️  JSON Parsing Failed: {error_message}")
        
        # Show thinking content if available
        if parsed_response and self.show_thinking and parsed_response.has_reasoning():
            self._display_thinking_section(parsed_response.reasoning.raw_reasoning,
                                         title="DeepSeek Thinking (Despite Parsing Failure)")
        
        # Show FULL raw content for debugging - no truncation for evaluation failures
        click.echo("📋 Full Raw Response Content (No Truncation):")
        click.echo("-" * 60)
        click.echo(raw_content)
        click.echo("-" * 60)
    
    def display_interaction_summary(self, query: str, agent_response: str, 
                                  evaluation_score: float, evaluation_reasoning: str,
                                  execution_time: float) -> None:
        """Display complete interaction summary without duplicating agent response."""
        click.echo(f"\n📊 Interaction Results:")
        
        # Query - show full query for reference
        click.echo(f"   Query: {query}")
        
        # Skip agent response display here since it was already shown above
        # Just show the evaluation summary
        click.echo(f"   Evaluation Score: {evaluation_score:.2f}")
        click.echo(f"   Execution Time: {execution_time:.2f}s")
        
        if evaluation_reasoning:
            click.echo(f"   Evaluation Reasoning: {evaluation_reasoning}")
    
    def _display_thinking_section(self, thinking_content: str, title: str = "DeepSeek Thinking") -> None:
        """Display thinking content with consistent formatting."""
        if not thinking_content.strip():
            return
        
        click.echo(f"\n🧠 {title}:")
        click.echo("-" * 40)
        
        # Don't truncate thinking content - it's valuable for debugging
        thinking_lines = thinking_content.strip().split('\n')
        for line in thinking_lines:
            click.echo(f"   {line}")
        
        click.echo("-" * 40)
    
    def _apply_content_truncation(self, content: str, max_length: Optional[int] = None) -> str:
        """Apply content truncation with length information."""
        if not content:
            return ""
        
        truncate_at = max_length or self.max_content_length
        
        if truncate_at and len(content) > truncate_at:
            truncated = content[:truncate_at].rstrip()
            return f"{truncated}...\n   [Content truncated - full length: {len(content)} characters]"
        
        return content
    
    def display_error(self, error_message: str, context: Optional[str] = None) -> None:
        """Display error messages with consistent formatting."""
        click.echo(f"\n❌ Error: {error_message}")
        if context:
            click.echo(f"   Context: {context}")
    
    def display_session_summary(self, total_interactions: int, successful_interactions: int, 
                               average_score: Optional[float] = None) -> None:
        """Display session completion summary with consistent formatting."""
        click.echo(f"\n🎉 Session completed successfully!")
        click.echo(f"   Total interactions: {total_interactions}")
        click.echo(f"   Successful interactions: {successful_interactions}")
        if average_score is not None:
            click.echo(f"   Average evaluation score: {average_score:.2f}")


# Global renderer instance
_default_renderer = DeepSeekDisplayRenderer(show_thinking=True, max_content_length=None)


def get_display_renderer() -> DeepSeekDisplayRenderer:
    """Get the default display renderer instance."""
    return _default_renderer


def set_display_renderer(renderer: DeepSeekDisplayRenderer) -> None:
    """Set a custom display renderer instance."""
    global _default_renderer
    _default_renderer = renderer