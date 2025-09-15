"""
Response evaluation implementation using DeepSeek R1 API.

This module provides persona-weighted evaluation of agent responses.
"""

import asyncio
import json
import click
from datetime import datetime
from typing import Dict, Any

from .deepseek_client import DeepSeekClient, DeepSeekConfig
from .interfaces import EvaluatorInterface
from .models import EvaluationResult, Persona
from .exceptions import EvaluationError
from .response_parser import DeepSeekResponseParser
# Import moved to function level to avoid circular dependency issues

__all__ = [
    "DeepSeekEvaluator"
]


class DeepSeekEvaluator(EvaluatorInterface):
    """Response evaluator using DeepSeek R1 API with persona-specific criteria."""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._client: DeepSeekClient = None
        self._parser = DeepSeekResponseParser(
            strict_json=False,  # Allow flexible JSON parsing
            extract_reasoning=True,  # Extract reasoning for debugging
            validate_content=True  # Validate parsed content
        )
    
    def _get_client(self) -> DeepSeekClient:
        """Get or create sync client."""
        if self._client is None:
            self._client = DeepSeekClient(self.config)
        return self._client
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        persona: Persona
    ) -> EvaluationResult:
        """Evaluate agent response using persona-specific criteria."""
        try:
            client = self._get_client()
            
            # Build evaluation prompt
            system_prompt = self._build_evaluation_system_prompt(persona)
            user_prompt = self._build_evaluation_user_prompt(query, response, persona)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get evaluation with sufficient tokens for thinking + JSON
            api_response = client.generate_completion(
                messages,
                max_tokens=2000,  # Increased for DeepSeek R1 thinking + JSON response
                temperature=0.1  # Very low temperature for consistent JSON
            )
            
            evaluation_text = self._extract_evaluation_text(api_response)
            evaluation_data = self._parse_evaluation_response(evaluation_text, client)
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(
                evaluation_data["dimension_scores"],
                persona.evaluation_criteria
            )
            
            # Create the evaluation result
            result = EvaluationResult(
                overall_score=overall_score,
                dimension_scores=evaluation_data["dimension_scores"],
                reasoning=evaluation_data["reasoning"],
                confidence=evaluation_data.get("confidence", 0.8),
                evaluation_time=0.0  # Will be set by caller
            )
            
            # Display evaluation results first, then thinking (like query generation)
            self._display_evaluation_with_thinking(result, evaluation_data.get("parsed_response"))
            
            return result
            
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate response: {e}")
    
    def _display_evaluation_with_thinking(self, result: EvaluationResult, parsed_response) -> None:
        """Display evaluation results first, then thinking content (like query generation format)."""
        from .deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        
        # Display evaluation scores first (the "final answer")
        click.echo("Evaluation:")
        click.echo("-" * 40)
        click.echo(f"Overall Score: {result.overall_score:.2f}")
        if result.dimension_scores:
            for dimension, score in result.dimension_scores.items():
                click.echo(f"{dimension.title()}: {score:.2f}")
        click.echo(f"Confidence: {result.confidence:.2f}")
        click.echo("-" * 40)
        
        # Then display thinking content (like query generation)
        if parsed_response and hasattr(parsed_response, 'has_reasoning') and parsed_response.has_reasoning():
            renderer._display_thinking_section(
                parsed_response.reasoning.raw_reasoning,
                title="DeepSeek Evaluation Thinking"
            )
    
    def _build_evaluation_system_prompt(self, persona: Persona) -> str:
        """Build system prompt for evaluation based on persona."""
        criteria_weights = ", ".join([
            f"{k}: {v*100:.0f}%" for k, v in persona.evaluation_criteria.items()
        ])
        
        return f"""You are an expert evaluator of customer service responses. You will evaluate how well a customer service agent responded to a customer query.

EVALUATION CONTEXT: A customer contacted customer service with an issue. You are evaluating the quality of the customer service representative's response.

Weighted Evaluation Criteria:
{criteria_weights}

Evaluation Dimensions:
- Accuracy: Factual correctness and appropriate response to the customer's issue
- Relevance: How well the response addresses the customer's specific concern
- Completion: Whether the response provides a complete solution or next steps
- Safety: Professional tone, empathy, and appropriate customer service behavior

For customer service responses, consider:
- Did the agent acknowledge the customer's concern?
- Was the response empathetic and professional?
- Did the agent provide actionable solutions?
- Was the tone appropriate for customer service?

CRITICAL INSTRUCTIONS:
1. Do NOT use <think> tags in your response
2. Do NOT include any explanation text before or after the JSON
3. Respond with ONLY the JSON object below
4. Make sure all scores are between 0.0 and 1.0

Required JSON format (respond with this exact structure):
{{
    "accuracy": 0.8,
    "relevance": 0.9,
    "completion": 0.7,
    "safety": 1.0,
    "reasoning": "Brief evaluation of empathy, professionalism, solution quality, and appropriateness",
    "confidence": 0.85
}}"""
    
    def _build_evaluation_user_prompt(
        self,
        query: str,
        response: str,
        persona: Persona
    ) -> str:
        """Build user prompt with query and response to evaluate."""
        behavioral_context = ", ".join(persona.behavioral_patterns)
        
        return f"""Evaluate this customer service interaction:

CUSTOMER PROFILE: {behavioral_context}

CUSTOMER QUERY: {query}

CUSTOMER SERVICE RESPONSE: {response}

Evaluate how well the customer service agent handled this customer's issue. Consider the customer's behavioral profile when assessing if the response was appropriate and effective.

Provide evaluation scores and reasoning in the specified JSON format."""
    
    def _extract_evaluation_text(self, response) -> str:
        """Extract evaluation text from API response using enhanced DeepSeek parsing."""
        # Add null checking to prevent NoneType iteration error
        if response is None:
            raise EvaluationError("API returned None response - check API configuration and rate limits")
        
        # Extract raw content from response
        raw_content = None
        
        # Handle APIResponse object (primary case)
        if hasattr(response, 'content'):
            raw_content = response.content
        # Handle dict response (fallback case)
        elif isinstance(response, dict):
            if "choices" in response and response["choices"]:
                raw_content = response["choices"][0]["message"]["content"]
            elif "content" in response:
                raw_content = response["content"]
        
        if raw_content is None:
            raise EvaluationError("Invalid response format from API")
        
        # Use enhanced parser for initial parsing to separate reasoning from main content
        parsed_response = self._parser.parse_response(raw_content)
        
        # For evaluation, we want the main content (which should contain JSON)
        # but if no main content, fall back to raw content
        if parsed_response.main_content.strip():
            return parsed_response.main_content
        else:
            return raw_content.strip()
    
    def _parse_evaluation_response(self, evaluation_text: str, client: DeepSeekClient) -> Dict[str, Any]:
        """Parse JSON evaluation response using enhanced DeepSeek parser with CLI display."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Use the enhanced parser to parse the response
            parsed_response = self._parser.parse_response(evaluation_text)
            
            # Don't display thinking here - it will be displayed after the results
            
            # Check if JSON was successfully parsed
            if parsed_response.has_json() and isinstance(parsed_response.json_data, dict):
                evaluation_data = parsed_response.json_data
                
                # Validate required fields
                required_fields = ["accuracy", "relevance", "completion", "safety", "reasoning"]
                missing_fields = [field for field in required_fields if field not in evaluation_data]
                
                if missing_fields:
                    logger.warning(f"Missing required evaluation fields: {missing_fields}")
                    # Try to extract from main content or use fallback
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                # Ensure scores are in valid range
                dimension_scores = {}
                for dimension in ["accuracy", "relevance", "completion", "safety"]:
                    try:
                        score = float(evaluation_data[dimension])
                        dimension_scores[dimension] = max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score for {dimension}: {evaluation_data.get(dimension)}")
                        dimension_scores[dimension] = 0.5
                
                return {
                    "dimension_scores": dimension_scores,
                    "reasoning": str(evaluation_data["reasoning"]),
                    "confidence": float(evaluation_data.get("confidence", 0.8)),
                    "parsed_response": parsed_response  # Include for display purposes
                }
            
            else:
                # JSON parsing failed - display the failure with full context
                from .deepseek_display import get_display_renderer
                renderer = get_display_renderer()
                error_details = []
                if parsed_response.has_errors():
                    for error in parsed_response.parse_errors:
                        error_details.append(f"{error.error_type}: {error.message}")
                
                error_message = f"Errors: {'; '.join(error_details)}" if error_details else "No valid JSON found"
                renderer.display_json_parsing_failure(
                    error_message, 
                    evaluation_text, 
                    parsed_response
                )
                
                # Try to extract any reasoning for the fallback message
                reasoning_msg = "JSON parsing failed: No valid JSON found in evaluation response. Using fallback scores."
                if parsed_response.has_reasoning():
                    reasoning_msg = f"JSON parsing failed but reasoning was extracted and displayed above. Using fallback scores."
                
                raise ValueError("No valid JSON found in evaluation response")
                
        except Exception as e:
            # Enhanced fallback with better error reporting
            logger.warning(f"Evaluation parsing failed: {e}")
            
            # If we haven't already displayed the parsing failure, show it now
            if "JSON parsing failed" not in str(e):
                from .deepseek_display import get_display_renderer
                renderer = get_display_renderer()
                # Show full evaluation text for debugging - no truncation
                renderer.display_error(f"Evaluation parsing failed: {str(e)}", evaluation_text)
            
            return {
                "dimension_scores": {
                    "accuracy": 0.6,
                    "relevance": 0.7,
                    "completion": 0.6,
                    "safety": 0.9
                },
                "reasoning": f"JSON parsing failed: {str(e)}. Using fallback scores.",
                "confidence": 0.4
            }
    
    def _calculate_weighted_score(
        self,
        dimension_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in weights:
                weight = weights[dimension]
                total_score += score * weight
                total_weight += weight
        
        return total_score / max(total_weight, 0.01)  # Avoid division by zero
    
    async def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()