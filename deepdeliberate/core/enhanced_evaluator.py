"""
Enhanced evaluation system with comprehensive LLM-as-a-Judge improvements.

This module provides a much more sophisticated evaluation of customer service responses
using advanced prompting techniques, richer context, and detailed analytical frameworks.
"""

import asyncio
import json
import click
from datetime import datetime
from typing import Dict, Any, List

from .deepseek_client import DeepSeekClient, DeepSeekConfig
from .interfaces import EvaluatorInterface
from .models import EvaluationResult, Persona
from .exceptions import EvaluationError
from .response_parser import DeepSeekResponseParser


class EnhancedDeepSeekEvaluator(EvaluatorInterface):
    """Enhanced evaluator with comprehensive LLM-as-a-Judge improvements."""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._client: DeepSeekClient = None
        self._parser = DeepSeekResponseParser(
            strict_json=False,
            extract_reasoning=True,
            validate_content=True
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
        persona: Persona,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:
        """Evaluate agent response using enhanced comprehensive criteria."""
        try:
            client = self._get_client()
            
            # Build enhanced evaluation prompts
            system_prompt = self._build_enhanced_system_prompt(persona, context or {})
            user_prompt = self._build_enhanced_user_prompt(query, response, persona, context or {})
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get evaluation with increased tokens for detailed reasoning
            api_response = client.generate_completion(
                messages,
                max_tokens=3000,  # Increased for comprehensive analysis
                temperature=0.1
            )
            
            evaluation_text = self._extract_evaluation_text(api_response)
            evaluation_data = self._parse_evaluation_response(evaluation_text, client)
            
            # Calculate weighted overall score with enhanced dimensions
            overall_score = self._calculate_enhanced_weighted_score(
                evaluation_data["dimension_scores"],
                persona.evaluation_criteria
            )
            
            result = EvaluationResult(
                overall_score=overall_score,
                dimension_scores=evaluation_data["dimension_scores"],
                reasoning=evaluation_data["reasoning"],
                confidence=evaluation_data.get("confidence", 0.8),
                evaluation_time=0.0
            )
            
            # Display enhanced evaluation results
            self._display_enhanced_evaluation(result, evaluation_data.get("parsed_response"))
            
            return result
            
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate response: {e}")
    
    def _build_enhanced_system_prompt(self, persona: Persona, context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt with advanced evaluation framework."""
        criteria_weights = ", ".join([
            f"{k}: {v*100:.0f}%" for k, v in persona.evaluation_criteria.items()
        ])
        
        return f"""You are an expert customer service evaluation specialist with deep expertise in:
- Customer psychology and emotional intelligence
- Service recovery and de-escalation techniques  
- Communication effectiveness across different customer personas
- Business impact of customer interactions

EVALUATION MISSION: Conduct a comprehensive analysis of how well a customer service agent handled a specific customer interaction, considering persona-specific needs, emotional context, and business outcomes.

=== EVALUATION FRAMEWORK ===

**Weighted Criteria:** {criteria_weights}

**Enhanced Evaluation Dimensions:**

1. **Emotional Intelligence & Empathy (25%)**
   - Acknowledgment of customer's emotional state and frustrations
   - Appropriate empathetic language and tone matching
   - Validation of customer's concerns and feelings
   - Emotional de-escalation techniques if needed

2. **Problem Resolution Effectiveness (30%)**
   - Accuracy and appropriateness of solution provided
   - Completeness of resolution steps
   - Feasibility and clarity of instructions
   - Proactive prevention of future similar issues

3. **Communication Excellence (20%)**
   - Clarity and organization of response structure
   - Appropriate technical level for customer's expertise
   - Professional yet personable tone
   - Clear next steps and expectations

4. **Customer-Centric Approach (15%)**
   - Personalization based on customer's specific situation
   - Going beyond minimum requirements (proactive assistance)
   - Consideration of customer's time and effort
   - Demonstration of ownership and accountability

5. **Business Impact & Brand Representation (10%)**
   - Likelihood of customer satisfaction and retention
   - Prevention of escalation or negative reviews
   - Reinforcement of brand values and trust
   - Professional representation of company standards

=== PERSONA-SPECIFIC EXCELLENCE CRITERIA ===

For FRUSTRATED customers, excellent responses should:
- Lead with immediate acknowledgment of frustration and apology
- Provide clear, quick resolution paths
- Show urgency and personal ownership
- Include escalation options or priority handling
- Use confident, decisive language

For TECHNICAL customers, excellent responses should:
- Include detailed technical explanations when appropriate
- Provide multiple solution approaches
- Reference documentation or advanced options
- Acknowledge their technical competence

For BASIC-LEVEL customers, excellent responses should:
- Use simple, jargon-free language
- Provide step-by-step instructions with clear guidance
- Include reassurance and support availability
- Check understanding and offer additional help

=== ANALYSIS STRUCTURE ===

Analyze the response using this structured approach:

1. **Context Assessment**: Understanding of customer's situation and needs
2. **Response Quality**: How well each dimension was addressed
3. **Persona Alignment**: Appropriateness for this specific customer type
4. **Business Impact**: Likely outcomes for customer satisfaction and retention
5. **Improvement Opportunities**: Specific, actionable enhancement suggestions

=== CRITICAL INSTRUCTIONS ===

1. Use <think> tags for your detailed reasoning process
2. Quote specific parts of the response in your analysis
3. Compare against best practices for this persona type
4. Provide specific, actionable improvement suggestions
5. Consider both immediate resolution and long-term customer relationship

Required JSON format (respond with this exact structure):
{{
    "emotional_intelligence": 0.85,
    "problem_resolution": 0.90,
    "communication_excellence": 0.88,
    "customer_centric": 0.82,
    "business_impact": 0.87,
    "reasoning": "Detailed analysis with quotes and specific observations",
    "improvement_suggestions": "Specific actionable recommendations for enhancement",
    "persona_alignment_score": 0.90,
    "confidence": 0.85
}}"""
    
    def _build_enhanced_user_prompt(
        self,
        query: str,
        response: str,
        persona: Persona,
        context: Dict[str, Any]
    ) -> str:
        """Build enhanced user prompt with comprehensive context."""
        
        # Extract rich persona context
        persona_details = {
            "name": persona.name,
            "behavioral_patterns": persona.behavioral_patterns,
            "communication_style": getattr(persona, 'communication_style', 'Unknown'),
            "technical_level": getattr(persona, 'technical_level', 'Unknown'),
            "patience_level": getattr(persona, 'patience_level', 'Unknown'),
            "emotional_state": getattr(persona, 'emotional_state', 'Unknown')
        }
        
        # Business context
        interaction_context = {
            "issue_complexity": context.get("complexity", "Medium"),
            "customer_tier": context.get("customer_tier", "Standard"),
            "previous_interactions": context.get("interaction_count", 0),
            "urgency": context.get("urgency", "Normal"),
            "resolution_timeframe": context.get("expected_resolution", "Standard")
        }
        
        return f"""COMPREHENSIVE CUSTOMER SERVICE INTERACTION EVALUATION

=== CUSTOMER PROFILE ===
• Persona: {persona_details['name']}
• Behavioral Patterns: {', '.join(persona_details['behavioral_patterns'])}
• Communication Style: {persona_details['communication_style']}
• Technical Level: {persona_details['technical_level']}
• Patience Level: {persona_details['patience_level']}
• Current Emotional State: {persona_details['emotional_state']}

=== INTERACTION CONTEXT ===
• Issue Complexity: {interaction_context['issue_complexity']}
• Customer Tier: {interaction_context['customer_tier']}
• Previous Contact Attempts: {interaction_context['previous_interactions']}
• Urgency Level: {interaction_context['urgency']}
• Expected Resolution Time: {interaction_context['resolution_timeframe']}

=== CUSTOMER QUERY ===
{query}

=== AGENT RESPONSE TO EVALUATE ===
{response}

=== EVALUATION TASK ===

Conduct a comprehensive evaluation of this customer service interaction. Consider:

1. How well did the agent understand and address this specific customer's needs?
2. Was the response appropriately tailored to this persona's characteristics?
3. What was the quality of problem resolution provided?
4. How effectively did the agent manage the customer's emotional state?
5. What business outcomes is this response likely to achieve?
6. What specific improvements would make this response more effective?

Provide your evaluation using the structured JSON format with detailed reasoning and specific improvement suggestions."""
    
    def _extract_evaluation_text(self, response) -> str:
        """Extract evaluation text from API response."""
        if response is None:
            raise EvaluationError("API returned None response")
        
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict):
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            elif "content" in response:
                return response["content"]
        
        raise EvaluationError("Invalid response format from API")
    
    def _parse_evaluation_response(self, evaluation_text: str, client: DeepSeekClient) -> Dict[str, Any]:
        """Parse enhanced evaluation response with detailed analysis."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            parsed_response = self._parser.parse_response(evaluation_text)
            
            if parsed_response.has_json() and isinstance(parsed_response.json_data, dict):
                evaluation_data = parsed_response.json_data
                
                # Validate enhanced fields
                required_fields = [
                    "emotional_intelligence", "problem_resolution", "communication_excellence",
                    "customer_centric", "business_impact", "reasoning"
                ]
                missing_fields = [field for field in required_fields if field not in evaluation_data]
                
                if missing_fields:
                    logger.warning(f"Missing enhanced evaluation fields: {missing_fields}")
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                # Map enhanced dimensions to scores
                dimension_scores = {}
                for dimension in required_fields[:5]:  # Exclude reasoning
                    try:
                        score = float(evaluation_data[dimension])
                        dimension_scores[dimension] = max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score for {dimension}: {evaluation_data.get(dimension)}")
                        dimension_scores[dimension] = 0.5
                
                return {
                    "dimension_scores": dimension_scores,
                    "reasoning": str(evaluation_data["reasoning"]),
                    "improvement_suggestions": evaluation_data.get("improvement_suggestions", ""),
                    "persona_alignment": evaluation_data.get("persona_alignment_score", 0.8),
                    "confidence": float(evaluation_data.get("confidence", 0.8)),
                    "parsed_response": parsed_response
                }
            
            else:
                # Enhanced fallback with thinking content
                from .deepseek_display import get_display_renderer
                renderer = get_display_renderer()
                
                error_details = []
                if parsed_response.has_errors():
                    for error in parsed_response.parse_errors:
                        error_details.append(f"{error.error_type}: {error.message}")
                
                error_message = f"Errors: {'; '.join(error_details)}" if error_details else "No valid JSON found"
                renderer.display_json_parsing_failure(error_message, evaluation_text, parsed_response)
                
                # Enhanced fallback scores
                return {
                    "dimension_scores": {
                        "emotional_intelligence": 0.6,
                        "problem_resolution": 0.7,
                        "communication_excellence": 0.6,
                        "customer_centric": 0.6,
                        "business_impact": 0.7
                    },
                    "reasoning": f"Enhanced JSON parsing failed: {error_message}. Using fallback scores.",
                    "improvement_suggestions": "Unable to provide suggestions due to parsing failure.",
                    "persona_alignment": 0.6,
                    "confidence": 0.4
                }
                
        except Exception as e:
            logger.warning(f"Enhanced evaluation parsing failed: {e}")
            
            return {
                "dimension_scores": {
                    "emotional_intelligence": 0.6,
                    "problem_resolution": 0.7,
                    "communication_excellence": 0.6,
                    "customer_centric": 0.6,
                    "business_impact": 0.7
                },
                "reasoning": f"Enhanced evaluation failed: {str(e)}. Using fallback scores.",
                "improvement_suggestions": "Unable to provide suggestions due to evaluation failure.",
                "persona_alignment": 0.6,
                "confidence": 0.4
            }
    
    def _calculate_enhanced_weighted_score(
        self,
        dimension_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score using enhanced dimensions."""
        # Map enhanced dimensions to original weights if needed
        enhanced_weights = {
            "emotional_intelligence": weights.get("safety", 0.25),
            "problem_resolution": weights.get("completion", 0.30),
            "communication_excellence": weights.get("relevance", 0.20),
            "customer_centric": weights.get("accuracy", 0.15),
            "business_impact": 0.10  # New dimension weight
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in enhanced_weights:
                weight = enhanced_weights[dimension]
                total_score += score * weight
                total_weight += weight
        
        return total_score / max(total_weight, 0.01)
    
    def _display_enhanced_evaluation(self, result: EvaluationResult, parsed_response) -> None:
        """Display enhanced evaluation results with detailed breakdown."""
        from .deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        
        # Display enhanced evaluation scores
        click.echo("Enhanced Evaluation:")
        click.echo("-" * 50)
        click.echo(f"Overall Score: {result.overall_score:.2f}")
        
        if result.dimension_scores:
            click.echo("\nDetailed Breakdown:")
            dimension_names = {
                "emotional_intelligence": "Emotional Intelligence",
                "problem_resolution": "Problem Resolution",
                "communication_excellence": "Communication Excellence", 
                "customer_centric": "Customer-Centric Approach",
                "business_impact": "Business Impact"
            }
            
            for dimension, score in result.dimension_scores.items():
                display_name = dimension_names.get(dimension, dimension.replace("_", " ").title())
                click.echo(f"  • {display_name}: {score:.2f}")
        
        click.echo(f"\nConfidence: {result.confidence:.2f}")
        click.echo("-" * 50)
        
        # Display thinking content if available
        if parsed_response and hasattr(parsed_response, 'has_reasoning') and parsed_response.has_reasoning():
            renderer._display_thinking_section(
                parsed_response.reasoning.raw_reasoning,
                title="Enhanced DeepSeek Evaluation Analysis"
            )
    
    async def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()