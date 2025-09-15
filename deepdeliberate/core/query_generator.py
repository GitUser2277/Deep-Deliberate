"""
Query generation implementation using DeepSeek R1 API.

This module provides persona-driven query generation for testing AI agents
with enhanced capabilities including template parameter injection, edge case
generation, and context-aware multi-turn conversation support.
"""

import asyncio
import random
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from .deepseek_client import DeepSeekClient, DeepSeekConfig
from .interfaces import QueryGeneratorInterface
from .models import GeneratedQuery, Persona, TestInteraction
from .exceptions import QueryGenerationError

__all__ = [
    "DeepSeekQueryGenerator"
]


class DeepSeekQueryGenerator(QueryGeneratorInterface):
    """
    Enhanced query generator using DeepSeek R1 API with persona-driven prompts.
    
    Features:
    - Template parameter injection for dynamic scenario generation
    - Edge case and adversarial query generation
    - Context-aware multi-turn conversation support
    """
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._client: DeepSeekClient = None
        
        # Edge case generation patterns
        self._edge_case_patterns = [
            "boundary_conditions",
            "adversarial_inputs", 
            "ambiguous_requests",
            "multi_part_queries",
            "context_switching",
            "error_scenarios"
        ]
    
    def _get_client(self) -> DeepSeekClient:
        """Get or create sync client."""
        if self._client is None:
            self._client = DeepSeekClient(self.config)
        return self._client
    
    async def generate_query(
        self,
        persona: Persona,
        context: Dict[str, Any]
    ) -> GeneratedQuery:
        """
        Generate a query based on persona and context with enhanced capabilities.
        Includes graceful fallback if DeepSeek API is unavailable or rate-limited.
        """
        try:
            client = self._get_client()
            
            # Determine query generation strategy
            generation_strategy = self._determine_generation_strategy(context)
            
            # Build enhanced persona-driven prompt
            system_prompt = self._build_enhanced_system_prompt(persona, generation_strategy)
            user_prompt = self._build_enhanced_user_prompt(persona, context, generation_strategy)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add conversation history for context-aware generation
            if context.get("conversation_history"):
                messages = self._inject_conversation_context(messages, context["conversation_history"])
            
            # Generate query with strategy-specific parameters
            temperature = self._get_temperature_for_strategy(generation_strategy)
            response = client.generate_completion(
                messages,
                max_tokens=500,
                temperature=temperature
            )
            
            # Null-safety check (can happen if provider returns None)
            if response is None or response.content is None:
                raise QueryGenerationError("DeepSeek API returned no content – possible rate limit or outage")
            
            query_text, thinking_content = self._extract_query_and_thinking(response)
            
            # Post-process query based on strategy
            query_text = self._post_process_query(query_text, persona, generation_strategy)
            
            metadata = {
                "generation_strategy": generation_strategy,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "temperature_used": temperature,
                "response_tokens": len(query_text.split()),
                "has_thinking": thinking_content is not None
            }
            
            return GeneratedQuery(
                text=query_text,
                persona_name=persona.name,
                context=context,
                generation_time=datetime.now(),
                metadata=metadata,
                thinking_content=thinking_content
            )
        
        except Exception as e:
            # If DeepSeek fails (rate limit, outage, etc.), fallback to simple generation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DeepSeek query generation failed – falling back to simple generator: {e}")
            fallback_query = self._generate_fallback_query(persona, context)
            return fallback_query
    
    def _generate_fallback_query(self, persona: Persona, context: Dict[str, Any]) -> GeneratedQuery:
        """Generate a basic fallback query without external API calls."""
        import random
        template = random.choice(persona.scenario_templates) if persona.scenario_templates else "{name} has an issue with your product."
        
        # Basic placeholder replacements using available context
        replacements = {
            "name": random.choice(["Customer", "User", "Client"]),
            "issue": random.choice(["login problem", "app crash", "payment failure"]),
        }
        
        # Replace known template parameters
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", value)
        
        metadata = {
            "generation_strategy": "fallback",
            "reason": "DeepSeek unavailable – local fallback",
        }
        
        return GeneratedQuery(
            text=template,
            persona_name=persona.name,
            context=context,
            generation_time=datetime.now(),
            metadata=metadata
        )
    
    def _determine_generation_strategy(self, context: Dict[str, Any]) -> str:
        """Determine the appropriate query generation strategy based on context."""
        iteration = context.get("iteration", 1)
        total = context.get("total", 1)
        force_edge_case = context.get("force_edge_case", False)
        conversation_history = context.get("conversation_history", [])
        
        # Force edge case if explicitly requested
        if force_edge_case:
            return random.choice(self._edge_case_patterns)
        
        # Multi-turn conversation context
        if conversation_history and len(conversation_history) > 0:
            return "context_aware"
        
        # Generate edge cases for later iterations (20% chance after iteration 3)
        if iteration > 3 and random.random() < 0.2:
            return random.choice(self._edge_case_patterns)
        
        # Default to standard generation
        return "standard"
    
    def _build_enhanced_system_prompt(self, persona: Persona, strategy: str) -> str:
        """Build enhanced system prompt based on persona characteristics and strategy."""
        behavioral_patterns = ", ".join(persona.behavioral_patterns)
        tone_specs = ", ".join([f"{k}: {v}" for k, v in persona.tone_specifications.items()])
        
        base_prompt = f"""You are a query generator for testing AI agents. Generate realistic queries that a user with these characteristics would ask:

Behavioral Patterns: {behavioral_patterns}
Communication Style: {tone_specs}"""

        strategy_instructions = self._get_strategy_instructions(strategy)
        
        return f"""{base_prompt}

{strategy_instructions}

Generate queries that are:
1. Natural and realistic for this persona
2. Appropriate for testing an AI agent
3. Varied in complexity and topic
4. Reflective of the persona's communication style

Return only the query text, no additional formatting or explanation."""

    def _get_strategy_instructions(self, strategy: str) -> str:
        """Get specific instructions for each generation strategy."""
        strategy_map = {
            "standard": "Generate typical, straightforward queries that this persona would naturally ask.",
            
            "boundary_conditions": """Generate queries that test boundary conditions:
- Extremely long or short inputs
- Requests at the limits of the agent's capabilities
- Edge cases in data ranges or parameters""",
            
            "adversarial_inputs": """Generate adversarial queries designed to test robustness:
- Attempts to confuse or mislead the agent
- Contradictory or paradoxical requests
- Queries designed to expose weaknesses""",
            
            "ambiguous_requests": """Generate intentionally ambiguous queries:
- Vague or unclear requests
- Multiple possible interpretations
- Missing critical context or details""",
            
            "multi_part_queries": """Generate complex multi-part queries:
- Multiple questions in one request
- Nested or dependent sub-questions
- Queries requiring multiple steps to answer""",
            
            "context_switching": """Generate queries that switch context or topic:
- Sudden topic changes mid-conversation
- References to previous unrelated topics
- Mixing different domains or subjects""",
            
            "error_scenarios": """Generate queries likely to cause errors:
- Invalid parameters or formats
- Impossible or contradictory requests
- Queries outside the agent's domain""",
            
            "context_aware": """Generate queries that build on previous conversation:
- Reference previous responses or topics
- Ask follow-up questions
- Build on established context"""
        }
        
        return strategy_map.get(strategy, strategy_map["standard"])

    def _build_system_prompt(self, persona: Persona) -> str:
        """Build system prompt based on persona characteristics (legacy method)."""
        return self._build_enhanced_system_prompt(persona, "standard")
    
    def _build_enhanced_user_prompt(self, persona: Persona, context: Dict[str, Any], strategy: str) -> str:
        """Build enhanced user prompt with template parameter injection and context awareness."""
        iteration = context.get("iteration", 1)
        total = context.get("total", 1)
        
        prompt = f"Generate query {iteration} of {total} for persona '{persona.name}'."
        
        # Add strategy-specific context
        if strategy != "standard":
            prompt += f"\n\nGeneration strategy: {strategy}"
        
        # Template parameter injection
        if persona.scenario_templates:
            template = random.choice(persona.scenario_templates)
            injected_template, used_parameters = self._inject_template_parameters(template, persona)
            prompt += f"\n\nUse this scenario template: {injected_template}"
            
            # Store used parameters in context for metadata
            context["template_parameters_used"] = used_parameters
        
        # Add conversation context for context-aware generation
        if strategy == "context_aware" and context.get("conversation_history"):
            recent_interactions = context["conversation_history"][-2:]  # Last 2 interactions
            context_summary = self._summarize_conversation_context(recent_interactions)
            prompt += f"\n\nConversation context: {context_summary}"
        
        return prompt

    def _build_user_prompt(self, persona: Persona, context: Dict[str, Any]) -> str:
        """Build user prompt with context and scenario templates (legacy method)."""
        return self._build_enhanced_user_prompt(persona, context, "standard")

    def _inject_template_parameters(self, template: str, persona: Persona) -> tuple[str, List[str]]:
        """
        Inject random parameters from persona.template_parameters into templates.
        
        Args:
            template: Template string with {parameter} placeholders
            persona: Persona containing template_parameters
            
        Returns:
            Tuple of (injected_template, list_of_used_parameters)
        """
        used_parameters = []
        injected_template = template
        
        # Find all parameter placeholders in the template
        parameter_matches = re.findall(r'\{(\w+)\}', template)
        
        for param_name in parameter_matches:
            if param_name in persona.template_parameters:
                # Choose random value from available parameters
                param_values = persona.template_parameters[param_name]
                if param_values:
                    chosen_value = random.choice(param_values)
                    injected_template = injected_template.replace(f"{{{param_name}}}", chosen_value)
                    used_parameters.append(f"{param_name}={chosen_value}")
        
        return injected_template, used_parameters

    def _inject_conversation_context(self, messages: List[Dict[str, str]], conversation_history: List[TestInteraction]) -> List[Dict[str, str]]:
        """Inject conversation history into messages for context-aware generation."""
        if not conversation_history:
            return messages
        
        # Add recent conversation history before the user prompt
        context_messages = []
        for interaction in conversation_history[-3:]:  # Last 3 interactions
            context_messages.extend([
                {"role": "user", "content": f"Previous query: {interaction.query}"},
                {"role": "assistant", "content": f"Previous response: {interaction.agent_response[:200]}..."}
            ])
        
        # Insert context messages before the final user prompt
        return messages[:-1] + context_messages + [messages[-1]]

    def _summarize_conversation_context(self, interactions: List[TestInteraction]) -> str:
        """Summarize recent conversation interactions for context."""
        if not interactions:
            return "No previous context"
        
        summaries = []
        for i, interaction in enumerate(interactions, 1):
            query_preview = interaction.query[:100] + "..." if len(interaction.query) > 100 else interaction.query
            response_preview = interaction.agent_response[:100] + "..." if len(interaction.agent_response) > 100 else interaction.agent_response
            summaries.append(f"Exchange {i}: Q: {query_preview} | A: {response_preview}")
        
        return " | ".join(summaries)

    def _get_temperature_for_strategy(self, strategy: str) -> float:
        """Get appropriate temperature setting for different generation strategies."""
        temperature_map = {
            "standard": 0.8,
            "boundary_conditions": 0.9,
            "adversarial_inputs": 0.95,
            "ambiguous_requests": 0.85,
            "multi_part_queries": 0.7,
            "context_switching": 0.9,
            "error_scenarios": 0.9,
            "context_aware": 0.75
        }
        return temperature_map.get(strategy, 0.8)

    def _post_process_query(self, query_text: str, persona: Persona, strategy: str) -> str:
        """Post-process generated query based on strategy and persona."""
        # Clean up the query text
        query_text = query_text.strip()
        
        # Remove any unwanted formatting or prefixes
        prefixes_to_remove = ["Query:", "Question:", "User:", "Request:"]
        for prefix in prefixes_to_remove:
            if query_text.startswith(prefix):
                query_text = query_text[len(prefix):].strip()
        
        # Strategy-specific post-processing
        if strategy == "adversarial_inputs":
            # Ensure adversarial queries maintain persona characteristics
            if not any(pattern.lower() in query_text.lower() for pattern in persona.behavioral_patterns):
                # Add a subtle persona marker if missing
                tone = persona.tone_specifications.get("emotion", "neutral")
                if tone != "neutral" and tone not in query_text.lower():
                    query_text = self._add_persona_marker(query_text, tone)
        
        return query_text

    def _add_persona_marker(self, query_text: str, tone: str) -> str:
        """Add subtle persona markers to maintain character consistency."""
        tone_markers = {
            "frustrated": ["This is really important", "I need this resolved"],
            "confused": ["I'm not sure", "Can you clarify"],
            "urgent": ["ASAP", "immediately", "right away"],
            "polite": ["Please", "Thank you", "I would appreciate"]
        }
        
        markers = tone_markers.get(tone, [])
        if markers and not any(marker.lower() in query_text.lower() for marker in markers):
            marker = random.choice(markers)
            # Add marker naturally to the query
            if query_text.endswith("?"):
                query_text = f"{marker}, {query_text.lower()}"
            else:
                query_text = f"{query_text} {marker}."
        
        return query_text
    
    def _extract_query_and_thinking(self, response) -> tuple[str, Optional[str]]:
        """Extract both query text and thinking content from API response."""
        # Use the standardized DeepSeek client method for consistent parsing
        client = self._get_client()
        
        # Handle APIResponse object (primary case)
        if hasattr(response, 'content'):
            raw_content = response.content
        # Handle dict response (fallback case)
        elif isinstance(response, dict):
            raw_content = None
            if "choices" in response and response["choices"]:
                raw_content = response["choices"][0]["message"]["content"]
            elif "content" in response:
                raw_content = response["content"]
            
            if raw_content is None:
                raise QueryGenerationError("Invalid response format from API")
        else:
            raise QueryGenerationError("Invalid response format from API")
        
        # Use the standardized parsing method to separate reasoning from content
        reasoning, query_content = client.parse_reasoning_content(raw_content)
        
        # Return both the query and the thinking content
        final_query = query_content if query_content.strip() else raw_content.strip()
        thinking = reasoning if reasoning.strip() else None
        
        return final_query, thinking
    
    async def close(self):
        """Close the client connection."""
        if self._client:
            self._client.close()