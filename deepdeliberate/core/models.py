"""
Core data models for the DeepDeliberate framework.

This module defines the fundamental data structures used throughout
the framework for configuration, testing, and evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator, AfterValidator

from .validators import (
    validate_persona_name, validate_behavioral_patterns, validate_tone_specifications,
    validate_scenario_templates, validate_evaluation_criteria, validate_template_parameters,
    validate_api_endpoint, validate_log_level, validate_output_directory
)

__all__ = [
    "ExecutionMode",
    "UserDecision", 
    "AgentMetadata",
    "Persona",
    "APISettings",
    "SecuritySettings",
    "LoggingSettings",
    "PerformanceSettings",
    "FrameworkConfig",
    "GeneratedQuery",
    "AgentResponse",
    "EvaluationResult",
    "TestInteraction",
    "SessionState",
    "CLIConfig"
]


class ExecutionMode(str, Enum):
    """Execution modes for the testing framework."""
    AUTO = "auto"
    APPROVE = "approve"


class UserDecision(str, Enum):
    """User decisions in approve mode."""
    CONTINUE = "continue"
    RETRY = "retry"
    EXIT = "exit"


class AgentMetadata(BaseModel):
    """Metadata about a discovered PydanticAI agent."""
    name: str
    model: str
    output_type: str
    deps_type: Optional[str] = None
    file_path: str
    
    
class Persona(BaseModel):
    """Configuration for a testing persona with inheritance support."""
    name: Annotated[str, AfterValidator(validate_persona_name)] = Field(
        ..., description="Unique identifier for the persona", min_length=1
    )
    description: Optional[str] = Field(None, description="Human-readable description of the persona")
    inherits_from: Optional[str] = Field(None, description="Name of parent persona to inherit from")
    
    # Core persona attributes with dedicated validators
    behavioral_patterns: Annotated[List[str], AfterValidator(validate_behavioral_patterns)] = Field(
        default_factory=list, 
        description="List of behavioral characteristics"
    )
    tone_specifications: Annotated[Dict[str, str], AfterValidator(validate_tone_specifications)] = Field(
        default_factory=dict, 
        description="Tone and communication style specifications"
    )
    scenario_templates: Annotated[List[str], AfterValidator(validate_scenario_templates)] = Field(
        default_factory=list,
        description="Template scenarios for query generation with {parameter} placeholders"
    )
    evaluation_criteria: Annotated[Dict[str, float], AfterValidator(validate_evaluation_criteria)] = Field(
        default_factory=lambda: {"accuracy": 0.25, "relevance": 0.25, "completion": 0.25, "safety": 0.25},
        description="Weighted evaluation criteria (must sum to 1.0)"
    )
    
    # Template parameters for dynamic injection
    template_parameters: Annotated[Dict[str, List[str]], AfterValidator(validate_template_parameters)] = Field(
        default_factory=dict,
        description="Parameters that can be injected into scenario templates"
    )
    

    
    def merge_with_parent(self, parent: 'Persona') -> 'Persona':
        """Merge this persona with its parent, with child values taking precedence."""
        merged_data = parent.model_dump(exclude={'name', 'description', 'inherits_from'})
        child_data = self.model_dump(exclude_unset=True, exclude={'inherits_from'})
        
        # Merge lists by extending parent lists with child lists
        for list_field in ['behavioral_patterns', 'scenario_templates']:
            if list_field in child_data and list_field in merged_data:
                merged_data[list_field] = merged_data[list_field] + child_data[list_field]
            elif list_field in child_data:
                merged_data[list_field] = child_data[list_field]
        
        # Merge dictionaries by updating parent dicts with child dicts
        for dict_field in ['tone_specifications', 'evaluation_criteria', 'template_parameters']:
            if dict_field in child_data and dict_field in merged_data:
                merged_data[dict_field].update(child_data[dict_field])
            elif dict_field in child_data:
                merged_data[dict_field] = child_data[dict_field]
        
        # Override scalar fields
        for scalar_field in ['name', 'description']:
            if scalar_field in child_data:
                merged_data[scalar_field] = child_data[scalar_field]
        
        return Persona(**merged_data)
    

class APISettings(BaseModel):
    """API configuration settings with comprehensive validation."""
    deepseek_endpoint: Annotated[str, AfterValidator(validate_api_endpoint)] = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API endpoint URL"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=5,
        le=300
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
        ge=0,
        le=10
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute",
        ge=1,
        le=1000
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates"
    )
    enable_request_signing: bool = Field(
        default=False,
        description="Whether to enable request signing for enhanced security"
    )
    max_input_length: int = Field(
        default=8000,
        description="Maximum input length in characters",
        ge=100,
        le=50000
    )


class SecuritySettings(BaseModel):
    """Security configuration settings."""
    enable_agent_validation: bool = Field(
        default=True,
        description="Whether to validate agents before execution"
    )
    allow_dangerous_functions: bool = Field(
        default=False,
        description="Whether to allow potentially dangerous function calls"
    )
    sandbox_agent_execution: bool = Field(
        default=True,
        description="Whether to sandbox agent execution"
    )
    enable_data_redaction: bool = Field(
        default=True,
        description="Whether to redact sensitive data in logs"
    )
    strict_mode: bool = Field(
        default=True,
        description="Whether to enable strict security mode"
    )


class LoggingSettings(BaseModel):
    """Logging configuration settings with comprehensive validation."""
    output_directory: Annotated[str, AfterValidator(validate_output_directory)] = Field(
        default="test_results",
        description="Directory for output files"
    )
    log_level: Annotated[str, AfterValidator(validate_log_level)] = Field(
        default="INFO",
        description="Logging level"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in logs"
    )
    encrypt_logs: bool = Field(
        default=False,
        description="Whether to encrypt log files"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Whether to enable audit logging"
    )
    retention_days: int = Field(
        default=30,
        description="Number of days to retain logs",
        ge=1,
        le=365
    )


class PerformanceSettings(BaseModel):
    """Performance optimization settings with comprehensive validation."""
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent API requests",
        ge=1,
        le=100
    )
    memory_limit_gb: int = Field(
        default=2,
        description="Memory limit in gigabytes",
        ge=1,
        le=32
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable response caching"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
        ge=60,
        le=86400
    )
    batch_size: int = Field(
        default=5,
        description="Batch size for processing multiple requests",
        ge=1,
        le=50
    )


class FrameworkConfig(BaseModel):
    """Main configuration for the DeepDeliberate framework with comprehensive validation."""
    personas: List[Persona] = Field(
        ...,
        description="List of testing personas",
        min_items=1
    )
    api_settings: APISettings = Field(
        default_factory=APISettings,
        description="API configuration settings"
    )
    security_settings: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security configuration settings"
    )
    logging_settings: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging configuration settings"
    )
    performance_settings: PerformanceSettings = Field(
        default_factory=PerformanceSettings,
        description="Performance optimization settings"
    )
    
    @field_validator('personas')
    @classmethod
    def validate_personas(cls, v):
        """Validate personas list and check for duplicates."""
        if not v:
            raise ValueError("At least one persona must be defined")
        
        # Check for duplicate persona names
        names = [persona.name for persona in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate persona names found: {set(duplicates)}")
        
        return v
    
    @model_validator(mode='after')
    def validate_persona_inheritance(self):
        """Validate persona inheritance relationships."""
        personas = self.personas
        if not personas:
            return self
        
        # Build persona name mapping
        persona_map = {persona.name: persona for persona in personas}
        
        # Validate inheritance relationships
        for persona in personas:
            if persona.inherits_from:
                if persona.inherits_from not in persona_map:
                    raise ValueError(
                        f"Persona '{persona.name}' inherits from '{persona.inherits_from}' "
                        f"which is not defined"
                    )
                
                # Check for circular inheritance
                visited = set()
                current = persona
                while current.inherits_from:
                    if current.name in visited:
                        raise ValueError(
                            f"Circular inheritance detected involving persona '{persona.name}'"
                        )
                    visited.add(current.name)
                    current = persona_map.get(current.inherits_from)
                    if not current:
                        break
        
        return self
    
    def resolve_persona_inheritance(self) -> List[Persona]:
        """Resolve persona inheritance and return fully merged personas."""
        persona_map = {persona.name: persona for persona in self.personas}
        resolved_personas = []
        
        def resolve_persona(persona: Persona, visited: set = None) -> Persona:
            if visited is None:
                visited = set()
            
            if persona.name in visited:
                raise ValueError(f"Circular inheritance detected for persona '{persona.name}'")
            
            if not persona.inherits_from:
                return persona
            
            visited.add(persona.name)
            parent = persona_map.get(persona.inherits_from)
            if not parent:
                raise ValueError(f"Parent persona '{persona.inherits_from}' not found")
            
            resolved_parent = resolve_persona(parent, visited.copy())
            return persona.merge_with_parent(resolved_parent)
        
        for persona in self.personas:
            resolved_personas.append(resolve_persona(persona))
        
        return resolved_personas


@dataclass
class GeneratedQuery:
    """A query generated for testing."""
    text: str
    persona_name: str
    context: Dict[str, Any]
    generation_time: datetime
    metadata: Dict[str, Any]
    thinking_content: Optional[str] = None  # DeepSeek thinking content


@dataclass
class AgentResponse:
    """Response from an AI agent."""
    text: str
    execution_time: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating an agent response."""
    overall_score: float
    dimension_scores: Dict[str, float]  # accuracy, relevance, completion, safety
    reasoning: str
    confidence: float
    evaluation_time: float


@dataclass
class TestInteraction:
    """Complete test interaction record."""
    timestamp: datetime
    session_id: str
    persona_name: str
    query: str
    agent_response: str
    evaluation_score: float
    evaluation_reasoning: str
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class SessionState:
    """State of a testing session."""
    session_id: str
    current_iteration: int
    total_iterations: int
    mode: ExecutionMode
    persona: Persona
    completed_interactions: List[TestInteraction]
    checkpoint_data: Dict[str, Any]


class CLIConfig(BaseModel):
    """Configuration from CLI arguments."""
    agent_file: str
    mode: ExecutionMode
    count: int = Field(default=10)
    persona: Optional[str] = None
    output_dir: Optional[str] = None
    config_file: str = Field(default="config.json")
    enhanced_evaluator: bool = Field(default=False)