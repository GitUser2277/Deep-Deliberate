"""
Validation utilities for the DeepDeliberate framework.

This module provides dedicated validators that can be used with Pydantic models
to separate validation logic from model definitions.
"""

import re
from typing import Dict, List, Set, Any, Optional
from urllib.parse import urlparse


class PersonaValidator:
    """Dedicated validator for persona configurations."""
    
    VALID_TONE_KEYS = {
        'formality', 'patience', 'technical_level', 'emotion', 
        'communication_style', 'urgency', 'politeness', 'directness'
    }
    
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validate persona name format."""
        if not name or not name.strip():
            raise ValueError("Persona name cannot be empty")
        
        name = name.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError(
                "Persona name must contain only alphanumeric characters, hyphens, and underscores"
            )
        
        if len(name) > 50:
            raise ValueError("Persona name must be 50 characters or less")
        
        return name
    
    @classmethod
    def validate_behavioral_patterns(cls, patterns: List[str]) -> List[str]:
        """Validate behavioral patterns."""
        if not patterns:
            return patterns
        
        validated_patterns = []
        for i, pattern in enumerate(patterns):
            if not pattern or not pattern.strip():
                raise ValueError(f"Behavioral pattern at index {i} cannot be empty")
            
            pattern = pattern.strip()
            if len(pattern) < 5:
                raise ValueError(f"Behavioral pattern at index {i} must be at least 5 characters long")
            
            if len(pattern) > 200:
                raise ValueError(f"Behavioral pattern at index {i} must be 200 characters or less")
            
            validated_patterns.append(pattern)
        
        return validated_patterns
    
    @classmethod
    def validate_tone_specifications(cls, tone_specs: Dict[str, str]) -> Dict[str, str]:
        """Validate tone specifications."""
        for key, value in tone_specs.items():
            if key not in cls.VALID_TONE_KEYS:
                raise ValueError(
                    f"Invalid tone specification key: '{key}'. "
                    f"Valid keys: {sorted(cls.VALID_TONE_KEYS)}"
                )
            
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Tone specification '{key}' must be a non-empty string")
            
            if len(value.strip()) > 100:
                raise ValueError(f"Tone specification '{key}' must be 100 characters or less")
        
        return {k: v.strip() for k, v in tone_specs.items()}
    
    @classmethod
    def validate_scenario_templates(cls, templates: List[str]) -> List[str]:
        """Validate scenario templates."""
        if not templates:
            return templates
        
        validated_templates = []
        for i, template in enumerate(templates):
            if not template or not template.strip():
                raise ValueError(f"Scenario template at index {i} cannot be empty")
            
            template = template.strip()
            if len(template) < 10:
                raise ValueError(f"Scenario template at index {i} must be at least 10 characters long")
            
            if len(template) > 500:
                raise ValueError(f"Scenario template at index {i} must be 500 characters or less")
            
            # Check for valid template parameter syntax
            cls._validate_template_parameters(template, i)
            
            validated_templates.append(template)
        
        return validated_templates
    
    @classmethod
    def _validate_template_parameters(cls, template: str, index: int) -> None:
        """Validate template parameter syntax."""
        # Find all {parameter} patterns
        parameter_pattern = r'\{([^}]+)\}'
        parameters = re.findall(parameter_pattern, template)
        
        for param in parameters:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', param):
                raise ValueError(
                    f"Invalid template parameter '{param}' in template at index {index}. "
                    f"Parameters must be valid identifiers (letters, numbers, underscores)"
                )
    
    @classmethod
    def validate_template_parameters(cls, template_params: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate template parameters."""
        validated_params = {}
        
        for param_name, param_values in template_params.items():
            if not param_name or not param_name.strip():
                raise ValueError("Template parameter names cannot be empty")
            
            param_name = param_name.strip()
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', param_name):
                raise ValueError(
                    f"Invalid template parameter name '{param_name}'. "
                    f"Must be a valid identifier (letters, numbers, underscores)"
                )
            
            if not isinstance(param_values, list):
                raise ValueError(f"Template parameter '{param_name}' values must be a list")
            
            if not param_values:
                raise ValueError(f"Template parameter '{param_name}' must have at least one value")
            
            validated_values = []
            for i, value in enumerate(param_values):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Template parameter '{param_name}' value at index {i} must be a non-empty string"
                    )
                
                value = value.strip()
                if len(value) > 100:
                    raise ValueError(
                        f"Template parameter '{param_name}' value at index {i} must be 100 characters or less"
                    )
                
                validated_values.append(value)
            
            validated_params[param_name] = validated_values
        
        return validated_params


class EvaluationCriteriaValidator:
    """Dedicated validator for evaluation criteria."""
    
    REQUIRED_CRITERIA = {'accuracy', 'relevance', 'completion', 'safety'}
    
    @classmethod
    def validate(cls, criteria: Dict[str, float]) -> Dict[str, float]:
        """Validate evaluation criteria weights."""
        if not criteria:
            raise ValueError("Evaluation criteria cannot be empty")
        
        cls._validate_required_criteria(criteria)
        cls._validate_criteria_weights(criteria)
        cls._validate_weights_sum(criteria)
        
        return criteria
    
    @classmethod
    def _validate_required_criteria(cls, criteria: Dict[str, float]) -> None:
        """Validate that all required criteria are present."""
        provided_criteria = set(criteria.keys())
        
        missing_criteria = cls.REQUIRED_CRITERIA - provided_criteria
        if missing_criteria:
            raise ValueError(f"Missing required evaluation criteria: {sorted(missing_criteria)}")
        
        invalid_criteria = provided_criteria - cls.REQUIRED_CRITERIA
        if invalid_criteria:
            raise ValueError(
                f"Invalid evaluation criteria: {sorted(invalid_criteria)}. "
                f"Valid criteria: {sorted(cls.REQUIRED_CRITERIA)}"
            )
    
    @classmethod
    def _validate_criteria_weights(cls, criteria: Dict[str, float]) -> None:
        """Validate individual criterion weights."""
        for criterion, weight in criteria.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Evaluation criterion '{criterion}' weight must be a number")
            
            if weight < 0 or weight > 1:
                raise ValueError(f"Evaluation criterion '{criterion}' weight must be between 0 and 1")
    
    @classmethod
    def _validate_weights_sum(cls, criteria: Dict[str, float]) -> None:
        """Validate that weights sum to 1.0."""
        total_weight = sum(criteria.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Evaluation criteria weights must sum to 1.0, got {total_weight:.3f}")


class APISettingsValidator:
    """Dedicated validator for API settings."""
    
    @classmethod
    def validate_endpoint(cls, endpoint: str) -> str:
        """Validate API endpoint URL."""
        if not endpoint or not endpoint.strip():
            raise ValueError("API endpoint cannot be empty")
        
        endpoint = endpoint.strip()
        
        # Parse URL to validate format
        try:
            parsed = urlparse(endpoint)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("API endpoint must be a valid URL with scheme and host")
            
            if parsed.scheme not in ('http', 'https'):
                raise ValueError("API endpoint must use HTTP or HTTPS protocol")
            
        except Exception as e:
            raise ValueError(f"Invalid API endpoint URL: {e}")
        
        return endpoint.rstrip('/')
    
    @classmethod
    def validate_timeout(cls, timeout: int) -> int:
        """Validate timeout value."""
        if not isinstance(timeout, int):
            raise ValueError("Timeout must be an integer")
        
        if timeout < 5:
            raise ValueError("Timeout must be at least 5 seconds")
        
        if timeout > 300:
            raise ValueError("Timeout cannot exceed 300 seconds (5 minutes)")
        
        return timeout
    
    @classmethod
    def validate_retry_attempts(cls, attempts: int) -> int:
        """Validate retry attempts."""
        if not isinstance(attempts, int):
            raise ValueError("Retry attempts must be an integer")
        
        if attempts < 0:
            raise ValueError("Retry attempts cannot be negative")
        
        if attempts > 10:
            raise ValueError("Retry attempts cannot exceed 10")
        
        return attempts
    
    @classmethod
    def validate_rate_limit(cls, rate_limit: int) -> int:
        """Validate rate limit."""
        if not isinstance(rate_limit, int):
            raise ValueError("Rate limit must be an integer")
        
        if rate_limit < 1:
            raise ValueError("Rate limit must be at least 1 request per minute")
        
        if rate_limit > 1000:
            raise ValueError("Rate limit cannot exceed 1000 requests per minute")
        
        return rate_limit


class LoggingSettingsValidator:
    """Dedicated validator for logging settings."""
    
    VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    
    @classmethod
    def validate_log_level(cls, level: str) -> str:
        """Validate log level."""
        if not isinstance(level, str):
            raise ValueError("Log level must be a string")
        
        level = level.upper().strip()
        if level not in cls.VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}. Valid levels: {sorted(cls.VALID_LOG_LEVELS)}")
        
        return level
    
    @classmethod
    def validate_output_directory(cls, directory: str) -> str:
        """Validate output directory path."""
        if not directory or not directory.strip():
            raise ValueError("Output directory cannot be empty")
        
        directory = directory.strip()
        
        # Check for invalid characters (basic validation)
        invalid_chars = '<>:"|?*'
        if any(char in directory for char in invalid_chars):
            raise ValueError(f"Output directory contains invalid characters: {invalid_chars}")
        
        return directory
    
    @classmethod
    def validate_retention_days(cls, days: int) -> int:
        """Validate log retention days."""
        if not isinstance(days, int):
            raise ValueError("Retention days must be an integer")
        
        if days < 1:
            raise ValueError("Retention days must be at least 1")
        
        if days > 365:
            raise ValueError("Retention days cannot exceed 365")
        
        return days


# Validation functions for use with Pydantic AfterValidator
def validate_persona_name(name: str) -> str:
    """Validation function for persona names."""
    return PersonaValidator.validate_name(name)


def validate_behavioral_patterns(patterns: List[str]) -> List[str]:
    """Validation function for behavioral patterns."""
    return PersonaValidator.validate_behavioral_patterns(patterns)


def validate_tone_specifications(tone_specs: Dict[str, str]) -> Dict[str, str]:
    """Validation function for tone specifications."""
    return PersonaValidator.validate_tone_specifications(tone_specs)


def validate_scenario_templates(templates: List[str]) -> List[str]:
    """Validation function for scenario templates."""
    return PersonaValidator.validate_scenario_templates(templates)


def validate_evaluation_criteria(criteria: Dict[str, float]) -> Dict[str, float]:
    """Validation function for evaluation criteria."""
    return EvaluationCriteriaValidator.validate(criteria)


def validate_template_parameters(params: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Validation function for template parameters."""
    return PersonaValidator.validate_template_parameters(params)


def validate_api_endpoint(endpoint: str) -> str:
    """Validation function for API endpoints."""
    return APISettingsValidator.validate_endpoint(endpoint)


def validate_log_level(level: str) -> str:
    """Validation function for log levels."""
    return LoggingSettingsValidator.validate_log_level(level)


def validate_output_directory(directory: str) -> str:
    """Validation function for output directories."""
    return LoggingSettingsValidator.validate_output_directory(directory)