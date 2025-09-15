"""
Configuration management using Pydantic Settings.

This module provides type-safe configuration management with automatic
environment variable resolution and validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Persona, APISettings, SecuritySettings, LoggingSettings, PerformanceSettings


class DeepSeekSettings(BaseSettings):
    """DeepSeek API configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix='DEEPSEEK_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    endpoint: str = Field(..., description="DeepSeek API endpoint URL")
    api_key: str = Field(..., description="DeepSeek API key")
    model_name: str = Field(default="DeepSeek-R1", description="Model name to use")
    max_tokens: int = Field(default=3000, description="Maximum tokens per request")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    
    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must start with http:// or https://")
        return v.rstrip('/')
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if len(v) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v


class FrameworkSettings(BaseSettings):
    """Main framework configuration with nested settings."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    # Nested configuration sections
    deepseek: DeepSeekSettings
    api_settings: APISettings = Field(default_factory=APISettings)
    security_settings: SecuritySettings = Field(default_factory=SecuritySettings)
    logging_settings: LoggingSettings = Field(default_factory=LoggingSettings)
    performance_settings: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Personas loaded from config file
    personas: List[Persona] = Field(default_factory=list)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'FrameworkSettings':
        """
        Load configuration from JSON file with environment variable resolution.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            FrameworkSettings instance with resolved configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract personas separately
            personas_data = config_data.pop('personas', [])
            personas = [Persona(**persona_data) for persona_data in personas_data]
            
            # Create settings with personas
            settings = cls(**config_data)
            settings.personas = personas
            
            return settings
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    @model_validator(mode='after')
    def validate_personas(self):
        """Validate personas after loading."""
        if not self.personas:
            raise ValueError("At least one persona must be defined")
        
        # Check for duplicate persona names
        names = [persona.name for persona in self.personas]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate persona names found: {set(duplicates)}")
        
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
    
    def get_persona_by_name(self, name: str) -> Optional[Persona]:
        """Get persona by name, with inheritance resolved."""
        resolved_personas = self.resolve_persona_inheritance()
        for persona in resolved_personas:
            if persona.name == name:
                return persona
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            "deepseek": self.deepseek.model_dump(),
            "api_settings": self.api_settings.model_dump(),
            "security_settings": self.security_settings.model_dump(),
            "logging_settings": self.logging_settings.model_dump(),
            "performance_settings": self.performance_settings.model_dump(),
            "personas": [persona.model_dump() for persona in self.personas]
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ConfigurationManager:
    """
    Centralized configuration management with caching and validation.
    
    Features:
    - Configuration caching for performance
    - Environment-specific overrides
    - Validation and error reporting
    - Hot reloading support
    """
    
    def __init__(self):
        self._config_cache: Dict[str, FrameworkSettings] = {}
        self._file_timestamps: Dict[str, float] = {}
    
    def load_config(
        self, 
        config_path: str, 
        force_reload: bool = False
    ) -> FrameworkSettings:
        """
        Load configuration with caching support.
        
        Args:
            config_path: Path to configuration file
            force_reload: Force reload even if cached
            
        Returns:
            FrameworkSettings instance
        """
        config_file = Path(config_path)
        
        # Check if we need to reload
        if not force_reload and config_path in self._config_cache:
            if config_file.exists():
                current_mtime = config_file.stat().st_mtime
                cached_mtime = self._file_timestamps.get(config_path, 0)
                
                if current_mtime <= cached_mtime:
                    return self._config_cache[config_path]
        
        # Load fresh configuration
        settings = FrameworkSettings.load_from_file(config_path)
        
        # Cache the configuration
        self._config_cache[config_path] = settings
        if config_file.exists():
            self._file_timestamps[config_path] = config_file.stat().st_mtime
        
        return settings
    
    def validate_config(self, config_path: str) -> List[str]:
        """
        Validate configuration file and return list of issues.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        try:
            settings = FrameworkSettings.load_from_file(config_path)
            
            # Additional validation checks
            if not settings.personas:
                issues.append("No personas defined in configuration")
            
            # Validate persona inheritance
            try:
                settings.resolve_persona_inheritance()
            except ValueError as e:
                issues.append(f"Persona inheritance error: {e}")
            
            # Validate API settings
            if not settings.deepseek.api_key:
                issues.append("DeepSeek API key not configured")
            
        except Exception as e:
            issues.append(f"Configuration loading error: {e}")
        
        return issues
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self._file_timestamps.clear()


# Global configuration manager instance
config_manager = ConfigurationManager()


def load_config(config_path: str = "config.json") -> FrameworkSettings:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FrameworkSettings instance
    """
    return config_manager.load_config(config_path)


def validate_config(config_path: str = "config.json") -> List[str]:
    """
    Convenience function to validate configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation issues
    """
    return config_manager.validate_config(config_path)