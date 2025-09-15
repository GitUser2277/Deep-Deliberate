"""
Configuration management for the DeepDeliberate framework.

This module provides centralized configuration loading, validation,
and management capabilities.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import ValidationError

from ..core.models import FrameworkConfig, Persona
from ..core.exceptions import ConfigurationError
from ..core.logging_config import get_logger
from .portability import ConfigurationPortability, ConfigurationMetadata
from .versioning import ConfigurationVersioning, MigrationResult

logger = get_logger(__name__)


class ConfigManager:
    """
    Centralized configuration manager for the DeepDeliberate framework.
    
    Features:
    - JSON configuration file loading
    - Pydantic-based validation
    - Environment variable resolution
    - Configuration caching
    - Error reporting with detailed messages
    """
    
    def __init__(self):
        self._config_cache: Dict[str, FrameworkConfig] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._portability = ConfigurationPortability()
        self._versioning = ConfigurationVersioning()
    
    def load_config(self, config_path: str) -> FrameworkConfig:
        """
        Load and validate configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Validated FrameworkConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid or file not found
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            # Check cache
            current_mtime = config_file.stat().st_mtime
            cached_mtime = self._file_timestamps.get(config_path, 0)
            
            if (config_path in self._config_cache and 
                current_mtime <= cached_mtime):
                logger.debug(f"Using cached configuration: {config_path}")
                return self._config_cache[config_path]
            
            # Load fresh configuration
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate and create configuration
            config = self._create_framework_config(config_data)
            
            # Cache the configuration
            self._config_cache[config_path] = config
            self._file_timestamps[config_path] = current_mtime
            
            logger.info(f"Configuration loaded successfully: {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {config_path}: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _create_framework_config(self, config_data: Dict[str, Any]) -> FrameworkConfig:
        """
        Create FrameworkConfig from loaded data.
        
        Args:
            config_data: Raw configuration dictionary
            
        Returns:
            Validated FrameworkConfig instance
        """
        try:
            # Extract personas
            personas_data = config_data.get('personas', [])
            personas = [Persona(**persona_data) for persona_data in personas_data]
            
            if not personas:
                raise ConfigurationError("At least one persona must be defined")
            
            # Resolve environment variables in config data
            resolved_config = self._resolve_environment_variables(config_data)
            
            # Create framework configuration using Pydantic validation
            framework_config = FrameworkConfig(**resolved_config)
            
            return framework_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create framework configuration: {e}")
    
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
            config = self.load_config(config_path)
            
            # Additional validation checks
            if not config.personas:
                issues.append("No personas defined in configuration")
            
            # Check for duplicate persona names
            persona_names = [p.name for p in config.personas]
            if len(persona_names) != len(set(persona_names)):
                duplicates = [name for name in persona_names if persona_names.count(name) > 1]
                issues.append(f"Duplicate persona names found: {set(duplicates)}")
            
            # Validate environment variables
            env_issues = self._validate_environment()
            issues.extend(env_issues)
            
        except ConfigurationError as e:
            issues.append(str(e))
        except Exception as e:
            issues.append(f"Unexpected validation error: {e}")
        
        return issues
    
    def _validate_environment(self) -> List[str]:
        """
        Validate required environment variables.
        
        Returns:
            List of environment validation issues
        """
        issues = []
        
        required_env_vars = [
            'DEEPSEEK_API_KEY',
            'DEEPSEEK_ENDPOINT'
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                issues.append(f"Required environment variable not set: {var}")
        
        return issues
    
    def _resolve_environment_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve environment variables in configuration data.
        
        Args:
            config_data: Configuration dictionary with potential env var references
            
        Returns:
            Configuration dictionary with resolved environment variables
        """
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract environment variable name
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                
                if env_value is None:
                    logger.warning(f"Environment variable {env_var} not set, using default")
                    return self._get_default_value(env_var)
                
                # Try to convert to appropriate type
                return self._convert_env_value(env_value)
            
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            
            return value
        
        return resolve_value(config_data)
    
    def _get_default_value(self, env_var: str) -> Any:
        """Get default value for environment variable."""
        defaults = {
            'DEEPSEEK_ENDPOINT': 'https://api.deepseek.com',
            'REQUEST_TIMEOUT_SECONDS': 30,
            'API_RATE_LIMIT_PER_MINUTE': 60,
            'VERIFY_SSL_CERTIFICATES': True,
            'ENABLE_REQUEST_SIGNING': True,
            'MAX_INPUT_LENGTH': 1000,
            'ENABLE_AGENT_VALIDATION': True,
            'ALLOW_DANGEROUS_FUNCTIONS': False,
            'SANDBOX_AGENT_EXECUTION': True,
            'ENABLE_DATA_REDACTION': True,
            'ENCRYPT_LOGS': True,
            'ENABLE_AUDIT_LOGGING': True,
            'LOG_RETENTION_DAYS': 30,
            'MAX_CONCURRENT_REQUESTS': 5
        }
        return defaults.get(env_var, '')
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self._file_timestamps.clear()
        logger.debug("Configuration cache cleared")
    
    def get_cached_configs(self) -> List[str]:
        """Get list of cached configuration file paths."""
        return list(self._config_cache.keys())
    
    def export_config(
        self,
        config_path: str,
        export_path: str,
        metadata: Optional[ConfigurationMetadata] = None
    ) -> str:
        """
        Export configuration to a portable format.
        
        Args:
            config_path: Path to configuration file to export
            export_path: Path for the exported configuration file
            metadata: Optional metadata for the export
            
        Returns:
            Path to the exported configuration file
        """
        config = self.load_config(config_path)
        return self._portability.export_configuration(config, export_path, metadata)
    
    def import_config(
        self,
        import_path: str,
        target_path: str,
        validate_checksum: bool = True
    ) -> FrameworkConfig:
        """
        Import configuration from exported file.
        
        Args:
            import_path: Path to the exported configuration file
            target_path: Path where to save the imported configuration
            validate_checksum: Whether to validate configuration checksum
            
        Returns:
            Imported FrameworkConfig instance
        """
        config, metadata = self._portability.import_configuration(
            import_path, validate_checksum
        )
        
        # Save imported configuration to target path
        config_dict = self._portability._config_to_dict(config)
        target_file = Path(target_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration imported and saved to: {target_path}")
        return config
    
    def validate_for_sharing(self, config_path: str) -> List[str]:
        """
        Validate configuration for team sharing scenarios.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            List of validation issues for sharing
        """
        config = self.load_config(config_path)
        return self._portability.validate_for_sharing(config)
    
    def create_environment_template(self, config_path: str, template_path: str) -> str:
        """
        Create environment variable template for team sharing.
        
        Args:
            config_path: Path to configuration file
            template_path: Path for the environment template file
            
        Returns:
            Path to the created template file
        """
        config = self.load_config(config_path)
        return self._portability.create_environment_template(config, template_path)
    
    def get_version_report(self, config_path: str) -> Dict[str, Any]:
        """
        Get version analysis report for configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with version analysis
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return self._versioning.create_version_report(config_dict)
    
    def migrate_config(
        self,
        config_path: str,
        target_path: str,
        from_version: str,
        to_version: str = None
    ) -> MigrationResult:
        """
        Migrate configuration between versions.
        
        Args:
            config_path: Path to source configuration file
            target_path: Path for migrated configuration file
            from_version: Source version
            to_version: Target version (defaults to current)
            
        Returns:
            Migration result
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        migrated_config, result, messages = self._versioning.migrate_configuration(
            config_dict, from_version, to_version
        )
        
        if result in [MigrationResult.SUCCESS, MigrationResult.PARTIAL]:
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(migrated_config, f, indent=2, default=str)
            
            logger.info(f"Configuration migrated and saved to: {target_path}")
        
        for message in messages:
            logger.info(f"Migration: {message}")
        
        return result


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: str = "config.json") -> FrameworkConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FrameworkConfig instance
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