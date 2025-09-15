"""
Configuration portability and sharing features for the DeepDeliberate framework.

This module provides functionality for exporting, importing, versioning,
and migrating configurations to support team sharing scenarios.
"""

import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import aiofiles
from pydantic import BaseModel, Field, ValidationError

from ..core.models import FrameworkConfig, Persona
from ..core.exceptions import ConfigurationError
from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigurationMetadata:
    """Metadata for configuration export/import."""
    version: str
    export_timestamp: datetime
    framework_version: str
    exported_by: Optional[str] = None
    description: Optional[str] = None
    checksum: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ConfigurationExport(BaseModel):
    """Exportable configuration package with metadata."""
    metadata: Dict[str, Any] = Field(..., description="Export metadata")
    configuration: Dict[str, Any] = Field(..., description="Configuration data")
    environment_template: Dict[str, str] = Field(
        default_factory=dict, 
        description="Environment variable template"
    )
    migration_notes: List[str] = Field(
        default_factory=list,
        description="Notes for configuration migration"
    )


class ConfigurationPortability:
    """
    Configuration portability manager for export, import, and versioning.
    
    Features:
    - Configuration export with metadata
    - Configuration import with validation
    - Version tracking and migration
    - Environment variable template generation
    - Team sharing support
    """
    
    CURRENT_VERSION = "1.0.0"
    FRAMEWORK_VERSION = "1.0.0"  # Should match actual framework version
    
    def __init__(self):
        self.supported_versions = ["1.0.0"]
    
    async def export_configuration(
        self,
        config: FrameworkConfig,
        export_path: str,
        metadata: Optional[ConfigurationMetadata] = None,
        include_environment_template: bool = True
    ) -> str:
        """
        Export configuration to a portable format.
        
        Args:
            config: FrameworkConfig to export
            export_path: Path for the exported configuration file
            metadata: Optional metadata for the export
            include_environment_template: Whether to include env var template
            
        Returns:
            Path to the exported configuration file
            
        Raises:
            ConfigurationError: If export fails
        """
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create metadata if not provided
            if metadata is None:
                metadata = ConfigurationMetadata(
                    version=self.CURRENT_VERSION,
                    export_timestamp=datetime.now(),
                    framework_version=self.FRAMEWORK_VERSION,
                    description="Exported DeepDeliberate configuration"
                )
            
            # Convert configuration to dictionary
            config_dict = self._config_to_dict(config)
            
            # Generate environment template
            env_template = {}
            if include_environment_template:
                env_template = self._generate_environment_template(config_dict)
            
            # Calculate checksum
            config_json = json.dumps(config_dict, sort_keys=True)
            checksum = hashlib.sha256(config_json.encode()).hexdigest()
            metadata.checksum = checksum
            
            # Create export package
            export_package = ConfigurationExport(
                metadata=asdict(metadata),
                configuration=config_dict,
                environment_template=env_template,
                migration_notes=self._generate_migration_notes(config)
            )
            
            # Write export file asynchronously
            async with aiofiles.open(export_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(export_package.model_dump(), indent=2, default=str))
            
            logger.info(f"Configuration exported successfully to: {export_path}")
            return str(export_file)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to export configuration: {e}")
    
    async def import_configuration(
        self,
        import_path: str,
        validate_checksum: bool = True,
        apply_migrations: bool = True
    ) -> Tuple[FrameworkConfig, ConfigurationMetadata]:
        """
        Import configuration from exported file.
        
        Args:
            import_path: Path to the exported configuration file
            validate_checksum: Whether to validate configuration checksum
            apply_migrations: Whether to apply version migrations
            
        Returns:
            Tuple of (FrameworkConfig, ConfigurationMetadata)
            
        Raises:
            ConfigurationError: If import fails
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                raise ConfigurationError(f"Import file not found: {import_path}")
            
            # Load export package asynchronously
            async with aiofiles.open(import_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                export_data = json.loads(content)
            
            export_package = ConfigurationExport(**export_data)
            metadata = ConfigurationMetadata(**export_package.metadata)
            
            # Validate version compatibility
            if metadata.version not in self.supported_versions:
                if not apply_migrations:
                    raise ConfigurationError(
                        f"Unsupported configuration version: {metadata.version}"
                    )
                logger.warning(f"Configuration version {metadata.version} requires migration")
            
            # Validate checksum if requested
            if validate_checksum and metadata.checksum:
                config_json = json.dumps(export_package.configuration, sort_keys=True)
                calculated_checksum = hashlib.sha256(config_json.encode()).hexdigest()
                
                if calculated_checksum != metadata.checksum:
                    raise ConfigurationError("Configuration checksum validation failed")
            
            # Apply migrations if needed
            config_dict = export_package.configuration
            if apply_migrations and metadata.version != self.CURRENT_VERSION:
                config_dict = self._apply_migrations(config_dict, metadata.version)
            
            # Create FrameworkConfig
            config = self._dict_to_config(config_dict)
            
            logger.info(f"Configuration imported successfully from: {import_path}")
            return config, metadata
            
        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration: {e}")
    
    def validate_for_sharing(self, config: FrameworkConfig) -> List[str]:
        """
        Validate configuration for team sharing scenarios.
        
        Args:
            config: FrameworkConfig to validate
            
        Returns:
            List of validation issues (empty if valid for sharing)
        """
        issues = []
        
        try:
            # Check for environment-specific settings
            config_dict = self._config_to_dict(config)
            
            # Check API settings for hardcoded values
            api_settings = config_dict.get('api_settings', {})
            if isinstance(api_settings.get('deepseek_endpoint'), str):
                if not api_settings['deepseek_endpoint'].startswith('${'):
                    issues.append("API endpoint should use environment variable for sharing")
            
            # Check logging settings for absolute paths
            logging_settings = config_dict.get('logging_settings', {})
            output_dir = logging_settings.get('output_directory', '')
            if output_dir and Path(output_dir).is_absolute():
                issues.append("Output directory should use relative path for sharing")
            
            # Check personas for team-specific content
            personas = config.personas
            for persona in personas:
                if persona.name.lower() in ['test', 'debug', 'local']:
                    issues.append(f"Persona '{persona.name}' appears to be for local testing only")
            
            # Check security settings
            security_settings = config_dict.get('security_settings', {})
            if not security_settings.get('strict_mode', True):
                issues.append("Strict mode should be enabled for shared configurations")
            
            if security_settings.get('allow_dangerous_functions', False):
                issues.append("Dangerous functions should not be allowed in shared configurations")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues
    
    async def create_environment_template(self, config: FrameworkConfig, output_path: str) -> str:
        """
        Create environment variable template file for team sharing.
        
        Args:
            config: FrameworkConfig to analyze
            output_path: Path for the environment template file
            
        Returns:
            Path to the created template file
        """
        try:
            config_dict = self._config_to_dict(config)
            env_template = self._generate_environment_template(config_dict)
            
            template_file = Path(output_path)
            template_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create .env template content
            template_content = [
                "# DeepDeliberate Framework Environment Variables",
                "# Copy this file to .env and fill in your values",
                "",
                "# DeepSeek API Configuration",
                "DEEPSEEK_API_KEY=your_api_key_here",
                "DEEPSEEK_ENDPOINT=https://api.deepseek.com",
                "",
            ]
            
            # Add discovered environment variables
            for var_name, description in env_template.items():
                template_content.append(f"# {description}")
                template_content.append(f"{var_name}=")
                template_content.append("")
            
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write('\n'.join(template_content))
            
            logger.info(f"Environment template created: {output_path}")
            return str(template_file)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment template: {e}")
    
    def _config_to_dict(self, config: FrameworkConfig) -> Dict[str, Any]:
        """Convert FrameworkConfig to dictionary."""
        return {
            "personas": [persona.model_dump() for persona in config.personas],
            "api_settings": config.api_settings.model_dump(),
            "security_settings": config.security_settings.model_dump(),
            "logging_settings": config.logging_settings.model_dump(),
            "performance_settings": config.performance_settings.model_dump()
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> FrameworkConfig:
        """Convert dictionary to FrameworkConfig."""
        return FrameworkConfig(**config_dict)
    
    def _generate_environment_template(self, config_dict: Dict[str, Any]) -> Dict[str, str]:
        """Generate environment variable template from configuration."""
        env_vars = {}
        
        def scan_for_env_vars(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        var_name = value[2:-1]
                        env_vars[var_name] = f"Used in {current_path}"
                    elif isinstance(value, (dict, list)):
                        scan_for_env_vars(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    scan_for_env_vars(item, current_path)
        
        scan_for_env_vars(config_dict)
        return env_vars
    
    def _generate_migration_notes(self, config: FrameworkConfig) -> List[str]:
        """Generate migration notes for configuration."""
        notes = []
        
        # Check for potential migration issues
        if len(config.personas) > 10:
            notes.append("Large number of personas may impact performance")
        
        # Check API settings
        if config.api_settings.timeout_seconds > 60:
            notes.append("High timeout values may cause issues in some environments")
        
        # Check performance settings
        if config.performance_settings.max_concurrent_requests > 20:
            notes.append("High concurrency may require additional system resources")
        
        return notes
    
    def _apply_migrations(self, config_dict: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """Apply version migrations to configuration."""
        migrated_config = config_dict.copy()
        
        # Future migration logic would go here
        # For now, we only support version 1.0.0
        if from_version == "1.0.0":
            # No migration needed for same version
            pass
        else:
            logger.warning(f"No migration path defined for version {from_version}")
        
        return migrated_config


# Global portability manager instance
portability_manager = ConfigurationPortability()


async def export_config(
    config: FrameworkConfig,
    export_path: str,
    metadata: Optional[ConfigurationMetadata] = None
) -> str:
    """
    Convenience function to export configuration.
    
    Args:
        config: FrameworkConfig to export
        export_path: Path for the exported configuration file
        metadata: Optional metadata for the export
        
    Returns:
        Path to the exported configuration file
    """
    return await portability_manager.export_configuration(config, export_path, metadata)


async def import_config(import_path: str) -> Tuple[FrameworkConfig, ConfigurationMetadata]:
    """
    Convenience function to import configuration.
    
    Args:
        import_path: Path to the exported configuration file
        
    Returns:
        Tuple of (FrameworkConfig, ConfigurationMetadata)
    """
    return await portability_manager.import_configuration(import_path)


def validate_for_sharing(config: FrameworkConfig) -> List[str]:
    """
    Convenience function to validate configuration for sharing.
    
    Args:
        config: FrameworkConfig to validate
        
    Returns:
        List of validation issues
    """
    return portability_manager.validate_for_sharing(config)