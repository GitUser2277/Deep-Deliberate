"""
Configuration versioning and migration support for the DeepDeliberate framework.

This module provides version tracking, migration paths, and backward compatibility
for configuration files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.exceptions import ConfigurationError
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class MigrationResult(Enum):
    """Result of a configuration migration."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_NEEDED = "not_needed"


@dataclass
class MigrationInfo:
    """Information about a configuration migration."""
    from_version: str
    to_version: str
    description: str
    breaking_changes: List[str]
    migration_function: Callable[[Dict[str, Any]], Dict[str, Any]]
    validation_function: Optional[Callable[[Dict[str, Any]], List[str]]] = None


@dataclass
class VersionInfo:
    """Information about a configuration version."""
    version: str
    release_date: datetime
    description: str
    schema_changes: List[str]
    deprecated_fields: List[str] = None
    new_fields: List[str] = None
    
    def __post_init__(self):
        if self.deprecated_fields is None:
            self.deprecated_fields = []
        if self.new_fields is None:
            self.new_fields = []


class ConfigurationVersioning:
    """
    Configuration versioning and migration manager.
    
    Features:
    - Version tracking and validation
    - Automated migration paths
    - Backward compatibility checks
    - Schema evolution support
    """
    
    def __init__(self):
        self.versions = self._initialize_versions()
        self.migrations = self._initialize_migrations()
        self.current_version = "1.0.0"
    
    def _initialize_versions(self) -> Dict[str, VersionInfo]:
        """Initialize supported configuration versions."""
        return {
            "1.0.0": VersionInfo(
                version="1.0.0",
                release_date=datetime(2024, 1, 1),
                description="Initial configuration schema",
                schema_changes=[
                    "Initial persona configuration structure",
                    "Basic API settings",
                    "Security and logging settings",
                    "Performance optimization settings"
                ]
            )
        }
    
    def _initialize_migrations(self) -> List[MigrationInfo]:
        """Initialize migration paths between versions."""
        migrations = []
        
        # Future migrations would be added here
        # Example:
        # migrations.append(MigrationInfo(
        #     from_version="1.0.0",
        #     to_version="1.1.0",
        #     description="Add new persona inheritance features",
        #     breaking_changes=[],
        #     migration_function=self._migrate_1_0_to_1_1
        # ))
        
        return migrations
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self.versions.get(version)
    
    def is_version_supported(self, version: str) -> bool:
        """Check if a version is supported."""
        return version in self.versions
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[MigrationInfo]:
        """
        Get migration path between two versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of migrations to apply in order
        """
        if from_version == to_version:
            return []
        
        # For now, we only support direct migrations
        # In the future, this could support multi-step migration paths
        for migration in self.migrations:
            if migration.from_version == from_version and migration.to_version == to_version:
                return [migration]
        
        return []
    
    def migrate_configuration(
        self,
        config_dict: Dict[str, Any],
        from_version: str,
        to_version: str = None
    ) -> Tuple[Dict[str, Any], MigrationResult, List[str]]:
        """
        Migrate configuration from one version to another.
        
        Args:
            config_dict: Configuration dictionary to migrate
            from_version: Source version
            to_version: Target version (defaults to current version)
            
        Returns:
            Tuple of (migrated_config, result, messages)
        """
        if to_version is None:
            to_version = self.current_version
        
        messages = []
        
        # Check if migration is needed
        if from_version == to_version:
            return config_dict, MigrationResult.NOT_NEEDED, ["No migration needed"]
        
        # Validate source version
        if not self.is_version_supported(from_version):
            return config_dict, MigrationResult.FAILED, [f"Unsupported source version: {from_version}"]
        
        # Get migration path
        migration_path = self.get_migration_path(from_version, to_version)
        if not migration_path:
            return config_dict, MigrationResult.FAILED, [f"No migration path from {from_version} to {to_version}"]
        
        # Apply migrations
        migrated_config = config_dict.copy()
        try:
            for migration in migration_path:
                logger.info(f"Applying migration: {migration.description}")
                
                # Apply migration function
                migrated_config = migration.migration_function(migrated_config)
                
                # Validate migration result
                if migration.validation_function:
                    validation_issues = migration.validation_function(migrated_config)
                    if validation_issues:
                        messages.extend(validation_issues)
                        return migrated_config, MigrationResult.PARTIAL, messages
                
                messages.append(f"Successfully applied migration to {migration.to_version}")
            
            return migrated_config, MigrationResult.SUCCESS, messages
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            return config_dict, MigrationResult.FAILED, [error_msg]
    
    def validate_version_compatibility(
        self,
        config_dict: Dict[str, Any],
        target_version: str
    ) -> List[str]:
        """
        Validate configuration compatibility with a target version.
        
        Args:
            config_dict: Configuration dictionary to validate
            target_version: Target version to validate against
            
        Returns:
            List of compatibility issues
        """
        issues = []
        
        version_info = self.get_version_info(target_version)
        if not version_info:
            issues.append(f"Unknown target version: {target_version}")
            return issues
        
        # Check for deprecated fields
        if version_info.deprecated_fields:
            for field_path in version_info.deprecated_fields:
                if self._has_field(config_dict, field_path):
                    issues.append(f"Field '{field_path}' is deprecated in version {target_version}")
        
        # Check for required new fields
        if version_info.new_fields:
            for field_path in version_info.new_fields:
                if not self._has_field(config_dict, field_path):
                    issues.append(f"Field '{field_path}' is required in version {target_version}")
        
        return issues
    
    def get_backward_compatibility_info(self, version: str) -> Dict[str, Any]:
        """
        Get backward compatibility information for a version.
        
        Args:
            version: Version to check
            
        Returns:
            Dictionary with compatibility information
        """
        version_info = self.get_version_info(version)
        if not version_info:
            return {"supported": False, "reason": "Unknown version"}
        
        # Check if version is current or has migration path
        has_migration_path = bool(self.get_migration_path(version, self.current_version))
        
        return {
            "supported": version == self.current_version or has_migration_path,
            "version_info": {
                "version": version_info.version,
                "release_date": version_info.release_date.isoformat(),
                "description": version_info.description,
                "deprecated_fields": version_info.deprecated_fields,
                "new_fields": version_info.new_fields
            },
            "migration_available": has_migration_path,
            "breaking_changes": self._get_breaking_changes(version, self.current_version)
        }
    
    def create_version_report(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive version report for a configuration.
        
        Args:
            config_dict: Configuration dictionary to analyze
            
        Returns:
            Dictionary with version analysis
        """
        # Try to detect version from configuration
        detected_version = self._detect_version(config_dict)
        
        report = {
            "detected_version": detected_version,
            "current_version": self.current_version,
            "migration_needed": detected_version != self.current_version,
            "compatibility_issues": [],
            "migration_path": [],
            "recommendations": []
        }
        
        if detected_version:
            # Check compatibility
            report["compatibility_issues"] = self.validate_version_compatibility(
                config_dict, self.current_version
            )
            
            # Get migration path
            migration_path = self.get_migration_path(detected_version, self.current_version)
            report["migration_path"] = [
                {
                    "from_version": m.from_version,
                    "to_version": m.to_version,
                    "description": m.description,
                    "breaking_changes": m.breaking_changes
                }
                for m in migration_path
            ]
            
            # Generate recommendations
            if report["migration_needed"]:
                report["recommendations"].append("Consider migrating to the latest version")
            
            if report["compatibility_issues"]:
                report["recommendations"].append("Address compatibility issues before sharing")
        
        return report
    
    def _has_field(self, config_dict: Dict[str, Any], field_path: str) -> bool:
        """Check if a field exists in the configuration using dot notation."""
        parts = field_path.split('.')
        current = config_dict
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return True
    
    def _detect_version(self, config_dict: Dict[str, Any]) -> Optional[str]:
        """Detect configuration version from structure."""
        # For now, assume all configurations are version 1.0.0
        # In the future, this could analyze the structure to detect version
        
        # Check for version field in metadata
        if "metadata" in config_dict and "version" in config_dict["metadata"]:
            return config_dict["metadata"]["version"]
        
        # Check for version field at root level
        if "version" in config_dict:
            return config_dict["version"]
        
        # Default to current version if structure matches
        required_sections = ["personas", "api_settings", "security_settings", "logging_settings"]
        if all(section in config_dict for section in required_sections):
            return "1.0.0"
        
        return None
    
    def _get_breaking_changes(self, from_version: str, to_version: str) -> List[str]:
        """Get breaking changes between versions."""
        breaking_changes = []
        
        migration_path = self.get_migration_path(from_version, to_version)
        for migration in migration_path:
            breaking_changes.extend(migration.breaking_changes)
        
        return breaking_changes


# Global versioning manager instance
versioning_manager = ConfigurationVersioning()


def migrate_config(
    config_dict: Dict[str, Any],
    from_version: str,
    to_version: str = None
) -> Tuple[Dict[str, Any], MigrationResult, List[str]]:
    """
    Convenience function to migrate configuration.
    
    Args:
        config_dict: Configuration dictionary to migrate
        from_version: Source version
        to_version: Target version
        
    Returns:
        Tuple of (migrated_config, result, messages)
    """
    return versioning_manager.migrate_configuration(config_dict, from_version, to_version)


def get_version_info(version: str) -> Optional[VersionInfo]:
    """
    Convenience function to get version information.
    
    Args:
        version: Version to get information for
        
    Returns:
        VersionInfo or None if not found
    """
    return versioning_manager.get_version_info(version)


def create_version_report(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to create version report.
    
    Args:
        config_dict: Configuration dictionary to analyze
        
    Returns:
        Dictionary with version analysis
    """
    return versioning_manager.create_version_report(config_dict)