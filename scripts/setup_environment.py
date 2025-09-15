#!/usr/bin/env python3
"""
Environment setup and validation script for DeepDeliberate framework.

This script validates the environment, checks dependencies, and guides users
through initial configuration setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import click
    from deepdeliberate.core.logging_config import get_logger
    from deepdeliberate.config.manager import ConfigManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

logger = get_logger(__name__)


class EnvironmentValidator:
    """Validates the environment for DeepDeliberate framework."""
    
    def __init__(self):
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.python_version = sys.version_info
        self.platform_info = platform.platform()
    
    def validate_python_version(self) -> bool:
        """Validate Python version compatibility."""
        min_version = (3, 8)
        if self.python_version < min_version:
            self.issues.append(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"found {self.python_version.major}.{self.python_version.minor}"
            )
            return False
        
        if self.python_version >= (3, 12):
            self.warnings.append(
                f"Python {self.python_version.major}.{self.python_version.minor} "
                "is very new - some dependencies may have compatibility issues"
            )
        
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate that all required dependencies are installed."""
        required_packages = [
            'pydantic', 'pydantic-ai', 'click', 'httpx', 'tenacity',
            'python-dotenv', 'aiofiles', 'pandas', 'tiktoken'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append(
                f"Missing required packages: {', '.join(missing_packages)}"
            )
            return False
        
        return True
    
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are configured."""
        required_keys = ['DEEPSEEK_API_KEY']
        optional_keys = ['OPENAI_API_KEY', 'OPENAI_AZURE_ENDPOINT']
        
        missing_required = []
        missing_optional = []
        
        for key in required_keys:
            if not os.getenv(key):
                missing_required.append(key)
        
        for key in optional_keys:
            if not os.getenv(key):
                missing_optional.append(key)
        
        if missing_required:
            self.issues.append(
                f"Missing required API keys: {', '.join(missing_required)}"
            )
        
        if missing_optional:
            self.warnings.append(
                f"Missing optional API keys: {', '.join(missing_optional)}"
            )
        
        return len(missing_required) == 0
    
    def validate_file_permissions(self) -> bool:
        """Validate file system permissions."""
        test_dirs = ['test_results', '.']
        
        for dir_path in test_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                test_file = os.path.join(dir_path, '.permission_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (OSError, PermissionError) as e:
                self.issues.append(
                    f"Cannot write to directory {dir_path}: {e}"
                )
                return False
        
        return True
    
    def validate_memory(self) -> bool:
        """Validate available memory."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_gb < 1:
                self.issues.append(
                    f"Insufficient memory: {available_gb:.1f}GB available, "
                    "minimum 1GB recommended"
                )
                return False
            elif available_gb < 2:
                self.warnings.append(
                    f"Low memory: {available_gb:.1f}GB available, "
                    "2GB+ recommended for optimal performance"
                )
        except ImportError:
            self.warnings.append("Cannot check memory usage (psutil not installed)")
        
        return True
    
    def run_validation(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks."""
        checks = [
            self.validate_python_version,
            self.validate_dependencies,
            self.validate_api_keys,
            self.validate_file_permissions,
            self.validate_memory
        ]
        
        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
        
        return all_passed, self.issues, self.warnings


class SetupWizard:
    """Interactive setup wizard for first-time users."""
    
    def __init__(self):
        self.config_path = Path('config.json')
        self.env_path = Path('.env')
    
    def run_wizard(self) -> bool:
        """Run the interactive setup wizard."""
        click.echo("🧙 DeepDeliberate Setup Wizard")
        click.echo("=" * 40)
        
        # Check if already configured
        if self.config_path.exists() and self.env_path.exists():
            if not click.confirm("Configuration files already exist. Reconfigure?"):
                return True
        
        # API Key Configuration
        if not self._configure_api_keys():
            return False
        
        # Basic Configuration
        if not self._configure_basic_settings():
            return False
        
        # Test Configuration
        if not self._test_configuration():
            return False
        
        click.echo("\n✅ Setup completed successfully!")
        click.echo("You can now run: python deepdeliberate.py -file <agent_file> -mode auto")
        
        return True
    
    def _configure_api_keys(self) -> bool:
        """Configure API keys."""
        click.echo("\n🔑 API Key Configuration")
        click.echo("-" * 25)
        
        # DeepSeek API Key (required)
        deepseek_key = click.prompt(
            "DeepSeek API Key (required)",
            type=str,
            hide_input=True
        )
        
        # DeepSeek Endpoint
        deepseek_endpoint = click.prompt(
            "DeepSeek Endpoint",
            default="https://api.deepseek.com/v1",
            type=str
        )
        
        # Optional OpenAI keys
        configure_openai = click.confirm("Configure OpenAI/Azure OpenAI? (optional)")
        openai_key = ""
        openai_endpoint = ""
        
        if configure_openai:
            openai_key = click.prompt(
                "OpenAI API Key",
                default="",
                type=str,
                hide_input=True
            )
            openai_endpoint = click.prompt(
                "OpenAI Azure Endpoint (if using Azure)",
                default="",
                type=str
            )
        
        # Write .env file
        env_content = f"""# DeepDeliberate Framework Environment Configuration
# Generated by setup wizard

# API Configuration
DEEPSEEK_API_KEY={deepseek_key}
DEEPSEEK_ENDPOINT={deepseek_endpoint}
OPENAI_API_KEY={openai_key}
OPENAI_AZURE_ENDPOINT={openai_endpoint}

# Security Settings
ENABLE_REQUEST_SIGNING=true
MAX_INPUT_LENGTH=1000
ENABLE_AUDIT_LOGGING=true
LOG_RETENTION_DAYS=30
VERIFY_SSL_CERTIFICATES=true

# Rate Limiting
API_RATE_LIMIT_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT_SECONDS=30

# Agent Security
ENABLE_AGENT_VALIDATION=true
ALLOW_DANGEROUS_FUNCTIONS=false
SANDBOX_AGENT_EXECUTION=true

# Data Protection
ENABLE_DATA_REDACTION=true
ENCRYPT_LOGS=true
SECURE_SESSION_TOKENS=true
"""
        
        try:
            with open(self.env_path, 'w') as f:
                f.write(env_content)
            click.echo(f"✅ Environment configuration saved to {self.env_path}")
            return True
        except Exception as e:
            click.echo(f"❌ Failed to save environment configuration: {e}")
            return False
    
    def _configure_basic_settings(self) -> bool:
        """Configure basic framework settings."""
        click.echo("\n⚙️  Basic Configuration")
        click.echo("-" * 22)
        
        # Output directory
        output_dir = click.prompt(
            "Output directory for test results",
            default="test_results",
            type=str
        )
        
        # Default persona
        default_persona = click.prompt(
            "Default persona for testing",
            default="test_user",
            type=str
        )
        
        # Performance settings
        max_concurrent = click.prompt(
            "Maximum concurrent requests",
            default=5,
            type=int
        )
        
        # Create basic config if it doesn't exist
        if not self.config_path.exists():
            try:
                # Copy from existing config.json template
                import shutil
                shutil.copy('config.json', self.config_path)
                click.echo(f"✅ Configuration template copied to {self.config_path}")
            except Exception as e:
                click.echo(f"⚠️  Could not copy config template: {e}")
        
        return True
    
    def _test_configuration(self) -> bool:
        """Test the configuration."""
        click.echo("\n🧪 Testing Configuration")
        click.echo("-" * 23)
        
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(self.env_path)
            
            # Test API connectivity
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')
            if not deepseek_key:
                click.echo("❌ DeepSeek API key not found")
                return False
            
            click.echo("✅ API keys loaded successfully")
            
            # Test configuration loading
            if self.config_path.exists():
                config_manager = ConfigManager()
                config = config_manager.load_config(str(self.config_path))
                click.echo("✅ Configuration file loaded successfully")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Configuration test failed: {e}")
            return False


@click.command()
@click.option('--validate-only', is_flag=True, help='Only validate environment, do not run setup wizard')
@click.option('--verbose', is_flag=True, help='Show detailed validation information')
def main(validate_only: bool, verbose: bool):
    """
    DeepDeliberate Environment Setup and Validation Tool
    
    This tool validates your environment and guides you through initial setup.
    """
    click.echo("🚀 DeepDeliberate Environment Setup")
    click.echo("=" * 50)
    
    # Run validation
    validator = EnvironmentValidator()
    passed, issues, warnings = validator.run_validation()
    
    # Display results
    if verbose or not passed:
        click.echo(f"\n📊 Environment Information:")
        click.echo(f"   Python: {validator.python_version.major}.{validator.python_version.minor}.{validator.python_version.micro}")
        click.echo(f"   Platform: {validator.platform_info}")
    
    if warnings:
        click.echo(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            click.echo(f"   • {warning}")
    
    if issues:
        click.echo(f"\n❌ Issues ({len(issues)}):")
        for issue in issues:
            click.echo(f"   • {issue}")
        
        click.echo("\n🔧 Please fix these issues before proceeding:")
        click.echo("   1. Install missing dependencies: pip install -r requirements.txt")
        click.echo("   2. Configure required API keys in .env file")
        click.echo("   3. Ensure proper file permissions")
        
        sys.exit(1)
    
    click.echo("\n✅ Environment validation passed!")
    
    if validate_only:
        return
    
    # Run setup wizard
    wizard = SetupWizard()
    if not wizard.run_wizard():
        click.echo("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()