#!/usr/bin/env python3
"""
DeepDeliberate Framework Deployment Script

This script automates the deployment and setup process for the DeepDeliberate framework,
including environment validation, dependency installation, and configuration setup.
"""

# Standard library imports
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "DeploymentManager",
    "main"
]


class DeploymentManager:
    """Manages the deployment process for DeepDeliberate framework."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.deployment_log = []
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log deployment messages."""
        log_entry = f"[{level}] {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        self.log("Checking Python version...")
        
        if self.python_version < (3, 8):
            self.log(f"Python {self.python_version.major}.{self.python_version.minor} detected. Python 3.8+ required.", "ERROR")
            return False
        
        self.log(f"Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} - OK")
        return True
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """
        Check system requirements and available tools.
        
        Returns:
            Dict mapping tool names to availability status
            
        Example:
            >>> deployer = DeploymentManager()
            >>> reqs = deployer.check_system_requirements()
            >>> print(reqs["python"])  # True if Python available
        """
        self.log("Checking system requirements...")
        
        requirements = {
            "python": self.check_python_version(),
            "pip": shutil.which("pip") is not None,
            "git": shutil.which("git") is not None,
        }
        
        # Check for UV (preferred package manager)
        requirements["uv"] = shutil.which("uv") is not None
        
        # Platform-specific checks
        if self.platform == "windows":
            requirements["powershell"] = shutil.which("powershell") is not None
        
        for tool, available in requirements.items():
            status = "OK" if available else "MISSING"
            level = "INFO" if available else "WARNING"
            self.log(f"{tool}: {status}", level)
        
        return requirements
    
    def install_uv_if_missing(self) -> bool:
        """Install UV package manager if not available."""
        if shutil.which("uv"):
            self.log("UV package manager already available")
            return True
        
        self.log("Installing UV package manager...")
        
        try:
            if self.platform == "windows":
                # Install UV on Windows using pip
                subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
            else:
                # Install UV on Unix-like systems
                subprocess.run(["pip", "install", "uv"], check=True)
            
            self.log("UV package manager installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install UV: {e}", "ERROR")
            return False
    
    def create_virtual_environment(self, project_dir: Path) -> bool:
        """Create virtual environment using UV or venv."""
        self.log("Creating virtual environment...")
        
        venv_path = project_dir / ".venv"
        
        try:
            if shutil.which("uv"):
                # Use UV to create virtual environment
                subprocess.run(["uv", "venv", str(venv_path)], cwd=project_dir, check=True)
            else:
                # Fallback to standard venv
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            
            self.log(f"Virtual environment created at: {venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to create virtual environment: {e}", "ERROR")
            return False
    
    def install_dependencies(self, project_dir: Path) -> bool:
        """Install project dependencies."""
        self.log("Installing dependencies...")
        
        requirements_file = project_dir / "requirements.txt"
        if not requirements_file.exists():
            self.log("requirements.txt not found", "ERROR")
            return False
        
        try:
            venv_path = project_dir / ".venv"
            
            if shutil.which("uv") and venv_path.exists():
                # Use UV for dependency installation
                subprocess.run([
                    "uv", "pip", "install", "-r", str(requirements_file)
                ], cwd=project_dir, check=True)
            else:
                # Fallback to pip
                if self.platform == "windows":
                    pip_path = venv_path / "Scripts" / "pip.exe"
                else:
                    pip_path = venv_path / "bin" / "pip"
                
                if pip_path.exists():
                    subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
                else:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            
            self.log("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install dependencies: {e}", "ERROR")
            return False
    
    def setup_environment_file(self, project_dir: Path) -> bool:
        """Set up .env file from template."""
        self.log("Setting up environment configuration...")
        
        env_template = project_dir / ".env.template"
        env_file = project_dir / ".env"
        
        if not env_template.exists():
            self.log(".env.template not found", "WARNING")
            return True  # Not critical for deployment
        
        if env_file.exists():
            self.log(".env file already exists, skipping template copy")
            return True
        
        try:
            shutil.copy2(env_template, env_file)
            self.log(f"Environment template copied to: {env_file}")
            self.log("Please edit .env file with your API keys and configuration", "WARNING")
            return True
            
        except Exception as e:
            self.log(f"Failed to copy environment template: {e}", "ERROR")
            return False
    
    def validate_configuration(self, project_dir: Path) -> bool:
        """Validate framework configuration files."""
        self.log("Validating configuration files...")
        
        config_file = project_dir / "config.json"
        if not config_file.exists():
            self.log("config.json not found - will be created on first run", "WARNING")
            return True
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Basic validation
            required_sections = ["personas", "api_settings", "logging_settings"]
            for section in required_sections:
                if section not in config:
                    self.log(f"Missing required configuration section: {section}", "ERROR")
                    return False
            
            self.log("Configuration file validation passed")
            return True
            
        except json.JSONDecodeError as e:
            self.log(f"Invalid JSON in config.json: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Configuration validation error: {e}", "ERROR")
            return False
    
    def create_deployment_summary(self, project_dir: Path) -> None:
        """Create deployment summary and next steps."""
        summary_file = project_dir / "DEPLOYMENT_SUMMARY.md"
        
        summary_content = f"""# DeepDeliberate Deployment Summary

## Deployment Information
- Date: {subprocess.run(['date'], capture_output=True, text=True, shell=True).stdout.strip()}
- Platform: {platform.system()} {platform.release()}
- Python Version: {sys.version}

## Deployment Log
```
{chr(10).join(self.deployment_log)}
```

## Next Steps

1. **Configure API Keys**
   - Edit `.env` file with your DeepSeek API key
   - Set DEEPSEEK_API_KEY environment variable

2. **Validate Installation**
   ```bash
   python scripts/health_check.py
   ```

3. **Run Quick Test**
   ```bash
   python deepdeliberate.py -file examples/agents/customer_service_agent.py -mode auto -count 5
   ```

4. **Review Configuration**
   - Check `config.json` for persona and evaluation settings
   - Customize personas for your specific use case

## Troubleshooting

If you encounter issues:
1. Run the health check script: `python scripts/health_check.py`
2. Check the deployment log above for specific errors
3. Refer to `docs/INSTALLATION.md` for detailed setup instructions
4. Check `examples/docs/troubleshooting_guide.md` for common issues

## Support

For additional support:
- Review documentation in `docs/` directory
- Check example configurations in `examples/configs/`
- Run diagnostics: `python scripts/diagnostics.py`
"""
        
        try:
            with open(summary_file, 'w') as f:
                f.write(summary_content)
            self.log(f"Deployment summary created: {summary_file}")
        except Exception as e:
            self.log(f"Failed to create deployment summary: {e}", "WARNING")
    
    def deploy(self, project_dir: Optional[Path] = None) -> bool:
        """
        Execute complete deployment process.
        
        Args:
            project_dir: Target directory for deployment. Defaults to current directory.
            
        Returns:
            True if deployment successful, False otherwise
            
        Raises:
            DeploymentError: If critical deployment steps fail
            
        Example:
            >>> deployer = DeploymentManager()
            >>> success = deployer.deploy(Path("/path/to/project"))
            >>> if success:
            ...     print("Deployment completed successfully")
        """
        if project_dir is None:
            project_dir = Path.cwd()
        
        self.log("Starting DeepDeliberate framework deployment...")
        self.log(f"Project directory: {project_dir}")
        
        # Step 1: Check system requirements
        requirements = self.check_system_requirements()
        if not requirements["python"]:
            return False
        
        # Step 2: Install UV if missing (optional but recommended)
        if not requirements["uv"]:
            self.install_uv_if_missing()
        
        # Step 3: Create virtual environment
        if not self.create_virtual_environment(project_dir):
            return False
        
        # Step 4: Install dependencies
        if not self.install_dependencies(project_dir):
            return False
        
        # Step 5: Set up environment configuration
        if not self.setup_environment_file(project_dir):
            return False
        
        # Step 6: Validate configuration
        if not self.validate_configuration(project_dir):
            return False
        
        # Step 7: Create deployment summary
        self.create_deployment_summary(project_dir)
        
        self.log("Deployment completed successfully!")
        self.log("Please review DEPLOYMENT_SUMMARY.md for next steps")
        
        return True


def main():
    """Main deployment script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepDeliberate Framework Deployment Script")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory path (default: current directory)"
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    
    args = parser.parse_args()
    
    deployer = DeploymentManager()
    
    try:
        success = deployer.deploy(args.project_dir)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Deployment failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()