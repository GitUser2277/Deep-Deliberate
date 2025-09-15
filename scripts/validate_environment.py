#!/usr/bin/env python3
"""
Environment validation script for DeepDeliberate framework.

This script validates Python version, dependencies, and API keys.
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Tuple

def validate_python_version() -> Tuple[bool, str]:
    """Validate Python version compatibility."""
    min_version = (3, 8)
    current = sys.version_info
    
    if current < min_version:
        return False, f"Python {min_version[0]}.{min_version[1]}+ required, found {current.major}.{current.minor}"
    
    return True, f"Python {current.major}.{current.minor}.{current.micro} ✅"

def validate_dependencies() -> Tuple[bool, List[str]]:
    """Validate that all required dependencies are installed."""
    required_packages = [
        'pydantic', 'click', 'httpx', 'tenacity',
        'dotenv', 'aiofiles', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def validate_api_keys() -> Tuple[bool, List[str]]:
    """Validate that required API keys are configured."""
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not available, skip loading
    
    required_keys = ['DEEPSEEK_API_KEY']
    missing = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)
    
    return len(missing) == 0, missing

def main():
    """Run environment validation."""
    print("🔍 DeepDeliberate Environment Validation")
    print("=" * 45)
    
    # Python version
    py_ok, py_msg = validate_python_version()
    print(f"Python Version: {py_msg}")
    
    # Dependencies
    deps_ok, missing_deps = validate_dependencies()
    if deps_ok:
        print("Dependencies: ✅ All required packages installed")
    else:
        print(f"Dependencies: ❌ Missing: {', '.join(missing_deps)}")
    
    # API Keys
    keys_ok, missing_keys = validate_api_keys()
    if keys_ok:
        print("API Keys: ✅ All required keys configured")
    else:
        print(f"API Keys: ❌ Missing: {', '.join(missing_keys)}")
    
    # Platform info
    print(f"Platform: {platform.platform()}")
    
    # Overall result
    all_ok = py_ok and deps_ok and keys_ok
    print("\n" + "=" * 45)
    if all_ok:
        print("✅ Environment validation PASSED")
        return 0
    else:
        print("❌ Environment validation FAILED")
        print("\nTo fix issues:")
        if not deps_ok:
            print("  pip install -r requirements.txt")
        if not keys_ok:
            print("  Set missing API keys in .env file")
        return 1

if __name__ == "__main__":
    sys.exit(main())