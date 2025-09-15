#!/usr/bin/env python3
"""
Diagnostics script for DeepDeliberate framework.

Provides comprehensive system diagnostics and troubleshooting information.
"""

import os
import sys
import platform
import json
from pathlib import Path
from typing import Dict, Any

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "working_directory": os.getcwd()
    }

def get_environment_info() -> Dict[str, Any]:
    """Get environment configuration info."""
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not available, skip loading
    
    env_vars = [
        'DEEPSEEK_API_KEY', 'DEEPSEEK_ENDPOINT',
        'API_RATE_LIMIT_PER_MINUTE', 'MAX_CONCURRENT_REQUESTS',
        'REQUEST_TIMEOUT_SECONDS'
    ]
    
    env_info = {}
    for var in env_vars:
        value = os.getenv(var)
        if var.endswith('_KEY'):
            # Mask API keys for security
            env_info[var] = "***CONFIGURED***" if value else "NOT_SET"
        else:
            env_info[var] = value or "NOT_SET"
    
    return env_info

def get_file_info() -> Dict[str, Any]:
    """Get file system information."""
    files_to_check = [
        'config.json', '.env', 'requirements.txt',
        'deepdeliberate.py', 'deepdeliberate/__init__.py'
    ]
    
    file_info = {}
    for file_path in files_to_check:
        path = Path(file_path)
        file_info[file_path] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "readable": os.access(path, os.R_OK) if path.exists() else False
        }
    
    return file_info

def check_dependencies() -> Dict[str, str]:
    """Check dependency status."""
    deps = [
        'pydantic', 'click', 'httpx', 'tenacity',
        'dotenv', 'aiofiles', 'pandas', 'tiktoken'
    ]
    
    dep_status = {}
    for dep in deps:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            dep_status[dep] = f"✅ {version}"
        except ImportError:
            dep_status[dep] = "❌ NOT_INSTALLED"
    
    return dep_status

def check_permissions() -> Dict[str, bool]:
    """Check file permissions."""
    dirs_to_check = ['.', 'test_results']
    
    permissions = {}
    for dir_path in dirs_to_check:
        try:
            os.makedirs(dir_path, exist_ok=True)
            test_file = os.path.join(dir_path, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            permissions[dir_path] = True
        except (OSError, PermissionError):
            permissions[dir_path] = False
    
    return permissions

def main():
    """Run diagnostics and display results."""
    print("🔍 DeepDeliberate Framework Diagnostics")
    print("=" * 45)
    
    # Collect all diagnostic information
    diagnostics = {
        "system_info": get_system_info(),
        "environment": get_environment_info(),
        "files": get_file_info(),
        "dependencies": check_dependencies(),
        "permissions": check_permissions()
    }
    
    # Display results
    print("\n📊 System Information:")
    for key, value in diagnostics["system_info"].items():
        print(f"  {key}: {value}")
    
    print("\n🔧 Environment Variables:")
    for key, value in diagnostics["environment"].items():
        print(f"  {key}: {value}")
    
    print("\n📁 File Status:")
    for file_path, info in diagnostics["files"].items():
        status = "✅" if info["exists"] else "❌"
        print(f"  {file_path}: {status} ({info['size']} bytes)")
    
    print("\n📦 Dependencies:")
    for dep, status in diagnostics["dependencies"].items():
        print(f"  {dep}: {status}")
    
    print("\n🔒 Permissions:")
    for dir_path, can_write in diagnostics["permissions"].items():
        status = "✅" if can_write else "❌"
        print(f"  {dir_path}: {status}")
    
    # Save detailed report
    report_file = "diagnostics_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    # Summary
    issues = []
    framework_files = [f for f in diagnostics["files"].keys() if "deepdeliberate" in f]
    if framework_files and not all(diagnostics["files"][f]["exists"] for f in framework_files):
        issues.append("Framework files missing")
    
    if "❌" in str(diagnostics["dependencies"]):
        issues.append("Missing dependencies")
    
    if not all(diagnostics["permissions"].values()):
        issues.append("Permission issues")
    
    if diagnostics["environment"]["DEEPSEEK_API_KEY"] == "NOT_SET":
        issues.append("API key not configured")
    
    print("\n" + "=" * 45)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return 1
    else:
        print("✅ No issues detected")
        return 0

if __name__ == "__main__":
    sys.exit(main())