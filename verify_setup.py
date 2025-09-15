#!/usr/bin/env python3
"""
Verification script for DeepDeliberate framework setup.
Tests all core components and interfaces.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        import deepdeliberate
        print(f"[OK] deepdeliberate package (v{deepdeliberate.__version__})")
    except ImportError as e:
        print(f"[FAIL] deepdeliberate package: {e}")
        return False
    
    try:
        from deepdeliberate.core.models import (
            FrameworkConfig, Persona, ExecutionMode, 
            TestInteraction, SessionState
        )
        print("[OK] Core models")
    except ImportError as e:
        print(f"[FAIL] Core models: {e}")
        return False
    
    try:
        from deepdeliberate.core.interfaces import (
            AgentInterface, QueryGenerator, EvaluationEngine,
            SessionLogger, ConfigurationManager, FrameworkCore
        )
        print("[OK] Core interfaces")
    except ImportError as e:
        print(f"[FAIL] Core interfaces: {e}")
        return False
    
    try:
        from deepdeliberate.config.manager import ConfigManager
        print("[OK] Configuration manager")
    except ImportError as e:
        print(f"[FAIL] Configuration manager: {e}")
        return False
    
    try:
        from deepdeliberate.cli import main, display_query_approval
        print("[OK] CLI interface")
    except ImportError as e:
        print(f"[FAIL] CLI interface: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading and validation."""
    print("\nTesting configuration...")
    
    try:
        from deepdeliberate.config.manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config('config.json')
        
        print(f"[OK] Configuration loaded with {len(config.personas)} personas")
        
        # Test persona validation
        is_valid = config_manager.validate_personas(config.personas)
        print(f"[OK] Personas validation: {is_valid}")
        
        # Test persona retrieval
        personas = config_manager.list_personas()
        print(f"[OK] Available personas: {personas}")
        
        # Test individual persona retrieval
        test_persona = config_manager.get_persona('test_user')
        if test_persona:
            print(f"[OK] Retrieved persona: {test_persona.name}")
        else:
            print("[FAIL] Failed to retrieve test_user persona")
            return False
            
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False
    
    return True

def test_data_models():
    """Test data model creation and validation."""
    print("\nTesting data models...")
    
    try:
        from deepdeliberate.core.models import (
            Persona, FrameworkConfig, APISettings, 
            LoggingSettings, PerformanceSettings
        )
        
        # Test persona creation
        persona = Persona(
            name="test_persona",
            behavioral_patterns=["Test pattern"],
            tone_specifications={"formality": "casual"},
            scenario_templates=["Test scenario: {topic}"],
            evaluation_criteria={"accuracy": 0.5, "relevance": 0.5}
        )
        print(f"[OK] Persona model: {persona.name}")
        
        # Test configuration creation
        config = FrameworkConfig(
            personas=[persona],
            api_settings=APISettings(),
            logging_settings=LoggingSettings(),
            performance_settings=PerformanceSettings()
        )
        print(f"[OK] Framework config with {len(config.personas)} personas")
        
    except Exception as e:
        print(f"[FAIL] Data models test failed: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("\nTesting directory structure...")
    
    required_files = [
        "deepdeliberate/__init__.py",
        "deepdeliberate/cli.py",
        "deepdeliberate/core/__init__.py",
        "deepdeliberate/core/models.py",
        "deepdeliberate/core/interfaces.py",
        "deepdeliberate/config/__init__.py",
        "deepdeliberate/config/manager.py",
        "deepdeliberate/config/config_template.json",
        "deepdeliberate/config/.env.template",
        "config.json",
        "requirements.txt",
        "pyproject.toml",
        "test_agent.py"
    ]
    
    required_dirs = [
        "deepdeliberate",
        "deepdeliberate/core",
        "deepdeliberate/config",
        "test_results"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            return False
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"[OK] {dir_path}/")
        else:
            print(f"[MISSING] {dir_path}")
            return False
    
    return True

def main():
    """Run all verification tests."""
    print("DeepDeliberate Framework Setup Verification")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_data_models,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"[FAIL] {test.__name__} failed")
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Framework setup is complete.")
        return 0
    else:
        print("WARNING: Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())