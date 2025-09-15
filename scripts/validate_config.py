#!/usr/bin/env python3
"""
Configuration validation script for DeepDeliberate framework.

Validates configuration files and provides detailed error reporting.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def validate_persona(persona: Dict[str, Any], persona_name: str) -> List[str]:
    """Validate a single persona configuration."""
    errors = []
    
    # Required fields
    required_fields = ['name', 'behavioral_patterns', 'tone_specifications', 
                      'scenario_templates', 'evaluation_criteria']
    
    for field in required_fields:
        if field not in persona:
            errors.append(f"Persona '{persona_name}': Missing required field '{field}'")
    
    # Validate evaluation criteria
    if 'evaluation_criteria' in persona:
        criteria = persona['evaluation_criteria']
        if not isinstance(criteria, dict):
            errors.append(f"Persona '{persona_name}': evaluation_criteria must be a dictionary")
        else:
            # Check weights sum to 1.0
            total_weight = sum(criteria.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Persona '{persona_name}': evaluation_criteria weights sum to {total_weight}, should sum to 1.0")
            
            # Check for required criteria
            required_criteria = ['accuracy', 'relevance', 'completion', 'safety']
            for criterion in required_criteria:
                if criterion not in criteria:
                    errors.append(f"Persona '{persona_name}': Missing evaluation criterion '{criterion}'")
    
    # Validate behavioral patterns
    if 'behavioral_patterns' in persona:
        if not isinstance(persona['behavioral_patterns'], list):
            errors.append(f"Persona '{persona_name}': behavioral_patterns must be a list")
        elif len(persona['behavioral_patterns']) == 0:
            errors.append(f"Persona '{persona_name}': behavioral_patterns cannot be empty")
    
    return errors

def validate_api_settings(api_settings: Dict[str, Any]) -> List[str]:
    """Validate API settings."""
    errors = []
    
    required_fields = ['deepseek_endpoint', 'timeout_seconds', 'retry_attempts', 'rate_limit_per_minute']
    
    for field in required_fields:
        if field not in api_settings:
            errors.append(f"API settings: Missing required field '{field}'")
    
    # Validate numeric fields
    numeric_fields = {
        'timeout_seconds': (1, 300),
        'retry_attempts': (1, 10),
        'rate_limit_per_minute': (1, 1000)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in api_settings:
            try:
                value = int(str(api_settings[field]).replace('${', '').replace('}', ''))
                if not (min_val <= value <= max_val):
                    errors.append(f"API settings: {field} should be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                # Skip validation for environment variables
                pass
    
    return errors

def validate_config_file(config_path: str) -> List[str]:
    """Validate the entire configuration file."""
    errors = []
    
    # Check if file exists
    if not Path(config_path).exists():
        return [f"Configuration file not found: {config_path}"]
    
    # Load and parse JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in configuration file: {e}"]
    except Exception as e:
        return [f"Error reading configuration file: {e}"]
    
    # Validate top-level structure
    required_sections = ['personas', 'api_settings', 'logging_settings', 'performance_settings']
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate personas
    if 'personas' in config:
        if not isinstance(config['personas'], list):
            errors.append("'personas' must be a list")
        elif len(config['personas']) == 0:
            errors.append("At least one persona must be defined")
        else:
            for i, persona in enumerate(config['personas']):
                persona_name = persona.get('name', f'persona_{i}')
                errors.extend(validate_persona(persona, persona_name))
    
    # Validate API settings
    if 'api_settings' in config:
        errors.extend(validate_api_settings(config['api_settings']))
    
    # Validate logging settings
    if 'logging_settings' in config:
        logging_settings = config['logging_settings']
        if 'output_directory' not in logging_settings:
            errors.append("Logging settings: Missing 'output_directory'")
    
    return errors

def main():
    """Run configuration validation."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    
    print(f"🔍 Validating configuration: {config_file}")
    print("=" * 50)
    
    errors = validate_config_file(config_file)
    
    if errors:
        print(f"❌ Found {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        
        print("\n🔧 Suggestions:")
        print("  • Check the config.json syntax with a JSON validator")
        print("  • Ensure all required fields are present")
        print("  • Verify evaluation criteria weights sum to 1.0")
        print("  • Use the provided config template as reference")
        
        return 1
    else:
        print("✅ Configuration validation passed!")
        print("  • All required sections present")
        print("  • All personas properly configured")
        print("  • API settings valid")
        return 0

if __name__ == "__main__":
    sys.exit(main())