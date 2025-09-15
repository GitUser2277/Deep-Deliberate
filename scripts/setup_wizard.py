#!/usr/bin/env python3
"""
Setup wizard for DeepDeliberate framework.

Interactive wizard to guide first-time users through configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with user input."""
    print("\n🔑 API Key Configuration")
    print("-" * 25)
    
    # Get DeepSeek API key
    deepseek_key = input("Enter DeepSeek API Key: ").strip()
    if not deepseek_key:
        print("❌ DeepSeek API key is required")
        return False
    
    # Get endpoint
    endpoint = input("DeepSeek Endpoint (press Enter for default): ").strip()
    if not endpoint:
        endpoint = "https://api.deepseek.com/v1"
    
    # Create .env content
    env_content = f"""# DeepDeliberate Framework Environment Configuration
DEEPSEEK_API_KEY={deepseek_key}
DEEPSEEK_ENDPOINT={endpoint}

# Security Settings
ENABLE_REQUEST_SIGNING=true
MAX_INPUT_LENGTH=1000
VERIFY_SSL_CERTIFICATES=true

# Rate Limiting
API_RATE_LIMIT_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT_SECONDS=30

# Agent Security
ENABLE_AGENT_VALIDATION=true
ALLOW_DANGEROUS_FUNCTIONS=false
SANDBOX_AGENT_EXECUTION=true
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def verify_setup():
    """Verify the setup is working."""
    print("\n🧪 Verifying Setup")
    print("-" * 18)
    
    try:
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check API key
        if os.getenv('DEEPSEEK_API_KEY'):
            print("✅ API key loaded")
        else:
            print("❌ API key not found")
            return False
        
        # Check config file
        if Path('config.json').exists():
            print("✅ Configuration file found")
        else:
            print("⚠️  config.json not found (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

def main():
    """Run the setup wizard."""
    print("🧙 DeepDeliberate Setup Wizard")
    print("=" * 35)
    
    # Check if already configured
    if Path('.env').exists():
        response = input("\n.env file already exists. Reconfigure? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return 0
    
    # Create environment file
    if not create_env_file():
        return 1
    
    # Verify setup
    if not verify_setup():
        return 1
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run:")
    print("  python deepdeliberate.py -file <agent_file> -mode auto")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())