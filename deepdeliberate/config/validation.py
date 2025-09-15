"""
Configuration validation and management for DeepDeliberate framework.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pathlib import Path


class APIConfig(BaseModel):
    """API configuration with validation."""
    
    deepseek_api_key: str = Field(..., min_length=10)
    deepseek_endpoint: str = Field(..., regex=r'^https?://')
    openai_api_key: Optional[str] = Field(None, min_length=10)
    openai_endpoint: Optional[str] = Field(None, regex=r'^https?://')
    
    @validator('deepseek_api_key', 'openai_api_key')
    def validate_api_key(cls, v):
        if v and len(v) < 10:
            raise ValueError('API key must be at least 10 characters')
        return v


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    enable_request_signing: bool = True
    max_input_length: int = Field(1000, ge=100, le=10000)
    enable_audit_logging: bool = True
    log_retention_days: int = Field(30, ge=1, le=365)
    verify_ssl_certificates: bool = True


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    
    api_rate_limit_per_minute: int = Field(60, ge=1, le=1000)
    max_concurrent_requests: int = Field(5, ge=1, le=50)
    request_timeout_seconds: int = Field(30, ge=5, le=300)


class FrameworkConfig(BaseModel):
    """Complete framework configuration."""
    
    api: APIConfig
    security: SecurityConfig
    rate_limit: RateLimitConfig
    
    @classmethod
    def from_env(cls) -> 'FrameworkConfig':
        """Load configuration from environment variables."""
        return cls(
            api=APIConfig(
                deepseek_api_key=os.getenv('DEEPSEEK_API_KEY', ''),
                deepseek_endpoint=os.getenv('DEEPSEEK_ENDPOINT', ''),
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                openai_endpoint=os.getenv('OPENAI_AZURE_ENDPOINT')
            ),
            security=SecurityConfig(
                enable_request_signing=os.getenv('ENABLE_REQUEST_SIGNING', 'true').lower() == 'true',
                max_input_length=int(os.getenv('MAX_INPUT_LENGTH', '1000')),
                enable_audit_logging=os.getenv('ENABLE_AUDIT_LOGGING', 'true').lower() == 'true',
                log_retention_days=int(os.getenv('LOG_RETENTION_DAYS', '30')),
                verify_ssl_certificates=os.getenv('VERIFY_SSL_CERTIFICATES', 'true').lower() == 'true'
            ),
            rate_limit=RateLimitConfig(
                api_rate_limit_per_minute=int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '60')),
                max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '5')),
                request_timeout_seconds=int(os.getenv('REQUEST_TIMEOUT_SECONDS', '30'))
            )
        )
    
    def validate_environment(self) -> List[str]:
        """Validate environment setup and return any issues."""
        issues = []
        
        if not self.api.deepseek_api_key:
            issues.append("DEEPSEEK_API_KEY is required")
        
        if not self.api.deepseek_endpoint:
            issues.append("DEEPSEEK_ENDPOINT is required")
        
        return issues


def load_and_validate_config() -> FrameworkConfig:
    """Load and validate configuration, raising errors for invalid setup."""
    try:
        config = FrameworkConfig.from_env()
        issues = config.validate_environment()
        
        if issues:
            raise ValueError(f"Configuration issues: {', '.join(issues)}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")