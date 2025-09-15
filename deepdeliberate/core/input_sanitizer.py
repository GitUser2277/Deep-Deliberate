"""
Input sanitization utilities for the DeepDeliberate framework.

This module provides security-focused input validation and sanitization
to prevent injection attacks and ensure safe query processing.
"""

import re
from typing import str, Optional
import bleach
# Import moved to function level to avoid circular dependency issues

__all__ = [
    "sanitize_query",
    "validate_query_safety",
    "sanitize_agent_response"
]


def sanitize_query(query: str, max_length: int = 8000) -> str:
    """
    Sanitize user query input for safe processing.
    
    Args:
        query: Raw query string from user
        max_length: Maximum allowed query length
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is invalid or unsafe
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    # Remove null bytes and control characters
    query = query.replace('\x00', '').strip()
    
    # Check length limits
    if len(query) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length} characters")
    
    if len(query) == 0:
        raise ValueError("Query cannot be empty")
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'data:text/html',            # Data URLs
        r'vbscript:',                 # VBScript
        r'on\w+\s*=',                 # Event handlers
    ]
    
    for pattern in dangerous_patterns:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE | re.DOTALL)
    
    # Use bleach for additional HTML sanitization
    query = bleach.clean(query, tags=[], attributes={}, strip=True)
    
    return query.strip()


def validate_query_safety(query: str) -> bool:
    """
    Validate that a query is safe for processing.
    
    Args:
        query: Query string to validate
        
    Returns:
        True if query is safe, False otherwise
    """
    try:
        sanitized = sanitize_query(query)
        return len(sanitized) > 0
    except ValueError:
        return False


def sanitize_agent_response(response: str, max_length: int = 50000) -> str:
    """
    Sanitize agent response for safe logging and display.
    
    Args:
        response: Raw agent response
        max_length: Maximum allowed response length
        
    Returns:
        Sanitized response string
    """
    if not isinstance(response, str):
        return str(response)
    
    # Use standardized truncation logic
    if len(response) > max_length:
        from .deepseek_display import get_display_renderer
        renderer = get_display_renderer()
        response = renderer._apply_content_truncation(response, max_length)
    
    # Remove null bytes and control characters except newlines and tabs
    response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)
    
    return response.strip()