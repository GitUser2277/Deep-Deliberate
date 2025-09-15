"""
DeepSeek R1 API client for Azure AI Inference SDK integration.

This module provides a robust client for interacting with DeepSeek R1 on Azure,
including retry mechanisms, rate limiting, and error handling.
"""

# Standard library imports
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
import hashlib
import json

# Third-party imports for environment loading
from dotenv import load_dotenv

# Third-party imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

__all__ = [
    "DeepSeekConfig",
    "DeepSeekClient", 
    "DeepSeekAPIError",
    "APIResponse",
    "APIMetrics",
    "RateLimiter",
    "Message",
]

logger = logging.getLogger(__name__)

# Constants
DEFAULT_HEALTH_CHECK_TOKENS = 10
RATE_LIMIT_CLEANUP_INTERVAL_SECONDS = 30
MIN_API_KEY_LENGTH = 10


class Message(TypedDict):
    """Type definition for chat messages."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek R1 API client."""
    endpoint: str
    api_key: str
    model_name: str = "DeepSeek-R1"
    max_tokens: int = 3000
    timeout_seconds: int = 60
    retry_attempts: int = 3
    rate_limit_per_minute: int = 60
    exponential_backoff_base: float = 2.0
    exponential_backoff_max: float = 60.0
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not self.endpoint or not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be a valid HTTP/HTTPS URL")
        
        if not self.api_key or len(self.api_key) < MIN_API_KEY_LENGTH:
            raise ValueError(f"api_key must be provided and at least {MIN_API_KEY_LENGTH} characters")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        elif self.max_tokens > 32000:
            raise ValueError(f"max_tokens exceeds limit of 32000, got {self.max_tokens}")
        
        if self.timeout_seconds <= 0 or self.timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        
        if self.retry_attempts < 0 or self.retry_attempts > 10:
            raise ValueError("retry_attempts must be between 0 and 10")
        
        if self.rate_limit_per_minute <= 0 or self.rate_limit_per_minute > 1000:
            raise ValueError("rate_limit_per_minute must be between 1 and 1000")
        
        if self.exponential_backoff_base <= 1.0 or self.exponential_backoff_base > 10.0:
            raise ValueError("exponential_backoff_base must be between 1.0 and 10.0")
        
        if self.exponential_backoff_max <= 0 or self.exponential_backoff_max > 300:
            raise ValueError("exponential_backoff_max must be between 1 and 300 seconds")
    
    @classmethod
    def from_environment(cls, env_file: Optional[str] = None) -> 'DeepSeekConfig':
        """
        Create configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in the DeepDeliberate project root.
            
        Returns:
            DeepSeekConfig instance with values from environment
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Load environment variables from .env file using cross-platform path handling
        if env_file is None:
            # Get the project root directory (where .env should be located)
            # This module is in deepdeliberate/core/, so we need to go up 2 levels
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from core/ to deepdeliberate/ to project root
            env_path = project_root / '.env'
        else:
            env_path = Path(env_file)
        
        # Load the .env file if found
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path.resolve()}")
        else:
            logger.warning(f"No .env file found at {env_path.resolve()}, using system environment variables only")
        
        def clean_env_value(value: str) -> str:
            """Clean environment variable value by removing comments and whitespace."""
            if value is None:
                return None
            # Split on # to remove comments, then strip whitespace
            return value.split('#')[0].strip()
        
        def get_int_env(env_var: str, fallback_var: str = None, default: str = '0') -> int:
            """Get integer value from environment variable, handling comments."""
            value = os.getenv(env_var)
            if value is None and fallback_var:
                value = os.getenv(fallback_var)
            if value is None:
                value = default
            
            cleaned_value = clean_env_value(value)
            try:
                return int(cleaned_value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid integer value for {env_var}: '{value}', using default: {default}")
                return int(default)
        
        required_vars = {
            'DEEPSEEK_ENDPOINT': 'endpoint',
            'DEEPSEEK_API_KEY': 'api_key',
        }
        
        config_dict = {}
        missing_vars = []
        
        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
            else:
                # Clean the value to remove comments
                config_dict[config_key] = clean_env_value(value)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set them in your .env file or system environment."
            )
        
        # Optional configuration with defaults from environment or fallback values
        config_dict['model_name'] = clean_env_value(os.getenv('DEEPSEEK_MODEL_NAME', 'DeepSeek-R1'))
        config_dict['max_tokens'] = get_int_env('DEEPSEEK_MAX_TOKENS', default='3000')
        config_dict['timeout_seconds'] = get_int_env('REQUEST_TIMEOUT_SECONDS', 'DEEPSEEK_TIMEOUT', '30')
        config_dict['retry_attempts'] = get_int_env('DEEPSEEK_RETRY_ATTEMPTS', default='3')
        config_dict['rate_limit_per_minute'] = get_int_env('API_RATE_LIMIT_PER_MINUTE', 'DEEPSEEK_RATE_LIMIT', '60')
        
        return cls(**config_dict)


@dataclass
class APIResponse:
    """Structured response from DeepSeek API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float
    timestamp: datetime


@dataclass
class APIMetrics:
    """Metrics for API usage tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    average_response_time: float = 0.0
    rate_limit_hits: int = 0
    last_request_time: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    response: APIResponse
    timestamp: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds


class ResponseCache:
    """Thread-safe response cache with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        
    def _generate_cache_key(self, messages: List[Message], **kwargs) -> str:
        """Generate cache key from messages and parameters."""
        # Create a deterministic hash of the request
        cache_data = {
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ['timestamp']}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, messages: List[Message], **kwargs) -> Optional[APIResponse]:
        """Get cached response if available and not expired."""
        cache_key = self._generate_cache_key(messages, **kwargs)
        
        with self.lock:
            if cache_key not in self.cache:
                return None
            
            entry = self.cache[cache_key]
            if entry.is_expired():
                # Remove expired entry
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                return None
            
            # Update access time for LRU
            self.access_times[cache_key] = datetime.now()
            return entry.response
    
    def put(self, messages: List[Message], response: APIResponse, ttl: Optional[int] = None, **kwargs) -> None:
        """Store response in cache with TTL."""
        cache_key = self._generate_cache_key(messages, **kwargs)
        ttl = ttl or self.default_ttl
        
        with self.lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            entry = CacheEntry(
                response=response,
                timestamp=datetime.now(),
                ttl_seconds=ttl
            )
            
            self.cache[cache_key] = entry
            self.access_times[cache_key] = datetime.now()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.access_times:
            return
        
        # Remove 10% of entries (LRU)
        num_to_remove = max(1, len(self.cache) // 10)
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        for key in sorted_keys[:num_to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count of removed entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
            return {
                "total_entries": len(self.cache),
                "expired_entries": expired_count,
                "active_entries": len(self.cache) - expired_count,
                "max_size": self.max_size,
                "utilization_percent": (len(self.cache) / self.max_size) * 100,
                "default_ttl": self.default_ttl
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


class ConnectionPool:
    """Connection pool for managing Azure AI Inference clients."""
    
    def __init__(self, config: DeepSeekConfig, pool_size: int = 5):
        self.config = config
        self.pool_size = pool_size
        self.pool: Queue = Queue(maxsize=pool_size)
        self.created_connections = 0
        self.lock = threading.Lock()
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()
        
        # Pre-populate pool
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool with clients."""
        for _ in range(self.pool_size):
            try:
                client = self._create_client()
                self.pool.put(client, block=False)
                self.created_connections += 1
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    def _create_client(self) -> ChatCompletionsClient:
        """Create a new Azure AI Inference client."""
        return ChatCompletionsClient(
            endpoint=self.config.endpoint,
            credential=AzureKeyCredential(self.config.api_key)
        )
    
    def get_client(self, timeout: float = 5.0) -> ChatCompletionsClient:
        """Get a client from the pool."""
        try:
            # Try to get from pool first
            client = self.pool.get(timeout=timeout)
            
            # Perform health check if needed
            if self._should_health_check():
                if not self._validate_client_health(client):
                    # Create new client if health check fails
                    client = self._create_client()
            
            return client
            
        except:
            # Create new client if pool is empty
            with self.lock:
                if self.created_connections < self.pool_size * 2:  # Allow some overflow
                    self.created_connections += 1
                    return self._create_client()
                else:
                    raise DeepSeekAPIError(
                        "Connection pool exhausted and max connections reached",
                        error_type="resource_exhaustion"
                    )
    
    def return_client(self, client: ChatCompletionsClient) -> None:
        """Return a client to the pool."""
        try:
            self.pool.put(client, block=False)
        except:
            # Pool is full, client will be garbage collected
            pass
    
    def _should_health_check(self) -> bool:
        """Check if health check is needed."""
        return (datetime.now() - self.last_health_check).total_seconds() > self.health_check_interval
    
    def _validate_client_health(self, client: ChatCompletionsClient) -> bool:
        """Validate client health with a minimal test."""
        try:
            # This is a basic validation - in a real implementation,
            # you might want to make a minimal API call
            return client is not None
        except Exception:
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status."""
        return {
            "pool_size": self.pool_size,
            "available_connections": self.pool.qsize(),
            "created_connections": self.created_connections,
            "utilization_percent": ((self.pool_size - self.pool.qsize()) / self.pool_size) * 100,
            "last_health_check": self.last_health_check.isoformat(),
            "health_check_interval": self.health_check_interval
        }
    
    def close(self) -> None:
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                client = self.pool.get(block=False)
                # Azure AI Inference clients don't have explicit close method
                # They will be garbage collected
            except:
                break


class RateLimiter:
    """Thread-safe rate limiter for API requests with sliding window."""
    
    def __init__(self, max_requests_per_minute: int):
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        if max_requests_per_minute > 10000:  # Reasonable upper limit
            raise ValueError("max_requests_per_minute exceeds reasonable limit of 10,000")
            
        self.max_requests = max_requests_per_minute
        self.requests = deque()  # Use deque for efficient operations
        self.lock = threading.RLock()  # Use RLock for reentrant locking
        self.condition = threading.Condition(self.lock)  # For efficient thread coordination
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(seconds=RATE_LIMIT_CLEANUP_INTERVAL_SECONDS)
        
        # Thread-safe statistics
        self._total_requests = 0
        self._total_waits = 0
        self._total_wait_time = 0.0
        self._max_queue_size = 0
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.condition:
            now = datetime.now()
            
            # Perform cleanup of old requests
            self._cleanup_old_requests(now)
            
            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                # Calculate wait time based on oldest request
                if not self.requests:  # Safety check
                    logger.warning("Rate limit check failed: no requests in queue despite count >= max_requests")
                    return
                
                oldest_request = self.requests[0]
                wait_time = 60 - (now - oldest_request).total_seconds()
                
                if wait_time > 0:
                    # Cap wait time to prevent excessive delays
                    wait_time = min(wait_time, 65.0)  # Max 65 seconds
                    
                    logger.warning(f"🛡️ LOCAL RATE LIMIT: Blocking request to prevent API overload. "
                                 f"Waiting {wait_time:.2f}s (Local limit: {self.max_requests}/min)")
                    
                    # Update wait statistics
                    self._total_waits += 1
                    self._total_wait_time += wait_time
                    
                    # Use condition.wait() for more efficient thread coordination
                    # This allows other threads to check status while we wait
                    try:
                        self.condition.wait(timeout=wait_time)
                    except KeyboardInterrupt:
                        logger.info("Rate limit wait interrupted by user")
                        raise
                    
                    # Clean up again after waiting
                    self._cleanup_old_requests(datetime.now())
            
            # Add current request timestamp
            self.requests.append(now)
            
            # Update statistics
            self._total_requests += 1
            self._max_queue_size = max(self._max_queue_size, len(self.requests))
    
    def _cleanup_old_requests(self, now: datetime) -> None:
        """Remove requests older than 1 minute. Thread-safe internal method."""
        cutoff = now - timedelta(minutes=1)
        
        # Use deque's efficient popleft for removing old requests
        removed_count = 0
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
            removed_count += 1
        
        # Log cleanup if significant number of requests were removed
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old rate limit entries")
            # Notify waiting threads that slots may be available
            self.condition.notify_all()
        
        # Update last cleanup time
        self._last_cleanup = now
    
    def get_current_request_count(self) -> int:
        """Get current number of requests in the sliding window."""
        with self.condition:
            self._cleanup_old_requests(datetime.now())
            return len(self.requests)
    
    def get_time_until_next_request(self) -> float:
        """Get time in seconds until next request can be made."""
        with self.condition:
            now = datetime.now()
            self._cleanup_old_requests(now)
            
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Time until oldest request expires
            oldest_request = self.requests[0]
            return max(0.0, 60 - (now - oldest_request).total_seconds())
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without waiting (non-blocking)."""
        with self.condition:
            now = datetime.now()
            self._cleanup_old_requests(now)
            return len(self.requests) < self.max_requests
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get detailed rate limit status information."""
        with self.condition:
            now = datetime.now()
            self._cleanup_old_requests(now)
            
            return {
                "current_requests": len(self.requests),
                "max_requests": self.max_requests,
                "requests_remaining": max(0, self.max_requests - len(self.requests)),
                "utilization_percent": (len(self.requests) / self.max_requests) * 100,
                "time_until_next_slot": self.get_time_until_next_request(),
                "can_make_request": len(self.requests) < self.max_requests,
                "oldest_request_age": (now - self.requests[0]).total_seconds() if self.requests else 0.0,
                "total_requests_processed": self._total_requests,
                "total_waits": self._total_waits,
                "total_wait_time": round(self._total_wait_time, 2),
                "average_wait_time": round(self._total_wait_time / max(self._total_waits, 1), 2),
                "max_queue_size_reached": self._max_queue_size,
                "wait_percentage": round((self._total_waits / max(self._total_requests, 1)) * 100, 1)
            }
    
    def reset(self) -> None:
        """Reset the rate limiter, clearing all request history and statistics."""
        with self.condition:
            self.requests.clear()
            self._last_cleanup = datetime.now()
            
            # Reset statistics
            self._total_requests = 0
            self._total_waits = 0
            self._total_wait_time = 0.0
            self._max_queue_size = 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.reset()
        return False
    
    def wait_with_backoff(self, attempt: int = 1) -> None:
        """
        Wait with exponential backoff for rate limiting.
        
        This method provides more sophisticated waiting behavior for scenarios
        where multiple consecutive rate limit hits occur.
        
        Args:
            attempt: The attempt number (1-based) for exponential backoff calculation
        """
        with self.lock:
            base_wait = self.get_time_until_next_request()
            
            if base_wait <= 0:
                return
            
            # Apply exponential backoff for repeated attempts
            backoff_multiplier = min(2 ** (attempt - 1), 8)  # Cap at 8x
            total_wait = base_wait * backoff_multiplier
            
            # Cap total wait time to prevent excessive delays
            total_wait = min(total_wait, 120.0)  # Max 2 minutes
            
            logger.info(f"Rate limit backoff: attempt {attempt}, waiting {total_wait:.2f} seconds")
            
            try:
                time.sleep(total_wait)
            except KeyboardInterrupt:
                logger.info("Rate limit backoff interrupted by user")
                raise
            
            # Clean up after waiting
            self._cleanup_old_requests(datetime.now())
    
    def acquire_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a rate limit slot with optional timeout.
        
        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.
            
        Returns:
            True if slot was acquired, False if timeout occurred
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                self._cleanup_old_requests(datetime.now())
                
                if len(self.requests) < self.max_requests:
                    self.requests.append(datetime.now())
                    return True
                
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False
                
                wait_time = self.get_time_until_next_request()
                if timeout is not None:
                    remaining_timeout = timeout - elapsed
                    wait_time = min(wait_time, remaining_timeout)
                
                if wait_time <= 0:
                    continue
            
            # Sleep outside the lock to allow other threads
            try:
                time.sleep(min(wait_time, 1.0))  # Sleep in 1-second chunks
            except KeyboardInterrupt:
                logger.info("Rate limit acquisition interrupted by user")
                return False


class DeepSeekAPIError(Exception):
    """
    Custom exception for DeepSeek API errors with detailed error categorization.
    
    Provides comprehensive error information including error type, status codes,
    retry information, and contextual details for better error handling.
    """
    
    def __init__(self, message: str, error_type: str = "unknown", 
                 status_code: Optional[int] = None, retry_after: Optional[int] = None,
                 error_code: Optional[str] = None, request_id: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after = retry_after
        self.error_code = error_code
        self.request_id = request_id
        self.original_error = original_error
        self.timestamp = datetime.now()
    
    def is_retryable(self) -> bool:
        """Determine if this error should trigger a retry."""
        retryable_types = {
            "rate_limit", "server_error", "service_error", "timeout", 
            "network", "bad_gateway", "service_unavailable", "gateway_timeout"
        }
        return self.error_type in retryable_types
    
    def is_client_error(self) -> bool:
        """Determine if this is a client-side error (4xx)."""
        client_error_types = {
            "bad_request", "authentication", "authorization", "not_found",
            "request_too_large", "validation"
        }
        return self.error_type in client_error_types or (
            self.status_code and 400 <= self.status_code < 500
        )
    
    def is_server_error(self) -> bool:
        """Determine if this is a server-side error (5xx)."""
        server_error_types = {
            "server_error", "bad_gateway", "service_unavailable", "gateway_timeout"
        }
        return self.error_type in server_error_types or (
            self.status_code and 500 <= self.status_code < 600
        )
    
    def get_suggested_action(self) -> str:
        """Get suggested action based on error type."""
        action_map = {
            "rate_limit": "Wait and retry. Consider implementing exponential backoff.",
            "authentication": "Check API key and credentials.",
            "authorization": "Verify account permissions and subscription status.",
            "bad_request": "Review request parameters and format.",
            "validation": "Check input data format and constraints.",
            "not_found": "Verify endpoint URL and resource existence.",
            "request_too_large": "Reduce request size or split into smaller requests.",
            "server_error": "Retry after a delay. Contact support if persistent.",
            "service_unavailable": "Service is temporarily down. Retry later.",
            "network": "Check network connectivity and retry.",
            "timeout": "Increase timeout or retry with exponential backoff.",
            "unexpected": "Review error details and contact support if needed."
        }
        return action_map.get(self.error_type, "Review error details and retry if appropriate.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging or serialization."""
        return {
            "message": str(self),
            "error_type": self.error_type,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "is_retryable": self.is_retryable(),
            "is_client_error": self.is_client_error(),
            "is_server_error": self.is_server_error(),
            "suggested_action": self.get_suggested_action(),
            "original_error_type": type(self.original_error).__name__ if self.original_error else None
        }
    
    def get_retry_delay(self) -> Optional[int]:
        """Get suggested retry delay from error response."""
        return self.retry_after


class DeepSeekClient:
    """
    Robust client for DeepSeek R1 API on Azure AI Inference SDK.
    
    Features:
    - Exponential backoff retry mechanism
    - Rate limiting with sliding window
    - Comprehensive error handling
    - Request/response logging
    - Connection pooling with health monitoring
    - Response caching with TTL
    """
    
    def __init__(self, config: DeepSeekConfig, enable_caching: bool = True, cache_ttl: int = 3600):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.metrics = APIMetrics()
        
        # Initialize connection pool
        self.connection_pool = ConnectionPool(config)
        
        # Initialize response cache
        self.enable_caching = enable_caching
        self.cache = ResponseCache(default_ttl=cache_ttl) if enable_caching else None
        
        # Initialize primary client for backwards compatibility
        try:
            self.client = ChatCompletionsClient(
                endpoint=config.endpoint,
                credential=AzureKeyCredential(config.api_key)
            )
            logger.info(f"DeepSeek client initialized for endpoint: {config.endpoint}")
        except Exception as e:
            raise DeepSeekAPIError(
                f"Failed to initialize DeepSeek client: {str(e)}", 
                error_type="initialization"
            )
    
    def _validate_messages(self, messages: List[Message]) -> None:
        """Validate message format before sending to API."""
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message at index {i} is missing 'role' or 'content' field")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Unsupported role '{msg['role']}' at message index {i}")
    
    def _get_retry_config(self):
        """Get retry configuration for API requests."""
        return {
            "stop": stop_after_attempt(self.config.retry_attempts),
            "wait": wait_exponential(
                multiplier=self.config.exponential_backoff_base,
                min=1,
                max=self.config.exponential_backoff_max
            ),
            "retry": retry_if_exception_type((HttpResponseError, ServiceRequestError)),
            "before_sleep": before_sleep_log(logger, logging.WARNING)
        }
    
    @classmethod
    def from_environment(cls, env_file: Optional[str] = None) -> 'DeepSeekClient':
        """
        Create client from environment variables.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in the DeepDeliberate project root.
        """
        # Load environment variables from .env file using cross-platform path handling
        if env_file is None:
            # Get the project root directory (where .env should be located)
            # This module is in deepdeliberate/core/, so we need to go up 2 levels
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from core/ to deepdeliberate/ to project root
            env_path = project_root / '.env'
        else:
            env_path = Path(env_file)
        
        # Load the .env file if found
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path.resolve()}")
        else:
            logger.warning(f"No .env file found at {env_path.resolve()}, using system environment variables only")
        
        required_vars = {
            'DEEPSEEK_ENDPOINT': 'endpoint',
            'DEEPSEEK_API_KEY': 'api_key',
        }
        
        config_dict = {}
        missing_vars = []
        
        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
            else:
                # Clean the value to remove comments
                config_dict[config_key] = clean_env_value(value)
        
        if missing_vars:
            raise DeepSeekAPIError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set them in your .env file or system environment.",
                error_type="configuration"
            )
        
        # Optional configuration with defaults from environment or fallback values
        config_dict['model_name'] = clean_env_value(os.getenv('DEEPSEEK_MODEL_NAME', 'DeepSeek-R1'))
        config_dict['max_tokens'] = get_int_env('DEEPSEEK_MAX_TOKENS', default='3000')
        config_dict['timeout_seconds'] = get_int_env('REQUEST_TIMEOUT_SECONDS', 'DEEPSEEK_TIMEOUT', '30')
        config_dict['retry_attempts'] = get_int_env('DEEPSEEK_RETRY_ATTEMPTS', default='3')
        config_dict['rate_limit_per_minute'] = get_int_env('API_RATE_LIMIT_PER_MINUTE', 'DEEPSEEK_RATE_LIMIT', '60')
        
        return cls(DeepSeekConfig(**config_dict))
    
    def _make_request_with_retry(self, messages: List[Message], **kwargs) -> ChatCompletions:
        """Make API request with retry logic applied."""
        retry_config = self._get_retry_config()
        
        @retry(**retry_config)
        def _execute_request():
            return self._make_request_internal(messages, **kwargs)
        
        return _execute_request()
    
    def _make_request_internal(self, messages: List[Message], **kwargs) -> ChatCompletions:
        """Internal method to make API request without retry logic."""
        # Check cache first if enabled
        if self.cache:
            cached_response = self.cache.get(messages, **kwargs)
            if cached_response:
                self.metrics.cache_hits += 1
                logger.debug("Cache hit - returning cached response")
                # Convert cached APIResponse back to ChatCompletions format for consistency
                return self._api_response_to_chat_completions(cached_response)
            else:
                self.metrics.cache_misses += 1
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        client = None
        
        try:
            # Validate message format
            self._validate_messages(messages)
            
            # Get client from connection pool
            client = self.connection_pool.get_client()
            
            # Convert messages to Azure AI Inference format
            azure_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    azure_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    azure_messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    azure_messages.append(AssistantMessage(content=msg["content"]))
            
            # Prepare API call parameters, avoiding duplicate max_tokens
            api_params = {
                'messages': azure_messages,
                'model': self.config.model_name,
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
                'timeout': self.config.timeout_seconds
            }
            
            # Add other kwargs except max_tokens to avoid duplication
            for key, value in kwargs.items():
                if key not in ['max_tokens', 'timeout']:
                    api_params[key] = value
            
            # Make the API call
            response = client.complete(**api_params)
            
            response_time = time.time() - start_time
            logger.info(f"API request completed in {response_time:.2f}s")
            
            return response
            
        except ClientAuthenticationError as e:
            raise DeepSeekAPIError(
                f"Authentication failed: {str(e)}", 
                error_type="authentication",
                status_code=401
            )
        except HttpResponseError as e:
            error_details = self._extract_error_details(e)
            
            if e.status_code == 429:
                retry_after = self._extract_retry_after(e)
                logger.error(f"🚫 DEEPSEEK API RATE LIMIT: Request rejected by DeepSeek API servers! "
                           f"Server returned HTTP 429. Your local limit is too aggressive for DeepSeek's actual limits. "
                           f"Consider reducing API_RATE_LIMIT_PER_MINUTE in .env. Retry-After: {retry_after}")
                raise DeepSeekAPIError(
                    f"Rate limit exceeded: {error_details}", 
                    error_type="rate_limit",
                    status_code=429,
                    retry_after=retry_after
                )
            elif e.status_code == 400:
                raise DeepSeekAPIError(
                    f"Bad request: {error_details}", 
                    error_type="bad_request",
                    status_code=400
                )
            elif e.status_code == 401:
                raise DeepSeekAPIError(
                    f"Unauthorized: {error_details}", 
                    error_type="authentication",
                    status_code=401
                )
            elif e.status_code == 403:
                raise DeepSeekAPIError(
                    f"Forbidden: {error_details}", 
                    error_type="authorization",
                    status_code=403
                )
            elif e.status_code == 404:
                raise DeepSeekAPIError(
                    f"Not found: {error_details}", 
                    error_type="not_found",
                    status_code=404
                )
            elif e.status_code == 413:
                raise DeepSeekAPIError(
                    f"Request too large: {error_details}", 
                    error_type="request_too_large",
                    status_code=413
                )
            elif e.status_code == 422:
                raise DeepSeekAPIError(
                    f"Unprocessable entity: {error_details}", 
                    error_type="validation",
                    status_code=422
                )
            elif e.status_code == 500:
                raise DeepSeekAPIError(
                    f"Internal server error: {error_details}", 
                    error_type="server_error",
                    status_code=500
                )
            elif e.status_code == 502:
                raise DeepSeekAPIError(
                    f"Bad gateway: {error_details}", 
                    error_type="bad_gateway",
                    status_code=502
                )
            elif e.status_code == 503:
                raise DeepSeekAPIError(
                    f"Service unavailable: {error_details}", 
                    error_type="service_unavailable",
                    status_code=503
                )
            elif e.status_code == 504:
                raise DeepSeekAPIError(
                    f"Gateway timeout: {error_details}", 
                    error_type="gateway_timeout",
                    status_code=504
                )
            elif e.status_code >= 500:
                raise DeepSeekAPIError(
                    f"Server error: {error_details}", 
                    error_type="server_error",
                    status_code=e.status_code
                )
            else:
                raise DeepSeekAPIError(
                    f"HTTP error: {error_details}", 
                    error_type="http_error",
                    status_code=e.status_code
                )
        except ServiceRequestError as e:
            raise DeepSeekAPIError(
                f"Service request failed: {str(e)}", 
                error_type="service_error"
            )
        except (ConnectionError, TimeoutError) as e:
            raise DeepSeekAPIError(
                f"Network error: {str(e)}", 
                error_type="network",
                status_code=None
            )
        except ValueError as e:
            raise DeepSeekAPIError(
                f"Invalid request data: {str(e)}", 
                error_type="validation",
                status_code=400
            )
        except Exception as e:
            raise DeepSeekAPIError(
                f"Unexpected error: {str(e)}", 
                error_type="unexpected"
            )
        finally:
            # Return client to pool
            if client:
                self.connection_pool.return_client(client)
    
    def generate_completion(
        self, 
        messages: List[Message], 
        **kwargs: Any
    ) -> APIResponse:
        """
        Generate completion from DeepSeek R1.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for the API call
                - temperature: Controls randomness (0.0-1.0)
                - top_p: Controls diversity via nucleus sampling (0.0-1.0)
                - max_tokens: Maximum tokens to generate (overrides config)
        
        Returns:
            APIResponse object with structured response data
        
        Examples:
            >>> client = DeepSeekClient.from_environment()
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Tell me about Python."}
            ... ]
            >>> response = client.generate_completion(messages, temperature=0.7)
            >>> print(response.content)
            
        Raises:
            DeepSeekAPIError: For various API-related errors
        """
        start_time = time.time()
        self.metrics.total_requests += 1  # Track request
        
        try:
            response = self._make_request_with_retry(messages, **kwargs)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            
            # Parse usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            response_time = time.time() - start_time
            
            # Update metrics for successful request
            self.metrics.successful_requests += 1
            self.metrics.total_tokens_used += usage["total_tokens"]
            self.metrics.last_request_time = datetime.now()
            
            # Calculate and track cost
            request_cost = self._calculate_cost(usage)
            self.metrics.total_cost_estimate += request_cost
            
            # Update average response time
            if self.metrics.successful_requests == 1:
                self.metrics.average_response_time = response_time
            else:
                total_time = (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time)
                self.metrics.average_response_time = total_time / self.metrics.successful_requests
            
            api_response = APIResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                response_time=response_time,
                timestamp=datetime.now()
            )
            
            # Cache the response if caching is enabled
            if self.cache:
                cache_ttl = kwargs.get('cache_ttl', None)
                self.cache.put(messages, api_response, ttl=cache_ttl, **kwargs)
            
            return api_response
            
        except DeepSeekAPIError as e:
            self.metrics.failed_requests += 1
            # Track rate limit hits specifically
            if e.error_type == "rate_limit":
                self.metrics.rate_limit_hits += 1
            
            # Add context to the error
            e.request_id = f"req_{int(time.time() * 1000)}"  # Simple request ID
            logger.error(f"API request failed: {e.to_dict()}")
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            
            # Categorize the error for better handling
            error_type = self._categorize_error(e)
            
            # Create enhanced error with context
            enhanced_error = DeepSeekAPIError(
                f"Failed to generate completion: {str(e)}", 
                error_type=error_type,
                original_error=e,
                request_id=f"req_{int(time.time() * 1000)}"
            )
            
            logger.error(f"Unexpected error in generate_completion: {enhanced_error.to_dict()}")
            raise enhanced_error
    
    def parse_reasoning_content(self, content: str) -> Tuple[str, str]:
        """
        Parse DeepSeek R1 response to separate reasoning from final answer.
        
        This method handles DeepSeek R1's unique response format that includes
        reasoning content wrapped in <think> tags followed by the actual answer.
        
        Args:
            content: Raw response content from DeepSeek R1
            
        Returns:
            Tuple of (reasoning, answer) where:
            - reasoning: Content from <think> tags (empty if no reasoning found)
            - answer: The actual response content after reasoning
        """
        # Multiple patterns for different reasoning formats
        patterns = [
            r"<think>(.*?)</think>(.*)",  # Standard think tags
            r"<reasoning>(.*?)</reasoning>(.*)",  # Alternative reasoning tags
            r"Let me think about this\.\.\.(.*?)(?:In conclusion|Therefore|So,)(.*)",  # Natural language
            r"I need to consider\.\.\.(.*?)(?:Based on this|Given this|Therefore)(.*)",  # Alternative natural language
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = self._clean_text(match.group(1))
                answer = self._clean_text(match.group(2))
                return reasoning, answer
        
        # No reasoning pattern found
        return "", content.strip()
    
    def extract_content_from_response(self, response: 'APIResponse') -> str:
        """
        Extract and parse content from APIResponse, handling DeepSeek R1's reasoning format.
        
        This is a convenience method that combines response content extraction
        with reasoning parsing, returning only the final answer content.
        
        Args:
            response: APIResponse object from generate_completion
            
        Returns:
            Clean answer content with reasoning removed
        """
        if response is None or response.content is None:
            raise DeepSeekAPIError("Invalid response: no content available")
        
        # Parse reasoning to get clean answer content
        reasoning, answer = self.parse_reasoning_content(response.content)
        
        # Return the answer part (which is the actual response content)
        return answer if answer.strip() else response.content.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common artifacts
        text = re.sub(r'^[:\-\s]+', '', text)
        # Remove trailing punctuation artifacts
        text = re.sub(r'[:\-\s]+$', '', text)
        return text
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the API connection.
        
        Returns:
            Dictionary with health check results and detailed information
        """
        start_time = time.time()
        health_result = {
            "healthy": False,
            "response_time": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "error_details": None,
            "suggestions": []
        }
        
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            response = self.generate_completion(test_messages, max_tokens=10)
            health_result["response_time"] = time.time() - start_time
            
            if response.content is not None:
                health_result["healthy"] = True
                health_result["model"] = response.model
                health_result["tokens_used"] = response.usage["total_tokens"]
                logger.info(f"Health check passed in {health_result['response_time']:.2f}s")
            else:
                health_result["error"] = "Empty response content"
                health_result["suggestions"] = ["Check API configuration", "Verify model availability"]
                
        except DeepSeekAPIError as e:
            health_result["response_time"] = time.time() - start_time
            health_result["error"] = str(e)
            health_result["error_details"] = e.to_dict()
            health_result["suggestions"] = [e.get_suggested_action()]
            
            # Log appropriate level based on error type
            self._log_error_metrics(e)
            
        except Exception as e:
            health_result["response_time"] = time.time() - start_time
            enhanced_error = self._handle_api_error(e, "health_check")
            health_result["error"] = str(enhanced_error)
            health_result["error_details"] = enhanced_error.to_dict()
            health_result["suggestions"] = [enhanced_error.get_suggested_action()]
        
        return health_result
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """
        Calculate estimated cost for API usage.
        
        Note: These are estimated rates and may not reflect actual pricing.
        Check DeepSeek's official pricing for accurate costs.
        """
        # Estimated pricing (per 1K tokens) - update with actual rates
        PROMPT_TOKEN_COST = 0.0014  # $0.0014 per 1K prompt tokens
        COMPLETION_TOKEN_COST = 0.0028  # $0.0028 per 1K completion tokens
        
        prompt_cost = (usage["prompt_tokens"] / 1000) * PROMPT_TOKEN_COST
        completion_cost = (usage["completion_tokens"] / 1000) * COMPLETION_TOKEN_COST
        
        return prompt_cost + completion_cost
    
    def _extract_error_details(self, error: HttpResponseError) -> str:
        """Extract detailed error information from HTTP response."""
        try:
            if hasattr(error, 'response') and error.response:
                # Try to parse JSON error response
                try:
                    error_data = error.response.json()
                    if isinstance(error_data, dict):
                        # Common error response formats
                        if 'error' in error_data:
                            error_info = error_data['error']
                            if isinstance(error_info, dict):
                                message = error_info.get('message', str(error))
                                code = error_info.get('code', '')
                                type_info = error_info.get('type', '')
                                return f"{message} (code: {code}, type: {type_info})".strip()
                            else:
                                return str(error_info)
                        elif 'message' in error_data:
                            return error_data['message']
                        elif 'detail' in error_data:
                            return error_data['detail']
                        else:
                            return str(error_data)
                except (ValueError, AttributeError):
                    # If JSON parsing fails, try to get text content
                    try:
                        return error.response.text()[:500]  # Limit to 500 chars
                    except:
                        pass
            
            # Fallback to string representation
            return str(error)
        except Exception:
            # Ultimate fallback
            return f"HTTP {getattr(error, 'status_code', 'unknown')} error"
    
    def _extract_retry_after(self, error: HttpResponseError) -> Optional[int]:
        """Extract Retry-After header from rate limit response."""
        try:
            if hasattr(error, 'response') and error.response and hasattr(error.response, 'headers'):
                retry_after = error.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        return int(retry_after)
                    except ValueError:
                        # Retry-After might be in HTTP date format, but we'll ignore that for now
                        pass
            return None
        except Exception:
            return None
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for better handling and logging."""
        error_type = type(error).__name__
        
        # Network-related errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            return "network"
        
        # Authentication/Authorization errors
        elif isinstance(error, ClientAuthenticationError):
            return "authentication"
        
        # HTTP errors
        elif isinstance(error, HttpResponseError):
            status_code = getattr(error, 'status_code', 0)
            if status_code == 429:
                return "rate_limit"
            elif 400 <= status_code < 500:
                return "client_error"
            elif 500 <= status_code < 600:
                return "server_error"
            else:
                return "http_error"
        
        # Service errors
        elif isinstance(error, ServiceRequestError):
            return "service_error"
        
        # Validation errors
        elif isinstance(error, (ValueError, TypeError)):
            return "validation"
        
        # Unknown errors
        else:
            return "unexpected"
    
    def _handle_api_error(self, error: Exception, context: str = "") -> DeepSeekAPIError:
        """
        Handle and enhance API errors with additional context.
        
        Args:
            error: The original exception
            context: Additional context about when/where the error occurred
            
        Returns:
            Enhanced DeepSeekAPIError with detailed information
        """
        if isinstance(error, DeepSeekAPIError):
            # Already a DeepSeekAPIError, just add context if needed
            if context and not error.request_id:
                error.request_id = f"req_{int(time.time() * 1000)}_{context}"
            return error
        
        # Create new enhanced error
        error_type = self._categorize_error(error)
        
        enhanced_error = DeepSeekAPIError(
            message=f"{context}: {str(error)}" if context else str(error),
            error_type=error_type,
            original_error=error,
            request_id=f"req_{int(time.time() * 1000)}_{context}" if context else f"req_{int(time.time() * 1000)}"
        )
        
        # Log the error with full context
        logger.error(f"API error in {context}: {enhanced_error.to_dict()}")
        
        return enhanced_error
    
    def _log_error_metrics(self, error: DeepSeekAPIError) -> None:
        """Log error metrics for monitoring and debugging."""
        error_info = {
            "error_type": error.error_type,
            "status_code": error.status_code,
            "is_retryable": error.is_retryable(),
            "timestamp": error.timestamp.isoformat(),
            "request_id": error.request_id
        }
        
        # Log at appropriate level based on error type
        if error.is_server_error():
            logger.warning(f"Server error encountered: {error_info}")
        elif error.error_type == "rate_limit":
            logger.info(f"Rate limit hit: {error_info}")
        elif error.is_client_error():
            logger.error(f"Client error: {error_info}")
        else:
            logger.error(f"Unexpected error: {error_info}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics for monitoring."""
        # Use the thread-safe methods instead of accessing internals
        current_requests = self.rate_limiter.get_current_request_count()
        time_until_next = self.rate_limiter.get_time_until_next_request()
        
        return {
            # Configuration info
            "rate_limit_per_minute": self.config.rate_limit_per_minute,
            "current_requests_in_window": current_requests,
            "time_until_next_request": round(time_until_next, 2),
            "max_tokens": self.config.max_tokens,
            "retry_attempts": self.config.retry_attempts,
            "model_name": self.config.model_name,
            
            # Usage metrics
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (self.metrics.successful_requests / max(self.metrics.total_requests, 1)) * 100,
            
            # Token and cost metrics
            "total_tokens_used": self.metrics.total_tokens_used,
            "estimated_total_cost": round(self.metrics.total_cost_estimate, 4),
            "average_tokens_per_request": (self.metrics.total_tokens_used / max(self.metrics.successful_requests, 1)),
            
            # Performance metrics
            "average_response_time": round(self.metrics.average_response_time, 3),
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None
        }
    
    def get_metrics(self) -> APIMetrics:
        """Get the raw metrics object."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all usage metrics to zero."""
        self.metrics = APIMetrics()
    
    def _api_response_to_chat_completions(self, api_response: APIResponse) -> ChatCompletions:
        """Convert APIResponse back to ChatCompletions format for internal consistency."""
        # This is a simplified conversion - in a real implementation,
        # you might need to reconstruct the full ChatCompletions object
        # For now, we'll create a mock object that has the necessary attributes
        class MockChoice:
            def __init__(self, content: str, finish_reason: str):
                self.message = type('Message', (), {'content': content})()
                self.finish_reason = finish_reason
        
        class MockUsage:
            def __init__(self, usage_dict: Dict[str, int]):
                self.prompt_tokens = usage_dict.get('prompt_tokens', 0)
                self.completion_tokens = usage_dict.get('completion_tokens', 0)
                self.total_tokens = usage_dict.get('total_tokens', 0)
        
        class MockChatCompletions:
            def __init__(self, api_response: APIResponse):
                self.choices = [MockChoice(api_response.content, api_response.finish_reason)]
                self.model = api_response.model
                self.usage = MockUsage(api_response.usage)
        
        return MockChatCompletions(api_response)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the DeepSeek API integration.
        
        Returns:
            Dictionary with health check results including connectivity,
            authentication, and component status.
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            # 1. API Connectivity Check
            connectivity_result = await self._check_api_connectivity()
            health_status["checks"]["api_connectivity"] = connectivity_result
            
            # 2. Authentication Check
            auth_result = await self._check_authentication()
            health_status["checks"]["authentication"] = auth_result
            
            # 3. Connection Pool Health
            pool_result = self._check_connection_pool_health()
            health_status["checks"]["connection_pool"] = pool_result
            
            # 4. Rate Limiter Status
            rate_limit_result = self._check_rate_limiter_health()
            health_status["checks"]["rate_limiter"] = rate_limit_result
            
            # 5. Cache Health (if enabled)
            if self.cache:
                cache_result = self._check_cache_health()
                health_status["checks"]["cache"] = cache_result
            
            # Determine overall status
            failed_checks = [
                check for check in health_status["checks"].values() 
                if not check.get("healthy", False)
            ]
            
            if failed_checks:
                health_status["overall_status"] = "degraded" if len(failed_checks) < 3 else "unhealthy"
                health_status["failed_checks"] = len(failed_checks)
            
        except Exception as e:
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    async def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check API connectivity with a minimal test request."""
        try:
            # Use a minimal test message to validate connectivity
            test_messages = [
                {"role": "user", "content": "test"}
            ]
            
            start_time = time.time()
            
            # Make a minimal request with very low token limit
            response = self._make_request_with_retry(
                test_messages, 
                max_tokens=DEFAULT_HEALTH_CHECK_TOKENS,
                temperature=0.1
            )
            
            response_time = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "endpoint": self.config.endpoint,
                "model": response.model if hasattr(response, 'model') else self.config.model_name,
                "message": "API connectivity verified"
            }
            
        except DeepSeekAPIError as e:
            return {
                "healthy": False,
                "error_type": e.error_type,
                "error_message": str(e),
                "status_code": e.status_code,
                "is_retryable": e.is_retryable(),
                "suggested_action": e.get_suggested_action()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error_type": "unexpected",
                "error_message": str(e),
                "suggested_action": "Check network connectivity and API configuration"
            }
    
    async def _check_authentication(self) -> Dict[str, Any]:
        """Check authentication status."""
        try:
            # Authentication is validated during connectivity check
            # Here we just verify the API key format and configuration
            if not self.config.api_key or len(self.config.api_key) < MIN_API_KEY_LENGTH:
                return {
                    "healthy": False,
                    "error_message": "Invalid or missing API key",
                    "suggested_action": "Verify DEEPSEEK_API_KEY environment variable"
                }
            
            return {
                "healthy": True,
                "api_key_length": len(self.config.api_key),
                "endpoint_configured": bool(self.config.endpoint),
                "message": "Authentication configuration valid"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e),
                "suggested_action": "Check API key and endpoint configuration"
            }
    
    def _check_connection_pool_health(self) -> Dict[str, Any]:
        """Check connection pool health."""
        try:
            pool_status = self.connection_pool.get_pool_status()
            
            # Consider pool healthy if at least 50% of connections are available
            utilization = pool_status.get("utilization_percent", 0)
            is_healthy = utilization < 90  # Less than 90% utilization
            
            return {
                "healthy": is_healthy,
                "pool_status": pool_status,
                "message": "Connection pool operational" if is_healthy else "Connection pool under high load"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e),
                "suggested_action": "Check connection pool configuration"
            }
    
    def _check_rate_limiter_health(self) -> Dict[str, Any]:
        """Check rate limiter health."""
        try:
            rate_limit_status = self.rate_limiter.get_rate_limit_status()
            
            # Consider healthy if not at maximum capacity
            is_healthy = rate_limit_status.get("requests_remaining", 0) > 0
            
            return {
                "healthy": is_healthy,
                "rate_limit_status": rate_limit_status,
                "message": "Rate limiter operational"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e),
                "suggested_action": "Check rate limiter configuration"
            }
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health if caching is enabled."""
        try:
            if not self.cache:
                return {
                    "healthy": True,
                    "message": "Caching disabled"
                }
            
            cache_stats = self.cache.get_stats()
            
            # Clean up expired entries
            expired_removed = self.cache.clear_expired()
            
            # Consider healthy if utilization is reasonable
            utilization = cache_stats.get("utilization_percent", 0)
            is_healthy = utilization < 95  # Less than 95% utilization
            
            return {
                "healthy": is_healthy,
                "cache_stats": cache_stats,
                "expired_entries_cleaned": expired_removed,
                "cache_hit_rate": (
                    (self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)) * 100
                ),
                "message": "Cache operational" if is_healthy else "Cache near capacity"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e),
                "suggested_action": "Check cache configuration and memory usage"
            }
    
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get detailed connection pool status."""
        return self.connection_pool.get_pool_status()
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self.cache:
            stats = self.cache.get_stats()
            stats["hit_rate"] = (
                (self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)) * 100
            )
            return stats
        return None
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Response cache cleared")
    
    def close(self) -> None:
        """Close all resources including connection pool."""
        try:
            self.connection_pool.close()
            if self.cache:
                self.cache.clear()
            logger.info("DeepSeek client resources closed")
        except Exception as e:
            logger.warning(f"Error closing DeepSeek client resources: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False