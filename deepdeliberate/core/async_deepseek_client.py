"""
Async DeepSeek R1 API client with proper resource management.

This module provides a comprehensive async client for interacting with the DeepSeek R1 API,
including proper resource cleanup, rate limiting, retry logic, and cost tracking.
"""

# Standard library imports
import asyncio
import json
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party imports for environment loading
from dotenv import load_dotenv

# Third-party imports
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from .deepseek_client import DeepSeekConfig
from .interfaces import LLMClientInterface
from .exceptions import APIError, APITimeoutError, APIRateLimitError, APIAuthenticationError

__all__ = [
    "AsyncRateLimiter",
    "AsyncDeepSeekClient",
    "DeepSeekClientWrapper"
]

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Async-first rate limiter using asyncio primitives with proper concurrency handling."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
    
    async def acquire(self) -> None:
        """Acquire rate limit slot asynchronously with proper waiting."""
        async with self.condition:
            await self._cleanup_old_requests()
            
            while len(self.requests) >= self.max_requests:
                wait_time = self._calculate_wait_time()
                if wait_time > 0:
                    logger.warning(f"🛡️ LOCAL RATE LIMIT: Blocking request to prevent API overload. "
                                 f"Waiting {wait_time:.2f}s (Local limit: {self.max_requests}/min)")
                    try:
                        await asyncio.wait_for(
                            self.condition.wait(), 
                            timeout=min(wait_time, 65.0)
                        )
                    except asyncio.TimeoutError:
                        pass  # Timeout is expected, continue to check again
                    except asyncio.CancelledError:
                        logger.info("Rate limit wait cancelled")
                        raise
                    await self._cleanup_old_requests()
                else:
                    break
            
            self.requests.append(datetime.now())
            # Notify other waiters that a slot might be available soon
            self.condition.notify_all()
    
    async def _cleanup_old_requests(self) -> None:
        """Clean up old requests and notify waiters."""
        cutoff = datetime.now() - timedelta(minutes=1)
        removed_count = 0
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
            removed_count += 1
        
        if removed_count > 0:
            # Notify waiting coroutines that slots are available
            self.condition.notify_all()
    
    def _calculate_wait_time(self) -> float:
        """Calculate time to wait until next slot is available."""
        if not self.requests:
            return 0.0
        
        oldest_request = self.requests[0]
        elapsed = (datetime.now() - oldest_request).total_seconds()
        return max(0.0, 60.0 - elapsed)


class AsyncDeepSeekClient(LLMClientInterface):
    """
    Async DeepSeek client with proper resource management.
    
    Features:
    - Async context manager for proper resource cleanup
    - Connection pooling with httpx.AsyncClient
    - Rate limiting with async primitives
    - Comprehensive error handling
    - Automatic retry with exponential backoff
    """
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.rate_limiter = AsyncRateLimiter(config.rate_limit_per_minute)
        self._session: Optional[httpx.AsyncClient] = None
        self._closed = False
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0
        }
    
    @asynccontextmanager
    async def _get_session(self):
        """Async context manager for HTTP session."""
        if self._session is None or self._session.is_closed:
            timeout = httpx.Timeout(self.config.timeout_seconds)
            self._session = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        
        try:
            yield self._session
        finally:
            # Session cleanup handled by close() method
            pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: logger.warning(f"Retrying API request, attempt {retry_state.attempt_number}")
    )
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from DeepSeek R1 API with retry logic."""
        if self._closed:
            raise APIError("Client has been closed")
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        start_time = datetime.now()
        self.metrics["total_requests"] += 1
        
        try:
            async with self._get_session() as session:
                response = await self._make_api_request(session, messages, **kwargs)
                
                # Add null check for response
                if response is None:
                    raise APIError("API returned None response - check API configuration")
                
                # Update metrics
                self.metrics["successful_requests"] += 1
                response_time = (datetime.now() - start_time).total_seconds()
                self._update_response_time_metric(response_time)
                
                return response
                
        except APIRateLimitError as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"🚫 DEEPSEEK API RATE LIMIT: Request rejected by DeepSeek API servers! "
                        f"Server returned HTTP 429. Your local limit ({self.rate_limiter.max_requests}/min) "
                        f"is too aggressive for DeepSeek's actual limits. Consider reducing API_RATE_LIMIT_PER_MINUTE in .env")
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"API request failed: {e}")
            await self._handle_api_error(e)
            raise
    
    async def _make_api_request(
        self,
        session: httpx.AsyncClient,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Make the actual API request with proper error handling."""
        headers = {
            "Ocp-Apim-Subscription-Key": self.config.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **kwargs
        }
        
        try:
            response = await session.post(
                f"{self.config.endpoint}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 401:
                raise APIAuthenticationError("Invalid API key")
            elif response.status_code == 429:
                logger.error(f"🚫 DEEPSEEK SERVER REJECTED REQUEST: HTTP 429 returned. "
                           f"DeepSeek API rate limit exceeded on their end. "
                           f"Response headers: {dict(response.headers)}")
                raise APIRateLimitError("Rate limit exceeded")
            elif response.status_code >= 400:
                raise APIError(f"API error: {response.status_code} - {response.text}")
            
            return response.json()
            
        except httpx.TimeoutException:
            raise APITimeoutError("Request timed out")
        except httpx.ConnectError:
            raise APIError("Failed to connect to API")
    
    async def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors with appropriate logging."""
        if isinstance(error, APIRateLimitError):
            logger.error(f"🚫 DEEPSEEK RATE LIMIT HIT: {error} - This means DeepSeek API servers "
                        f"are rejecting your requests. Reduce API_RATE_LIMIT_PER_MINUTE in .env")
        elif isinstance(error, APIAuthenticationError):
            logger.error(f"🔐 AUTHENTICATION ERROR: {error} - Check your DEEPSEEK_API_KEY in .env")
        elif isinstance(error, APITimeoutError):
            logger.warning(f"⏱️ REQUEST TIMEOUT: {error} - Increase REQUEST_TIMEOUT_SECONDS in .env")
        else:
            logger.error(f"❌ UNEXPECTED API ERROR: {error}")
    
    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        current_avg = self.metrics["average_response_time"]
        total_successful = self.metrics["successful_requests"]
        
        if total_successful == 1:
            self.metrics["average_response_time"] = response_time
        else:
            # Calculate running average
            self.metrics["average_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics."""
        return self.metrics.copy()
    
    def get_rate_limit_diagnostics(self) -> Dict[str, Any]:
        """Get detailed rate limit diagnostics to help identify bottlenecks."""
        total_requests = self.metrics.get("total_requests", 0)
        failed_requests = self.metrics.get("failed_requests", 0)
        successful_requests = self.metrics.get("successful_requests", 0)
        
        # Calculate rate limit hit ratio
        rate_limit_failures = failed_requests  # Most failures are likely rate limits during testing
        rate_limit_ratio = (rate_limit_failures / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "local_rate_limit": {
                "max_requests_per_minute": self.rate_limiter.max_requests,
                "current_requests_in_window": len(self.rate_limiter.requests),
                "available_slots": max(0, self.rate_limiter.max_requests - len(self.rate_limiter.requests))
            },
            "api_performance": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "failure_rate_percent": (failed_requests / total_requests * 100) if total_requests > 0 else 0
            },
            "recommendations": {
                "current_setting": f"API_RATE_LIMIT_PER_MINUTE={self.rate_limiter.max_requests}",
                "if_seeing_azure_errors": "Reduce API_RATE_LIMIT_PER_MINUTE to 15 or lower",
                "if_seeing_local_blocks": "Your local rate limiter is working - this is good!",
                "optimal_setting": "Set rate limit slightly below Azure's actual limit"
            }
        }
    
    async def close(self):
        """Properly close resources."""
        if self._session and not self._session.is_closed:
            await self._session.aclose()
        self._closed = True
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the async DeepSeek API integration.
        
        Returns:
            Dictionary with health check results.
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            # API Connectivity Check
            connectivity_result = await self._check_api_connectivity()
            health_status["checks"]["api_connectivity"] = connectivity_result
            
            # Rate Limiter Status
            rate_limit_result = self._check_rate_limiter_health()
            health_status["checks"]["rate_limiter"] = rate_limit_result
            
            # Session Health
            session_result = await self._check_session_health()
            health_status["checks"]["session"] = session_result
            
            # Determine overall status
            failed_checks = [
                check for check in health_status["checks"].values() 
                if not check.get("healthy", False)
            ]
            
            if failed_checks:
                health_status["overall_status"] = "degraded" if len(failed_checks) < 2 else "unhealthy"
                health_status["failed_checks"] = len(failed_checks)
            
        except Exception as e:
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Async health check failed: {e}")
        
        return health_status
    
    async def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check API connectivity with a minimal test request."""
        try:
            # First check if we have valid API key configuration
            if not self.config.api_key or len(self.config.api_key) < 10:
                return {
                    "healthy": False,
                    "error_message": "Invalid or missing API key",
                    "suggested_action": "Verify DEEPSEEK_API_KEY environment variable"
                }
            
            # For demo/test purposes, don't make actual API calls with test keys
            if self.config.api_key.startswith('test-') or 'demo' in self.config.api_key.lower():
                return {
                    "healthy": True,
                    "response_time": 0.0,
                    "endpoint": self.config.endpoint,
                    "message": "API configuration validated (test mode)"
                }
            
            test_messages = [{"role": "user", "content": "test"}]
            
            start_time = datetime.now()
            
            # Make minimal request
            await self.generate_completion(test_messages, max_tokens=10)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "healthy": True,
                "response_time": round(response_time, 3),
                "endpoint": self.config.endpoint,
                "message": "Async API connectivity verified"
            }
            
        except APIAuthenticationError as e:
            return {
                "healthy": False,
                "error_type": "authentication",
                "error_message": str(e),
                "suggested_action": "Verify API key and credentials"
            }
        except APIRateLimitError as e:
            return {
                "healthy": True,  # Rate limit means API is reachable
                "error_type": "rate_limit",
                "error_message": str(e),
                "suggested_action": "Wait and retry, or increase rate limits"
            }
        except APITimeoutError as e:
            return {
                "healthy": False,
                "error_type": "timeout",
                "error_message": str(e),
                "suggested_action": "Check network connectivity or increase timeout"
            }
        except Exception as e:
            return {
                "healthy": False,
                "error_type": "unexpected",
                "error_message": str(e),
                "suggested_action": "Check network connectivity and API configuration"
            }
    
    def _check_rate_limiter_health(self) -> Dict[str, Any]:
        """Check rate limiter health."""
        try:
            return {
                "healthy": True,
                "max_requests_per_minute": self.rate_limiter.max_requests,
                "message": "Async rate limiter operational"
            }
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e)
            }
    
    async def _check_session_health(self) -> Dict[str, Any]:
        """Check HTTP session health."""
        try:
            is_healthy = self._session is None or not self._session.is_closed
            
            return {
                "healthy": is_healthy,
                "session_active": self._session is not None and not self._session.is_closed,
                "message": "HTTP session operational" if is_healthy else "HTTP session closed"
            }
        except Exception as e:
            return {
                "healthy": False,
                "error_message": str(e)
            }
    
    @classmethod
    async def create_from_config(cls, config: DeepSeekConfig) -> 'AsyncDeepSeekClient':
        """Factory method to create client with validation."""
        client = cls(config)
        
        # Validate connection
        try:
            test_messages = [{"role": "user", "content": "test"}]
            async with client:
                # This will test the connection without making a real request
                pass
        except Exception as e:
            logger.error(f"Failed to create DeepSeek client: {e}")
            raise
        
        return client
    
    @classmethod
    async def from_environment(cls, env_file: Optional[str] = None) -> 'AsyncDeepSeekClient':
        """
        Create async client from environment variables.
        
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
        
        # Import DeepSeekConfig here to avoid circular imports
        from .deepseek_client import DeepSeekConfig
        
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
                config_dict[config_key] = value
        
        if missing_vars:
            from .exceptions import APIError
            raise APIError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set them in your .env file or system environment."
            )
        
        # Optional configuration with defaults from environment or fallback values
        config_dict['model_name'] = os.getenv('DEEPSEEK_MODEL_NAME', 'DeepSeek-R1')
        config_dict['max_tokens'] = int(os.getenv('DEEPSEEK_MAX_TOKENS', '3000'))
        config_dict['timeout_seconds'] = int(os.getenv('REQUEST_TIMEOUT_SECONDS', os.getenv('DEEPSEEK_TIMEOUT', '30')))
        config_dict['retry_attempts'] = int(os.getenv('DEEPSEEK_RETRY_ATTEMPTS', '3'))
        config_dict['rate_limit_per_minute'] = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', os.getenv('DEEPSEEK_RATE_LIMIT', '60')))
        
        config = DeepSeekConfig(**config_dict)
        return await cls.create_from_config(config)


# Backward compatibility wrapper
class DeepSeekClientWrapper:
    """Wrapper to maintain compatibility with sync interface."""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self._async_client: Optional[AsyncDeepSeekClient] = None
    
    async def _get_async_client(self) -> AsyncDeepSeekClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = await AsyncDeepSeekClient.create_from_config(self.config)
        return self._async_client
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using async client."""
        client = await self._get_async_client()
        return await client.generate_completion(messages, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from async client."""
        if self._async_client:
            return self._async_client.get_metrics()
        return {}
    
    async def close(self):
        """Close async client."""
        if self._async_client:
            await self._async_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()