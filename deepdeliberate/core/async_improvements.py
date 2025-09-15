"""
Async improvements for DeepSeek client and agent handling.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

class AsyncDeepSeekClient:
    """Async version of DeepSeek client for better performance."""
    
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = AsyncRateLimiter(config.rate_limit_per_minute)
    
    @asynccontextmanager
    async def session_context(self):
        """Async context manager for HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
        try:
            yield self.session
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    async def generate_completion_async(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Async completion generation with proper error handling."""
        await self.rate_limiter.acquire()
        
        async with self.session_context() as session:
            try:
                # Implement async API call logic here
                pass
            except asyncio.TimeoutError:
                raise DeepSeekAPIError("Request timeout", error_type="timeout")
            except Exception as e:
                raise DeepSeekAPIError(f"Async request failed: {e}", error_type="network")

class AsyncRateLimiter:
    """Async rate limiter using asyncio primitives."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Async rate limit acquisition."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            # Remove old requests
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.requests.append(now)