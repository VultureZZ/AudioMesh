"""
Rate limiting middleware.

Implements per-API-key rate limiting with configurable requests per minute.
"""
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..config import config


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting per API key."""

    def __init__(self, app, requests_per_minute: int = None):
        """
        Initialize rate limiting middleware.

        Args:
            app: The FastAPI application
            requests_per_minute: Number of requests allowed per minute (defaults to config)
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or config.RATE_LIMIT_PER_MINUTE
        # Store request timestamps per API key
        # Structure: {api_key: [timestamp1, timestamp2, ...]}
        self.request_timestamps: dict[str, list[float]] = defaultdict(list)
        self.window_seconds = 60  # 1 minute window

    def _cleanup_old_requests(self, api_key: str) -> None:
        """
        Remove timestamps older than the time window.

        Args:
            api_key: The API key to clean up
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Keep only timestamps within the window
        self.request_timestamps[api_key] = [
            ts for ts in self.request_timestamps[api_key] if ts > cutoff_time
        ]

    def _check_rate_limit(self, api_key: str) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Args:
            api_key: The API key to check

        Returns:
            Tuple of (is_allowed, limit, remaining)
        """
        current_time = time.time()

        # Clean up old requests
        self._cleanup_old_requests(api_key)

        # Get current request count
        request_count = len(self.request_timestamps[api_key])

        # Check if limit exceeded
        if request_count >= self.requests_per_minute:
            return False, self.requests_per_minute, 0

        # Add current request timestamp
        self.request_timestamps[api_key].append(current_time)

        remaining = self.requests_per_minute - request_count - 1
        return True, self.requests_per_minute, remaining

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and apply rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response from the next handler or 429 if rate limit exceeded
        """
        # Skip rate limiting for health check and docs endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # Get API key from request state (set by auth middleware)
        api_key = getattr(request.state, "api_key", "anonymous")

        # Check rate limit
        is_allowed, limit, remaining = self._check_rate_limit(api_key)

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        # Call next handler
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
