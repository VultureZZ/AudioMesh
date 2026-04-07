"""
API key authentication middleware.

If API_KEY is not set in environment, any API key (or no key) is accepted.
If API_KEY is set, validates against it.
"""
from typing import Callable, Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..config import config


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and validate API key if required.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response from the next handler or 401 if authentication fails
        """
        # Skip authentication for health check and docs endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # Extract API key from header or query parameter
        api_key: Optional[str] = None

        # Check X-API-Key header
        if "X-API-Key" in request.headers:
            api_key = request.headers["X-API-Key"]
        # Check api_key query parameter
        elif "api_key" in request.query_params:
            api_key = request.query_params["api_key"]

        # Validate API key
        if not config.validate_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

        # Store API key in request state for rate limiting
        # Use a default key if none provided (for rate limiting purposes)
        request.state.api_key = api_key or "anonymous"

        return await call_next(request)
