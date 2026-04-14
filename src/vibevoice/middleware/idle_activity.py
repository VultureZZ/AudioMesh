"""
Touch global idle-memory activity on HTTP requests (excluding health/docs).
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..idle_memory import touch_activity

_EXCLUDED_PATHS = frozenset({"/", "/health"})
_EXCLUDED_PREFIXES = ("/docs", "/openapi.json", "/redoc", "/static")


class IdleActivityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path not in _EXCLUDED_PATHS and not any(
            path.startswith(p) for p in _EXCLUDED_PREFIXES
        ):
            touch_activity()
        return await call_next(request)
