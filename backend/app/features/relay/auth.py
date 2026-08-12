"""Local worker-to-ml-api relay authentication."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def authorize_relay(request: Request, relay_token: str | None) -> None:
    """Require the single process-configured local relay credential."""
    expected = getattr(request.app.state, "edge_relay_token", None)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="relay token is not configured",
        )
    if relay_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="relay token required",
        )
    if relay_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="relay token mismatch",
        )


__all__ = ["authorize_relay"]
