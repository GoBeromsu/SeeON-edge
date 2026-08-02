"""Server-side dashboard login, logout, and credential-rotation routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.shared.dashboard_auth import (
    DASHBOARD_SESSION_COOKIE,
    DashboardSessionStore,
    WrongCurrentPasswordError,
    authorize_dashboard,
    dashboard_sessions,
    rotate_dashboard_credentials,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class DashboardLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class DashboardCredentialsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=512)
    username: str | None = Field(default=None, min_length=1, max_length=128)
    new_password: str = Field(min_length=4, max_length=512)


def _set_session_cookie(
    response: Response, request: Request, sessions: DashboardSessionStore, token: str
) -> None:
    response.set_cookie(
        DASHBOARD_SESSION_COOKIE,
        token,
        max_age=sessions.ttl_seconds,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


@router.post("/session", status_code=status.HTTP_204_NO_CONTENT)
def login(payload: DashboardLoginRequest, request: Request, response: Response) -> None:
    sessions = dashboard_sessions(request)
    token = sessions.authenticate(payload.username, payload.password)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid dashboard credentials",
        )
    _set_session_cookie(response, request, sessions, token)


@router.get("/session", status_code=status.HTTP_204_NO_CONTENT)
def session(request: Request) -> None:
    authorize_dashboard(request)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response) -> None:
    dashboard_sessions(request).revoke(request.cookies.get(DASHBOARD_SESSION_COOKIE))
    response.delete_cookie(DASHBOARD_SESSION_COOKIE, path="/", samesite="strict")


@router.put("/credentials", status_code=status.HTTP_204_NO_CONTENT)
def update_credentials(
    payload: DashboardCredentialsUpdateRequest, request: Request, response: Response
) -> None:
    authorize_dashboard(request)
    try:
        token = rotate_dashboard_credentials(
            request,
            current_password=payload.current_password,
            new_username=payload.username,
            new_password=payload.new_password,
        )
    except WrongCurrentPasswordError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="current password is incorrect",
        ) from exc
    _set_session_cookie(response, request, dashboard_sessions(request), token)


__all__ = ["router"]
