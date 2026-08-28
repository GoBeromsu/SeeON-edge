"""HEAD support for routes that already build a full GET response.

RFC 9110 section 9.3.2: a HEAD response carries the same header section the
matching GET would send -- ``Content-Length`` and ``Content-Type`` included --
and no body. FastAPI's ``APIRoute`` does not synthesise HEAD from GET the way
Starlette's plain ``Route`` does, so a ``@router.get`` media route answers HEAD
with 405/404 (issue #452). The fix is to register both methods on one endpoint
and drop the body here, so headers are computed exactly once and the two
methods can never drift.

This module is a leaf: it imports nothing from ``backend.app.features``/
``routes``/``main``/``lifespan``, satisfying the "backend base (core/shared)
does not import upper layers" import-linter contract in ``pyproject.toml``.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response

HEAD_METHODS = ("GET", "HEAD")
"""``methods=`` for a route that answers HEAD with the GET header section."""


def is_head(request: Request) -> bool:
    return request.method.upper() == "HEAD"


def drop_body_for_head(request: Request, response: Response) -> Response:
    """Blank ``response``'s body when the request is a HEAD.

    ``Response.init_headers`` has already frozen ``raw_headers`` from the body
    the GET path rendered, so emptying ``body`` afterwards leaves every header
    -- ``Content-Length`` above all -- byte-identical to the GET.

    Only for responses whose body is already in memory and whose length cannot
    be known without it. A response that would *stream* a file on GET must
    suppress its own reads instead (see
    ``backend.app.features.clips.media_response.OpenedFileResponse``), so that
    a HEAD never pays to read bytes it will not send.
    """
    if is_head(request):
        response.body = b""
    return response


__all__ = ["HEAD_METHODS", "drop_body_for_head", "is_head"]
