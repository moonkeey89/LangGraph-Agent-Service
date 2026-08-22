from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from ai_agent_learning.api.models import (
    MAX_IDENTIFIER_LENGTH,
    _normalized_identifier,
)
from ai_agent_learning.api.service import AgentService


def get_agent_service(request: Request) -> AgentService:
    service = getattr(request.app.state, "agent_service", None)
    if not isinstance(service, AgentService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent service is not ready",
        )
    return service


def get_user_id(
    x_user_id: Annotated[
        str,
        Header(
            alias="X-User-ID",
            min_length=1,
            max_length=MAX_IDENTIFIER_LENGTH,
        ),
    ],
) -> str:
    try:
        return _normalized_identifier(x_user_id, "X-User-ID")
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
