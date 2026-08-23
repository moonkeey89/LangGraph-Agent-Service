from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from ai_agent_learning.api.models import (
    MAX_IDENTIFIER_LENGTH,
    _normalized_identifier,
)
from ai_agent_learning.api.service import AgentService
from ai_agent_learning.knowledge.service import KnowledgeLibraryService
from ai_agent_learning.research.service import ResearchService
from ai_agent_learning.research.execution import ResearchExecutionService


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


def get_knowledge_service(
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> KnowledgeLibraryService:
    if service.knowledge_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge service is not ready",
        )
    return service.knowledge_service


def get_research_service(
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> ResearchService:
    if not isinstance(service.research_service, ResearchService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research service is not ready",
        )
    return service.research_service


def get_research_execution_service(
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> ResearchExecutionService:
    if not isinstance(
        service.research_execution_service,
        ResearchExecutionService,
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research execution service is not ready",
        )
    return service.research_execution_service
