from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from ai_agent_learning.auth import AuthenticatedSession, AuthService
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


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if not isinstance(service, AuthService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not ready",
        )
    return service


def get_authenticated_session(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthenticatedSession:
    config = service.cookie_config
    return service.authenticate(request.cookies.get(config.session_cookie_name))


def get_user_id(
    request: Request,
    authenticated: Annotated[
        AuthenticatedSession,
        Depends(get_authenticated_session),
    ],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> str:
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        config = service.cookie_config
        service.validate_csrf(
            authenticated,
            csrf_cookie=request.cookies.get(config.csrf_cookie_name),
            csrf_header=request.headers.get(config.csrf_header_name),
        )
    return authenticated.user.user_id


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
