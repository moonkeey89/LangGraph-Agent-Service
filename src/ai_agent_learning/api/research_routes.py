from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status
from starlette.concurrency import run_in_threadpool

from ai_agent_learning.api.dependencies import get_research_service, get_user_id
from ai_agent_learning.api.models import (
    CreateResearchProjectRequest,
    ErrorResponse,
    ResearchProjectResponse,
    UpdateResearchProjectRequest,
)
from ai_agent_learning.research.models import ResearchProject
from ai_agent_learning.research.service import ResearchService


router = APIRouter(prefix="/api/v1/research/projects", tags=["research-projects"])
BUSINESS_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Project or knowledge base not found"},
    409: {"model": ErrorResponse, "description": "Project data conflict"},
    422: {"model": ErrorResponse, "description": "Request or business validation failed"},
}


def _response(project: ResearchProject) -> ResearchProjectResponse:
    return ResearchProjectResponse.model_validate(project)


@router.post(
    "",
    response_model=ResearchProjectResponse,
    status_code=status.HTTP_201_CREATED,
    responses=BUSINESS_RESPONSES,
)
async def create_research_project(
    request: CreateResearchProjectRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchProjectResponse:
    project = await run_in_threadpool(
        service.create_project,
        owner_user_id=user_id,
        **request.model_dump(),
    )
    return _response(project)


@router.get("", response_model=list[ResearchProjectResponse])
async def list_research_projects(
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> list[ResearchProjectResponse]:
    projects = await run_in_threadpool(service.list_projects, user_id)
    return [_response(project) for project in projects]


@router.get(
    "/{project_id}",
    response_model=ResearchProjectResponse,
    responses=BUSINESS_RESPONSES,
)
async def get_research_project(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchProjectResponse:
    project = await run_in_threadpool(
        service.get_project,
        project_id,
        user_id,
    )
    return _response(project)


@router.patch(
    "/{project_id}",
    response_model=ResearchProjectResponse,
    responses=BUSINESS_RESPONSES,
)
async def update_research_project(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: UpdateResearchProjectRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchProjectResponse:
    project = await run_in_threadpool(
        service.update_project,
        project_id,
        user_id,
        **request.changes(),
    )
    return _response(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=BUSINESS_RESPONSES,
)
async def delete_research_project(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> Response:
    await run_in_threadpool(service.delete_project, project_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
