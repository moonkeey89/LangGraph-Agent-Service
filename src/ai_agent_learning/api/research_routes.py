from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status
from starlette.concurrency import run_in_threadpool

from ai_agent_learning.api.dependencies import get_research_service, get_user_id
from ai_agent_learning.api.models import (
    AgentRunResponse,
    ArtifactStatus,
    ArtifactType,
    CreateResearchArtifactRequest,
    CreateResearchProjectRequest,
    CreateResearchTaskRequest,
    ErrorResponse,
    ResearchProjectResponse,
    ResearchArtifactResponse,
    ResearchTaskResponse,
    TransitionResearchTaskRequest,
    UpdateResearchProjectRequest,
    UpdateResearchArtifactRequest,
    UpdateResearchTaskRequest,
)
from ai_agent_learning.research.models import (
    AgentRun,
    ResearchArtifact,
    ResearchProject,
    ResearchTask,
)
from ai_agent_learning.research.service import ResearchService


router = APIRouter(prefix="/api/v1/research/projects", tags=["research-projects"])
BUSINESS_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Project or knowledge base not found"},
    409: {"model": ErrorResponse, "description": "Project data conflict"},
    422: {"model": ErrorResponse, "description": "Request or business validation failed"},
}
TASK_BUSINESS_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Project or task not found"},
    409: {"model": ErrorResponse, "description": "Task state conflict"},
    422: {"model": ErrorResponse, "description": "Request or business validation failed"},
}
ARTIFACT_BUSINESS_RESPONSES = {
    404: {
        "model": ErrorResponse,
        "description": "Project, task, artifact, or evidence source not found",
    },
    409: {"model": ErrorResponse, "description": "Artifact state conflict"},
    422: {"model": ErrorResponse, "description": "Request or business validation failed"},
}
RUN_BUSINESS_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Project, task, or run not found"},
}


def _response(project: ResearchProject) -> ResearchProjectResponse:
    return ResearchProjectResponse.model_validate(project)


def _task_response(task: ResearchTask) -> ResearchTaskResponse:
    return ResearchTaskResponse.model_validate(task)


def _artifact_response(artifact: ResearchArtifact) -> ResearchArtifactResponse:
    return ResearchArtifactResponse.model_validate(artifact)


def _run_response(run: AgentRun) -> AgentRunResponse:
    return AgentRunResponse.model_validate(run)


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


@router.post(
    "/{project_id}/tasks",
    response_model=ResearchTaskResponse,
    status_code=status.HTTP_201_CREATED,
    responses=TASK_BUSINESS_RESPONSES,
)
async def create_research_task(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: CreateResearchTaskRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchTaskResponse:
    task = await run_in_threadpool(
        service.create_task,
        project_id,
        user_id,
        **request.model_dump(),
    )
    return _task_response(task)


@router.get(
    "/{project_id}/tasks",
    response_model=list[ResearchTaskResponse],
    responses=TASK_BUSINESS_RESPONSES,
)
async def list_research_tasks(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> list[ResearchTaskResponse]:
    tasks = await run_in_threadpool(
        service.list_tasks,
        project_id,
        user_id,
    )
    return [_task_response(task) for task in tasks]


@router.get(
    "/{project_id}/tasks/{task_id}",
    response_model=ResearchTaskResponse,
    responses=TASK_BUSINESS_RESPONSES,
)
async def get_research_task(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchTaskResponse:
    task = await run_in_threadpool(
        service.get_task,
        project_id,
        task_id,
        user_id,
    )
    return _task_response(task)


@router.patch(
    "/{project_id}/tasks/{task_id}",
    response_model=ResearchTaskResponse,
    responses=TASK_BUSINESS_RESPONSES,
)
async def update_research_task(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: UpdateResearchTaskRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchTaskResponse:
    task = await run_in_threadpool(
        service.update_task,
        project_id,
        task_id,
        user_id,
        **request.changes(),
    )
    return _task_response(task)


@router.post(
    "/{project_id}/tasks/{task_id}/transition",
    response_model=ResearchTaskResponse,
    responses=TASK_BUSINESS_RESPONSES,
)
async def transition_research_task(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: TransitionResearchTaskRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchTaskResponse:
    task = await run_in_threadpool(
        service.transition_task,
        project_id,
        task_id,
        user_id,
        **request.model_dump(),
    )
    return _task_response(task)


@router.delete(
    "/{project_id}/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=TASK_BUSINESS_RESPONSES,
)
async def delete_research_task(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> Response:
    await run_in_threadpool(
        service.delete_task,
        project_id,
        task_id,
        user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{project_id}/artifacts",
    response_model=ResearchArtifactResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def create_research_artifact(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: CreateResearchArtifactRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchArtifactResponse:
    artifact = await run_in_threadpool(
        service.create_artifact,
        project_id,
        user_id,
        created_by="user",
        **request.model_dump(),
    )
    return _artifact_response(artifact)


@router.get(
    "/{project_id}/artifacts",
    response_model=list[ResearchArtifactResponse],
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def list_research_artifacts(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
    task_id: Annotated[str | None, Query(max_length=128)] = None,
    artifact_type: ArtifactType | None = None,
    artifact_status: Annotated[
        ArtifactStatus | None,
        Query(alias="status"),
    ] = None,
) -> list[ResearchArtifactResponse]:
    artifacts = await run_in_threadpool(
        service.list_artifacts,
        project_id,
        user_id,
        task_id=task_id,
        artifact_type=artifact_type,
        status=artifact_status,
    )
    return [_artifact_response(artifact) for artifact in artifacts]


@router.get(
    "/{project_id}/artifacts/{artifact_id}",
    response_model=ResearchArtifactResponse,
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def get_research_artifact(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    artifact_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchArtifactResponse:
    artifact = await run_in_threadpool(
        service.get_artifact,
        project_id,
        artifact_id,
        user_id,
    )
    return _artifact_response(artifact)


@router.patch(
    "/{project_id}/artifacts/{artifact_id}",
    response_model=ResearchArtifactResponse,
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def update_research_artifact(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    artifact_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: UpdateResearchArtifactRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchArtifactResponse:
    artifact = await run_in_threadpool(
        service.update_artifact,
        project_id,
        artifact_id,
        user_id,
        **request.changes(),
    )
    return _artifact_response(artifact)


@router.post(
    "/{project_id}/artifacts/{artifact_id}/finalize",
    response_model=ResearchArtifactResponse,
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def finalize_research_artifact(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    artifact_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchArtifactResponse:
    artifact = await run_in_threadpool(
        service.finalize_artifact,
        project_id,
        artifact_id,
        user_id,
    )
    return _artifact_response(artifact)


@router.delete(
    "/{project_id}/artifacts/{artifact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ARTIFACT_BUSINESS_RESPONSES,
)
async def delete_research_artifact(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    artifact_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> Response:
    await run_in_threadpool(
        service.delete_artifact,
        project_id,
        artifact_id,
        user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{project_id}/tasks/{task_id}/runs",
    response_model=list[AgentRunResponse],
    responses=RUN_BUSINESS_RESPONSES,
)
async def list_agent_runs(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> list[AgentRunResponse]:
    runs = await run_in_threadpool(
        service.list_runs,
        project_id,
        task_id,
        user_id,
    )
    return [_run_response(run) for run in runs]


@router.get(
    "/{project_id}/tasks/{task_id}/runs/{run_id}",
    response_model=AgentRunResponse,
    responses=RUN_BUSINESS_RESPONSES,
)
async def get_agent_run(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
    task_id: Annotated[str, Path(min_length=1, max_length=128)],
    run_id: Annotated[str, Path(min_length=1, max_length=128)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> AgentRunResponse:
    run = await run_in_threadpool(
        service.get_run,
        project_id,
        task_id,
        run_id,
        user_id,
    )
    return _run_response(run)
