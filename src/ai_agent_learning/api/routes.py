import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ai_agent_learning.api.dependencies import get_agent_service, get_user_id
from ai_agent_learning.api.models import (
    AgentResponse,
    HealthResponse,
    InterruptResponse,
    InvokeRequest,
    ResumeRequest,
)
from ai_agent_learning.api.service import AgentExecutionResult, AgentService


logger = logging.getLogger(__name__)
router = APIRouter()


def _response(result: AgentExecutionResult) -> AgentResponse:
    return AgentResponse(
        status=result.status,
        thread_id=result.thread_id,
        answer=result.answer,
        interrupts=[
            InterruptResponse(
                interrupt_id=item.interrupt_id,
                payload=item.payload,
            )
            for item in result.interrupts
        ],
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/api/v1/agent/invoke", response_model=AgentResponse)
async def invoke_agent(
    request: InvokeRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AgentResponse:
    result = await run_in_threadpool(
        service.invoke,
        message=request.message,
        thread_id=request.thread_id,
        user_id=user_id,
    )
    return _response(result)


@router.post("/api/v1/agent/resume", response_model=AgentResponse)
async def resume_agent(
    request: ResumeRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AgentResponse:
    result = await run_in_threadpool(
        service.resume,
        thread_id=request.thread_id,
        user_id=user_id,
        decision=request.decision,
        reason=request.reason,
    )
    return _response(result)
