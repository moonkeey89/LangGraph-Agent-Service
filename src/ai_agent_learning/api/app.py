import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextlib import AbstractContextManager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_agent_learning.api.frontend import attach_frontend
from ai_agent_learning.api.knowledge_routes import router as knowledge_router
from ai_agent_learning.api.research_routes import router as research_router
from ai_agent_learning.api.resources import open_agent_service
from ai_agent_learning.api.routes import router
from ai_agent_learning.api.service import AgentService, AgentServiceError
from ai_agent_learning.knowledge.service import KnowledgeServiceError
from ai_agent_learning.research.service import ResearchServiceError


logger = logging.getLogger(__name__)
ServiceFactory = Callable[[], AbstractContextManager[AgentService]]


def create_app(
    service_factory: ServiceFactory | None = None,
) -> FastAPI:
    factory = service_factory or open_agent_service

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with factory() as service:
            app.state.agent_service = service
            logger.info("FastAPI Agent resources started")
            try:
                yield
            finally:
                app.state.agent_service = None
                logger.info("FastAPI Agent resources stopped")

    application = FastAPI(
        title="AI Agent Learning API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    application.include_router(knowledge_router)
    application.include_router(research_router)
    attach_frontend(application)

    @application.exception_handler(AgentServiceError)
    async def handle_service_error(
        _request: Request,
        error: AgentServiceError,
    ) -> JSONResponse:
        logger.warning("Agent service rejected request: %s", type(error).__name__)
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.public_message},
        )

    @application.exception_handler(KnowledgeServiceError)
    async def handle_knowledge_service_error(
        _request: Request,
        error: KnowledgeServiceError,
    ) -> JSONResponse:
        logger.warning(
            "Knowledge service rejected request: %s",
            type(error).__name__,
        )
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.public_message},
        )

    @application.exception_handler(ResearchServiceError)
    async def handle_research_service_error(
        _request: Request,
        error: ResearchServiceError,
    ) -> JSONResponse:
        logger.warning(
            "Research service rejected request: %s",
            type(error).__name__,
        )
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.public_message},
        )

    @application.exception_handler(Exception)
    async def handle_internal_error(
        _request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled Agent API error", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal agent error"},
        )

    return application


app = create_app()
