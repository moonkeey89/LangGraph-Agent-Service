from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from ai_agent_learning.api.dependencies import (
    get_knowledge_service,
    get_user_id,
)
from ai_agent_learning.api.models import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseResponse,
    KnowledgeDocumentResponse,
    KnowledgeUploadItemResponse,
    KnowledgeUploadResponse,
)
from ai_agent_learning.knowledge.models import (
    KnowledgeBaseRecord,
    KnowledgeDocumentRecord,
)
from ai_agent_learning.knowledge.service import (
    KnowledgeLibraryService,
    UploadCandidate,
)


router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


def _base_response(record: KnowledgeBaseRecord) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(**record.__dict__)


def _document_response(
    record: KnowledgeDocumentRecord,
) -> KnowledgeDocumentResponse:
    data = dict(record.__dict__)
    data.pop("stored_filename", None)
    return KnowledgeDocumentResponse(**data)


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> KnowledgeBaseResponse:
    result = await run_in_threadpool(
        service.create_knowledge_base,
        owner_user_id=user_id,
        name=request.name,
        description=request.description,
    )
    return _base_response(result)


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> list[KnowledgeBaseResponse]:
    records = await run_in_threadpool(service.list_knowledge_bases, user_id)
    return [_base_response(record) for record in records]


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> KnowledgeBaseResponse:
    record = await run_in_threadpool(
        service.get_knowledge_base,
        knowledge_base_id,
        user_id,
    )
    return _base_response(record)


@router.delete("/{knowledge_base_id}", status_code=204)
async def delete_knowledge_base(
    knowledge_base_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> Response:
    await run_in_threadpool(
        service.delete_knowledge_base,
        knowledge_base_id=knowledge_base_id,
        owner_user_id=user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=KnowledgeUploadResponse,
)
async def upload_documents(
    knowledge_base_id: str,
    files: Annotated[list[UploadFile], File(...)],
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> KnowledgeUploadResponse:
    uploads = [
        UploadCandidate(
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            stream=file.file,
        )
        for file in files
    ]
    try:
        results = await run_in_threadpool(
            service.upload_documents,
            knowledge_base_id=knowledge_base_id,
            owner_user_id=user_id,
            uploads=uploads,
        )
    finally:
        for file in files:
            await file.close()
    return KnowledgeUploadResponse(
        items=[
            KnowledgeUploadItemResponse(
                document=_document_response(item.document),
                duplicate=item.duplicate,
            )
            for item in results
        ]
    )


@router.get(
    "/{knowledge_base_id}/documents",
    response_model=list[KnowledgeDocumentResponse],
)
async def list_documents(
    knowledge_base_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> list[KnowledgeDocumentResponse]:
    records = await run_in_threadpool(
        service.list_documents,
        knowledge_base_id=knowledge_base_id,
        owner_user_id=user_id,
    )
    return [_document_response(record) for record in records]


@router.delete(
    "/{knowledge_base_id}/documents/{document_id}",
    status_code=204,
)
async def delete_document(
    knowledge_base_id: str,
    document_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
    service: Annotated[KnowledgeLibraryService, Depends(get_knowledge_service)],
) -> Response:
    await run_in_threadpool(
        service.delete_document,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        owner_user_id=user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
