import logging
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, cast
from uuid import uuid4

from ai_agent_learning.knowledge.service import KnowledgeServiceError
from ai_agent_learning.research.catalog import ResearchCatalog
from ai_agent_learning.research.models import (
    RESEARCH_PROJECT_STATUSES,
    RESEARCH_TASK_TYPES,
    ResearchProject,
    ResearchProjectStatus,
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
)
from ai_agent_learning.research.task_state import (
    InvalidTaskTransitionError,
    InvalidTaskTransitionInputError,
    transition_task_state,
)


logger = logging.getLogger(__name__)
MAX_PROJECT_NAME_LENGTH = 120
MAX_PROJECT_DESCRIPTION_LENGTH = 2_000
MAX_RESEARCH_QUESTION_LENGTH = 5_000
MAX_TASK_TITLE_LENGTH = 200
MAX_TASK_OBJECTIVE_LENGTH = 5_000
MAX_ACCEPTANCE_CRITERIA = 20
MAX_ACCEPTANCE_CRITERION_LENGTH = 1_000
_UNSET = object()


class KnowledgeOwnershipVerifier(Protocol):
    def ensure_owned(
        self,
        knowledge_base_id: str,
        owner_user_id: str,
    ) -> None: ...


class ResearchServiceError(RuntimeError):
    status_code = 400
    public_message = "Research project request could not be completed"


class ResearchProjectNotFoundError(ResearchServiceError):
    status_code = 404
    public_message = "Research project resource was not found"


class ResearchKnowledgeBaseNotFoundError(ResearchServiceError):
    status_code = 404
    public_message = "Knowledge base resource was not found"


class ResearchProjectConflictError(ResearchServiceError):
    status_code = 409
    public_message = "Research project conflicts with existing data"


class ResearchProjectValidationError(ResearchServiceError):
    status_code = 422

    def __init__(self, message: str):
        super().__init__(message)
        self.public_message = message


class ResearchTaskNotFoundError(ResearchServiceError):
    status_code = 404
    public_message = "Research task resource was not found"


class ResearchTaskConflictError(ResearchServiceError):
    status_code = 409
    public_message = "Research task operation conflicts with current state"


class ResearchTaskValidationError(ResearchServiceError):
    status_code = 422

    def __init__(self, message: str):
        super().__init__(message)
        self.public_message = message


class ResearchPersistenceError(ResearchServiceError):
    status_code = 500
    public_message = "Research project storage is unavailable"


class ResearchService:
    """Apply project policy before accessing ResearchCatalog."""

    def __init__(
        self,
        catalog: ResearchCatalog,
        knowledge_service: KnowledgeOwnershipVerifier,
    ):
        self.catalog = catalog
        self.knowledge_service = knowledge_service

    def create_project(
        self,
        *,
        owner_user_id: str,
        name: str,
        description: str = "",
        research_question: str = "",
        status: ResearchProjectStatus = "draft",
        default_knowledge_base_id: str | None = None,
    ) -> ResearchProject:
        owner = self._owner(owner_user_id)
        normalized_name = _name(name)
        normalized_description = _optional_text(
            description,
            "description",
            MAX_PROJECT_DESCRIPTION_LENGTH,
        )
        normalized_question = _optional_text(
            research_question,
            "research_question",
            MAX_RESEARCH_QUESTION_LENGTH,
        )
        normalized_status = _status(status)
        normalized_knowledge_base_id = self._knowledge_base(
            default_knowledge_base_id,
            owner,
        )
        timestamp = _now()
        project = ResearchProject(
            project_id=f"rp_{uuid4().hex}",
            owner_user_id=owner,
            name=normalized_name,
            description=normalized_description,
            research_question=normalized_question,
            status=normalized_status,
            default_knowledge_base_id=normalized_knowledge_base_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        try:
            return self.catalog.create(project)
        except sqlite3.IntegrityError as error:
            raise ResearchProjectConflictError from error
        except sqlite3.Error as error:
            logger.exception("Research project creation failed")
            raise ResearchPersistenceError from error

    def list_projects(self, owner_user_id: str) -> list[ResearchProject]:
        try:
            return self.catalog.list_by_owner(self._owner(owner_user_id))
        except sqlite3.Error as error:
            logger.exception("Research project listing failed")
            raise ResearchPersistenceError from error

    def get_project(
        self,
        project_id: str,
        owner_user_id: str,
    ) -> ResearchProject:
        owner = self._owner(owner_user_id)
        return self._get_owned(_project_id(project_id), owner)

    def update_project(
        self,
        project_id: str,
        owner_user_id: str,
        *,
        name: str | object = _UNSET,
        description: str | object = _UNSET,
        research_question: str | object = _UNSET,
        status: ResearchProjectStatus | object = _UNSET,
        default_knowledge_base_id: str | None | object = _UNSET,
    ) -> ResearchProject:
        if all(
            value is _UNSET
            for value in (
                name,
                description,
                research_question,
                status,
                default_knowledge_base_id,
            )
        ):
            raise ResearchProjectValidationError(
                "PATCH 至少需要提供一个可更新字段"
            )

        owner = self._owner(owner_user_id)
        current = self._get_owned(_project_id(project_id), owner)
        updated = ResearchProject(
            project_id=current.project_id,
            owner_user_id=current.owner_user_id,
            name=current.name if name is _UNSET else _name(cast(str, name)),
            description=(
                current.description
                if description is _UNSET
                else _optional_text(
                    cast(str, description),
                    "description",
                    MAX_PROJECT_DESCRIPTION_LENGTH,
                )
            ),
            research_question=(
                current.research_question
                if research_question is _UNSET
                else _optional_text(
                    cast(str, research_question),
                    "research_question",
                    MAX_RESEARCH_QUESTION_LENGTH,
                )
            ),
            status=(
                current.status
                if status is _UNSET
                else _status(cast(ResearchProjectStatus, status))
            ),
            default_knowledge_base_id=(
                current.default_knowledge_base_id
                if default_knowledge_base_id is _UNSET
                else self._knowledge_base(
                    cast(str | None, default_knowledge_base_id),
                    owner,
                )
            ),
            created_at=current.created_at,
            updated_at=_now(),
        )
        try:
            return self.catalog.update(updated)
        except KeyError as error:
            raise ResearchProjectNotFoundError from error
        except sqlite3.IntegrityError as error:
            raise ResearchProjectConflictError from error
        except sqlite3.Error as error:
            logger.exception("Research project update failed")
            raise ResearchPersistenceError from error

    def delete_project(self, project_id: str, owner_user_id: str) -> None:
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        self._get_owned(normalized_project_id, owner)
        try:
            if self.catalog.has_tasks(normalized_project_id):
                raise ResearchProjectConflictError
            deleted = self.catalog.delete(normalized_project_id, owner)
        except ResearchProjectConflictError:
            raise
        except sqlite3.IntegrityError as error:
            raise ResearchProjectConflictError from error
        except sqlite3.Error as error:
            logger.exception("Research project deletion failed")
            raise ResearchPersistenceError from error
        if not deleted:
            raise ResearchProjectNotFoundError

    def create_task(
        self,
        project_id: str,
        owner_user_id: str,
        *,
        title: str,
        objective: str = "",
        task_type: ResearchTaskType = "general",
        acceptance_criteria: list[str] | None = None,
    ) -> ResearchTask:
        owner = self._owner(owner_user_id)
        project = self._get_owned(_project_id(project_id), owner)
        self._ensure_project_tasks_mutable(project)
        timestamp = _now()
        task = ResearchTask(
            task_id=f"rt_{uuid4().hex}",
            project_id=project.project_id,
            title=_task_title(title),
            objective=_task_objective(objective),
            task_type=_task_type(task_type),
            status="pending",
            acceptance_criteria=_acceptance_criteria(
                acceptance_criteria or []
            ),
            result_summary=None,
            error_message=None,
            created_at=timestamp,
            updated_at=timestamp,
            started_at=None,
            completed_at=None,
        )
        try:
            return self.catalog.create_task(task)
        except sqlite3.IntegrityError as error:
            raise ResearchTaskConflictError from error
        except sqlite3.Error as error:
            logger.exception("Research task creation failed")
            raise ResearchPersistenceError from error

    def list_tasks(
        self,
        project_id: str,
        owner_user_id: str,
    ) -> list[ResearchTask]:
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        self._get_owned(normalized_project_id, owner)
        try:
            return self.catalog.list_tasks(normalized_project_id)
        except sqlite3.Error as error:
            logger.exception("Research task listing failed")
            raise ResearchPersistenceError from error

    def get_task(
        self,
        project_id: str,
        task_id: str,
        owner_user_id: str,
    ) -> ResearchTask:
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        self._get_owned(normalized_project_id, owner)
        return self._get_task(normalized_project_id, _task_id(task_id))

    def update_task(
        self,
        project_id: str,
        task_id: str,
        owner_user_id: str,
        *,
        title: str | object = _UNSET,
        objective: str | object = _UNSET,
        task_type: ResearchTaskType | object = _UNSET,
        acceptance_criteria: list[str] | object = _UNSET,
    ) -> ResearchTask:
        if all(
            value is _UNSET
            for value in (title, objective, task_type, acceptance_criteria)
        ):
            raise ResearchTaskValidationError(
                "PATCH 至少需要提供一个可更新字段"
            )
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        project = self._get_owned(normalized_project_id, owner)
        self._ensure_project_tasks_mutable(project)
        current = self._get_task(normalized_project_id, _task_id(task_id))
        updated = replace(
            current,
            title=(
                current.title
                if title is _UNSET
                else _task_title(cast(str, title))
            ),
            objective=(
                current.objective
                if objective is _UNSET
                else _task_objective(cast(str, objective))
            ),
            task_type=(
                current.task_type
                if task_type is _UNSET
                else _task_type(cast(ResearchTaskType, task_type))
            ),
            acceptance_criteria=(
                current.acceptance_criteria
                if acceptance_criteria is _UNSET
                else _acceptance_criteria(
                    cast(list[str], acceptance_criteria)
                )
            ),
            updated_at=_now(),
        )
        return self._save_task(updated)

    def transition_task(
        self,
        project_id: str,
        task_id: str,
        owner_user_id: str,
        *,
        target_status: ResearchTaskStatus,
        reason: str | None = None,
        result_summary: str | None = None,
    ) -> ResearchTask:
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        project = self._get_owned(normalized_project_id, owner)
        self._ensure_project_tasks_mutable(project)
        current = self._get_task(normalized_project_id, _task_id(task_id))
        try:
            transitioned = transition_task_state(
                current,
                target_status,
                reason=reason,
                result_summary=result_summary,
            )
        except InvalidTaskTransitionError as error:
            raise ResearchTaskConflictError from error
        except InvalidTaskTransitionInputError as error:
            raise ResearchTaskValidationError(str(error)) from error
        return self._save_task(transitioned)

    def delete_task(
        self,
        project_id: str,
        task_id: str,
        owner_user_id: str,
    ) -> None:
        owner = self._owner(owner_user_id)
        normalized_project_id = _project_id(project_id)
        project = self._get_owned(normalized_project_id, owner)
        self._ensure_project_tasks_mutable(project)
        normalized_task_id = _task_id(task_id)
        task = self._get_task(normalized_project_id, normalized_task_id)
        if task.status not in {"pending", "cancelled"}:
            raise ResearchTaskConflictError
        try:
            deleted = self.catalog.delete_task(
                normalized_project_id,
                normalized_task_id,
            )
        except sqlite3.Error as error:
            logger.exception("Research task deletion failed")
            raise ResearchPersistenceError from error
        if not deleted:
            raise ResearchTaskNotFoundError

    def _get_owned(self, project_id: str, owner_user_id: str) -> ResearchProject:
        try:
            project = self.catalog.get_by_id(project_id)
        except sqlite3.Error as error:
            logger.exception("Research project lookup failed")
            raise ResearchPersistenceError from error
        if project is None or project.owner_user_id != owner_user_id:
            # Missing and foreign-owned IDs intentionally have the same response.
            raise ResearchProjectNotFoundError
        return project

    def _knowledge_base(
        self,
        knowledge_base_id: str | None,
        owner_user_id: str,
    ) -> str | None:
        if knowledge_base_id is None:
            return None
        normalized = knowledge_base_id.strip()
        if not normalized:
            raise ResearchProjectValidationError(
                "default_knowledge_base_id 不能为空字符串"
            )
        try:
            self.knowledge_service.ensure_owned(normalized, owner_user_id)
        except (KnowledgeServiceError, KeyError, ValueError) as error:
            # A foreign and a missing knowledge base are deliberately indistinguishable.
            raise ResearchKnowledgeBaseNotFoundError from error
        return normalized

    def _get_task(self, project_id: str, task_id: str) -> ResearchTask:
        try:
            task = self.catalog.get_task(project_id, task_id)
        except sqlite3.Error as error:
            logger.exception("Research task lookup failed")
            raise ResearchPersistenceError from error
        if task is None:
            # A missing task and a task under another project look identical.
            raise ResearchTaskNotFoundError
        return task

    def _save_task(self, task: ResearchTask) -> ResearchTask:
        try:
            return self.catalog.update_task(task)
        except KeyError as error:
            raise ResearchTaskNotFoundError from error
        except sqlite3.IntegrityError as error:
            raise ResearchTaskConflictError from error
        except sqlite3.Error as error:
            logger.exception("Research task update failed")
            raise ResearchPersistenceError from error

    @staticmethod
    def _ensure_project_tasks_mutable(project: ResearchProject) -> None:
        if project.status == "archived":
            raise ResearchTaskConflictError

    @staticmethod
    def _owner(owner_user_id: str) -> str:
        normalized = owner_user_id.strip()
        if not normalized:
            raise ResearchProjectValidationError("owner_user_id 不能为空")
        return normalized


def _project_id(project_id: str) -> str:
    normalized = project_id.strip()
    if not normalized:
        raise ResearchProjectNotFoundError
    return normalized


def _name(value: str) -> str:
    if not isinstance(value, str):
        raise ResearchProjectValidationError("name 不能为 null")
    normalized = value.strip()
    if not normalized:
        raise ResearchProjectValidationError("name 不能为空")
    if len(normalized) > MAX_PROJECT_NAME_LENGTH:
        raise ResearchProjectValidationError(
            f"name 不能超过 {MAX_PROJECT_NAME_LENGTH} 个字符"
        )
    return normalized


def _optional_text(value: str, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ResearchProjectValidationError(f"{field_name} 不能为 null")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ResearchProjectValidationError(
            f"{field_name} 不能超过 {max_length} 个字符"
        )
    return normalized


def _status(value: str) -> ResearchProjectStatus:
    if value not in RESEARCH_PROJECT_STATUSES:
        raise ResearchProjectValidationError(
            "status 只能是 draft、active 或 archived"
        )
    return cast(ResearchProjectStatus, value)


def _task_id(task_id: str) -> str:
    normalized = task_id.strip()
    if not normalized:
        raise ResearchTaskNotFoundError
    return normalized


def _task_title(value: str) -> str:
    if not isinstance(value, str):
        raise ResearchTaskValidationError("title 不能为 null")
    normalized = value.strip()
    if not normalized:
        raise ResearchTaskValidationError("title 不能为空")
    if len(normalized) > MAX_TASK_TITLE_LENGTH:
        raise ResearchTaskValidationError(
            f"title 不能超过 {MAX_TASK_TITLE_LENGTH} 个字符"
        )
    return normalized


def _task_objective(value: str) -> str:
    if not isinstance(value, str):
        raise ResearchTaskValidationError("objective 不能为 null")
    normalized = value.strip()
    if len(normalized) > MAX_TASK_OBJECTIVE_LENGTH:
        raise ResearchTaskValidationError(
            f"objective 不能超过 {MAX_TASK_OBJECTIVE_LENGTH} 个字符"
        )
    return normalized


def _task_type(value: str) -> ResearchTaskType:
    if value not in RESEARCH_TASK_TYPES:
        raise ResearchTaskValidationError(
            "task_type 只能是 literature_review、analysis、synthesis 或 general"
        )
    return cast(ResearchTaskType, value)


def _acceptance_criteria(values: list[str]) -> list[str]:
    if not isinstance(values, list):
        raise ResearchTaskValidationError("acceptance_criteria 必须是字符串列表")
    if len(values) > MAX_ACCEPTANCE_CRITERIA:
        raise ResearchTaskValidationError(
            f"acceptance_criteria 最多包含 {MAX_ACCEPTANCE_CRITERIA} 项"
        )
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ResearchTaskValidationError(
                "acceptance_criteria 必须是字符串列表"
            )
        criterion = value.strip()
        if not criterion:
            raise ResearchTaskValidationError("验收标准不能为空")
        if len(criterion) > MAX_ACCEPTANCE_CRITERION_LENGTH:
            raise ResearchTaskValidationError(
                "单条验收标准不能超过 "
                f"{MAX_ACCEPTANCE_CRITERION_LENGTH} 个字符"
            )
        normalized.append(criterion)
    return normalized


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
