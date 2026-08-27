from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from starlette.concurrency import run_in_threadpool

from ai_agent_learning.api.dependencies import (
    get_auth_service,
    get_authenticated_session,
)
from ai_agent_learning.api.models import (
    AuthUserResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
)
from ai_agent_learning.auth import AuthenticatedSession, AuthService, User
from ai_agent_learning.auth.models import from_utc_text


router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
CSRF_COOKIE_CONTRACT_HEADER = "X-ResearchFlow-CSRF-Cookie"
CSRF_HEADER_CONTRACT_HEADER = "X-ResearchFlow-CSRF-Header"


def _user_response(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def _set_auth_cookies(response: Response, service: AuthService, issued) -> None:
    config = service.cookie_config
    common = {
        "secure": config.secure,
        "samesite": config.same_site,
        "path": "/",
        "domain": config.domain,
        "max_age": config.max_age_seconds,
        "expires": from_utc_text(issued.session.expires_at),
    }
    response.set_cookie(
        config.session_cookie_name,
        issued.session_token,
        httponly=True,
        **common,
    )
    response.set_cookie(
        config.csrf_cookie_name,
        issued.csrf_token,
        httponly=False,
        **common,
    )
    _set_csrf_contract_headers(response, service)


def _set_csrf_contract_headers(response: Response, service: AuthService) -> None:
    config = service.cookie_config
    response.headers[CSRF_COOKIE_CONTRACT_HEADER] = config.csrf_cookie_name
    response.headers[CSRF_HEADER_CONTRACT_HEADER] = config.csrf_header_name


def _clear_auth_cookies(response: Response, service: AuthService) -> None:
    config = service.cookie_config
    for name, httponly in (
        (config.session_cookie_name, True),
        (config.csrf_cookie_name, False),
    ):
        response.delete_cookie(
            name,
            path="/",
            domain=config.domain,
            secure=config.secure,
            httponly=httponly,
            samesite=config.same_site,
        )


@router.post(
    "/register",
    response_model=AuthUserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    request: RegisterRequest,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthUserResponse:
    user = await run_in_threadpool(
        service.register,
        username=request.username,
        email=request.email,
        password=request.password.get_secret_value(),
    )
    return _user_response(user)


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    response: Response,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> LoginResponse:
    issued = await run_in_threadpool(
        service.login,
        login=request.login,
        password=request.password.get_secret_value(),
    )
    _set_auth_cookies(response, service, issued)
    return LoginResponse(
        user=_user_response(issued.user),
        expires_at=issued.session.expires_at,
    )


@router.get("/me", response_model=AuthUserResponse)
async def me(
    response: Response,
    authenticated: Annotated[
        AuthenticatedSession,
        Depends(get_authenticated_session),
    ],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthUserResponse:
    _set_csrf_contract_headers(response, service)
    return _user_response(authenticated.user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    authenticated: Annotated[
        AuthenticatedSession,
        Depends(get_authenticated_session),
    ],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> Response:
    config = service.cookie_config
    service.validate_csrf(
        authenticated,
        csrf_cookie=request.cookies.get(config.csrf_cookie_name),
        csrf_header=request.headers.get(config.csrf_header_name),
    )
    await run_in_threadpool(service.logout, authenticated)
    _clear_auth_cookies(response, service)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
