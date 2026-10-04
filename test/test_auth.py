"""
Тесты для backend/app/auth.py.

Запуск: из корня проекта  ->  pytest tests/test_auth.py
"""

import types

import pytest
from fastapi import HTTPException

from backend.app.auth import require_admin


def _fake_request(admin_api_key: str):
    state = types.SimpleNamespace(settings=types.SimpleNamespace(admin_api_key=admin_api_key))
    app = types.SimpleNamespace(state=state)
    return types.SimpleNamespace(app=app)


class TestRequireAdmin:
    @pytest.mark.asyncio
    async def test_correct_token_passes(self):
        request = _fake_request(admin_api_key="secret-123")
        # Не должно поднять исключение.
        await require_admin(request, x_admin_token="secret-123")

    @pytest.mark.asyncio
    async def test_wrong_token_raises_403(self):
        request = _fake_request(admin_api_key="secret-123")
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(request, x_admin_token="wrong-token")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_token_raises_403(self):
        request = _fake_request(admin_api_key="secret-123")
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(request, x_admin_token="")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_non_ascii_token_gives_403_not_server_error(self):
        # Раньше secrets.compare_digest(str, str) на не-ASCII символах бросал
        # TypeError, и такой запрос получал 500 вместо 403.
        request = _fake_request(admin_api_key="secret-123")
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(request, x_admin_token="правильный-токен")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_non_ascii_configured_key_is_compared_correctly(self):
        request = _fake_request(admin_api_key="ключ-админа")
        await require_admin(request, x_admin_token="ключ-админа")  # не падает
        with pytest.raises(HTTPException):
            await require_admin(request, x_admin_token="другой-ключ")