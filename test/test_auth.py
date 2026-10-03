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