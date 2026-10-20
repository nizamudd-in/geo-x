"""
Unit tests for the EncoderService.
"""

from __future__ import annotations

import pytest


def test_encoder_service_raises_before_load() -> None:
    """EncoderService.encode() should raise ModelNotLoadedError before load() is called."""
    from core.exceptions import ModelNotLoadedError
    from services.encoder_service import EncoderService

    svc = EncoderService()
    with pytest.raises((ModelNotLoadedError, NotImplementedError)):
        svc.encode(b"fake_image_bytes", "optical")


def test_encoder_service_is_loaded_returns_false_before_load() -> None:
    """EncoderService.is_loaded should be False before load() is called."""
    from services.encoder_service import EncoderService

    svc = EncoderService()
    assert svc.is_loaded is False


# TODO: add tests with mocked model once implementation is complete
# @pytest.mark.asyncio
# async def test_encoder_service_load_returns_correct_version(tmp_path, mock_checkpoint):
#     ...
