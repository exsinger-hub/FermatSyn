import os
import sys

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models.modules as modules  # noqa: E402


def test_default_checkpoint_uses_modules_dir(monkeypatch, tmp_path):
    fake_modules_py = tmp_path / "modules.py"
    fake_modules_py.write_text("")
    checkpoint = tmp_path / "sam2_hiera_large.pt"
    checkpoint.write_bytes(b"stub")

    monkeypatch.delenv(modules.SAM2_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(modules, "__file__", str(fake_modules_py))

    assert modules.resolve_sam2_checkpoint_path() == str(checkpoint)


def test_env_checkpoint_overrides_default(monkeypatch, tmp_path):
    checkpoint = tmp_path / "env_sam2.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setenv(modules.SAM2_CHECKPOINT_ENV, str(checkpoint))

    assert modules.resolve_sam2_checkpoint_path() == str(checkpoint)


def test_missing_checkpoint_raises_clear_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pt"
    monkeypatch.delenv(modules.SAM2_CHECKPOINT_ENV, raising=False)

    with pytest.raises(FileNotFoundError) as exc_info:
        modules.resolve_sam2_checkpoint_path(str(missing))

    message = str(exc_info.value)
    assert str(missing) in message
    assert modules.SAM2_CHECKPOINT_ENV in message
