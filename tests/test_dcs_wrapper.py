"""Unit tests for the DCS wrapper's env-construction wiring.

These do NOT require dm_control / distracting_control / MuJoCo: they inject a
fake ``distracting_control.suite`` into ``sys.modules`` and assert that
``_build_env`` calls the crate's REAL API (``suite.load``), NOT the
``DistractingControlEnv`` class that the code previously assumed (which does
not exist in the library and would raise ``ImportError`` at runtime). They also
pin the difficulty / resolution / background-video-split threading so the
wiring cannot regress before a real MuJoCo smoke test is run.
"""

from __future__ import annotations

import sys
import types

import pytest

from ciro_rl.envs import dcs_wrapper


class _FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSuite(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.loaded = []

    def load(self, domain, task, **kwargs):
        self.loaded.append((domain, task, kwargs))
        return _FakeEnv(**kwargs)


@pytest.fixture
def fake_suite(monkeypatch):
    parent = types.ModuleType("distracting_control")
    parent.__path__ = []
    mod = _FakeSuite("distracting_control.suite")
    monkeypatch.setitem(sys.modules, "distracting_control", parent)
    monkeypatch.setitem(sys.modules, "distracting_control.suite", mod)
    return mod


def test_build_env_uses_suite_load_api(fake_suite):
    env = dcs_wrapper._build_env(
        "cartpole", "balance", 0,
        difficulty="medium", resolution=64, background_dataset_videos="train",
    )
    assert isinstance(env, _FakeEnv)
    assert len(fake_suite.loaded) == 1
    domain, task, kwargs = fake_suite.loaded[0]
    assert (domain, task) == ("cartpole", "balance")
    assert kwargs["difficulty"] == "medium"
    assert kwargs["dynamic"] is False
    assert kwargs["render_kwargs"] == {"height": 64, "width": 64}
    assert kwargs["task_kwargs"] == {"time_limit": 20.0}
    assert kwargs["background_dataset_videos"] == "train"


def test_build_env_validates_difficulty(fake_suite):
    with pytest.raises(ValueError):
        dcs_wrapper._build_env("cartpole", "balance", 0, difficulty="impossible")
    assert fake_suite.loaded == []


def test_background_videos_mapping():
    assert dcs_wrapper._background_videos("train") == "train"
    assert dcs_wrapper._background_videos("eval") == "val"


def test_wrapper_threads_difficulty_and_resolution(monkeypatch):
    calls = []

    def fake_build(domain, task, seed, difficulty, resolution,
                   background_dataset_videos):
        calls.append((domain, task, seed, difficulty, resolution,
                      background_dataset_videos))
        return _FakeEnv()

    monkeypatch.setattr(dcs_wrapper, "_require_deps", lambda: None)
    monkeypatch.setattr(dcs_wrapper, "_build_env", fake_build)
    w = dcs_wrapper.DCSWrapper("walker", "walk", seed=3, difficulty="hard")
    w._ensure("eval")
    assert calls == [("walker", "walk", 3, "hard", 64, "val")]


def test_wrapper_rejects_bad_difficulty(monkeypatch):
    monkeypatch.setattr(dcs_wrapper, "_require_deps", lambda: None)
    with pytest.raises(ValueError):
        dcs_wrapper.DCSWrapper("cartpole", "balance", difficulty="ludicrous")
