"""Thin wrapper around `distracting_control` (PyPI) over `dm_control`.

Exposes a gym-like interface whose `reset`/`step` take an explicit
`distractor_pool` argument so DTG's train/eval distractor split is a
first-class, explicit choice rather than something hacked in later.

Environment gate. `distracting_control` requires MuJoCo and a rendering backend
(OpenGL/EGL). On a default Colab CPU runtime these are absent, and on
Windows/WSL they need manual MuJoCo + EGL/X setup. We lazy-import the
third-party stack inside `is_available()`/the constructor and raise a clear,
actionable :class:`DCSError` when it is missing, so
``experiments/run_table1.py`` can skip D_TG datasets gracefully instead of
crashing with a cryptic import error.

Resolution default. We default to ``resolution=64``, ``frame_stack=1``, which is
deliberately below the Distracting Control Suite's own defaults (100x100, frame
stack 3) to stay tractable on a free-tier T4. This is the same kind of explicit,
compute-driven deviation the CIR paper documented for Waterbirds' resolution.

License note. ``distracting_control`` and ``dm_control`` are Apache-2.0. Verify
their upstream LICENSE files before publishing this repo publicly; see README.
"""

from typing import Tuple

import numpy as np

RESOLUTION_DEFAULT = 64
FRAME_STACK_DEFAULT = 1


class DCSError(RuntimeError):
    """Raised when ``dm_control``/``distracting_control`` is unavailable."""


def _require_deps():
    try:
        import dm_control  # noqa: F401
        import distracting_control  # noqa: F401
    except ImportError as e:
        raise DCSError(
            "DCS environments need `dm-control` and `distracting-control`. They are "
            "OPTIONAL dependencies (require MuJoCo + a rendering backend; see "
            "scripts/setup_colab.sh). Import failed: " + str(e)
        ) from e


def is_available() -> bool:
    """True iff the Distracting Control Suite stack can be imported."""
    try:
        _require_deps()
        return True
    except DCSError:
        return False


def _set_pool(store, pool):
    """Best-effort selection of the DCS distractor pool (version-dependent API)."""
    try:
        store.distractor_pool = pool  # many versions expose an attribute
    except Exception:  # pragma: no cover - env-gated
        pass
    return store


def _build_env(domain, task, seed):
    """Lazily build a wrapped distractor env (pool set later by reset())."""
    import dm_control.suite as suite
    from distracting_control.suite import DistractingControlEnv

    base = suite.load(
        domain,
        task,
        task_kwargs={"time_limit": 20.0},
        visualize_reward=False,
        from_pixels=True,
    )
    # NOTE: the exact DistractingControlEnv constructor signature is version-
    # dependent and has not been smoke-tested here (no dm_control/MuJoCo in the
    # dev environment). If the published API differs, adjust this one call.
    pool_env = DistractingControlEnv(
        base_env=base,
        dynamic_distraction=False,
    )
    return pool_env


class DCSWrapper:
    """gym-like wrapper over a Distracting Control Suite task.

    Args:
        domain: 'cartpole' | 'walker'
        task: 'balance' | 'walk'
        resolution: square pixel side length (default 64).
        frame_stack: frames stacked (default 1).
        seed: base seed.
    """

    def __init__(
        self,
        domain: str,
        task: str,
        resolution: int = RESOLUTION_DEFAULT,
        frame_stack: int = FRAME_STACK_DEFAULT,
        seed: int = 0,
    ):
        _require_deps()
        self.domain = domain
        self.task = task
        self.resolution = resolution
        self.frame_stack = frame_stack
        self.seed = seed
        self._store = None
        self._pool = None
        self.action_dim = 1  # overwritten after env build from the action spec

    def _ensure(self, distractor_pool: str):
        # Rebuild iff the pool changes (train vs eval distractor pool switch).
        if self._store is None or self._pool != distractor_pool:
            self._store = _build_env(self.domain, self.task, self.seed)
            self._store = _set_pool(self._store, distractor_pool)
            try:
                spec = self._store.action_spec()
                self.action_dim = spec.shape[-1]
            except Exception:  # pragma: no cover - env-gated, best-effort
                self.action_dim = 1
            self._pool = distractor_pool

    def reset(self, distractor_pool: str = "train") -> np.ndarray:
        """Reset into the given distractor pool; returns the first frame (H,W,3)."""
        _require_deps()
        if distractor_pool not in {"train", "eval"}:
            raise ValueError("distractor_pool must be 'train' or 'eval'")
        self._ensure(distractor_pool)
        obs = self._store.reset()
        return self._to_array(obs)

    def step(self, action):
        """Apply `action`, return (obs, reward, terminated, info) gym-style."""
        obs, reward, done, info = self._store.step(action)
        return self._to_array(obs), float(reward), bool(done), info

    def _to_array(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 4:  # stacked TimeLimit => take last frame
            arr = arr[..., -1]
        if arr.shape[0] != self.resolution or arr.shape[1] != self.resolution:
            arr = _resize(arr, self.resolution)
        # ensure channel-last 3-channel (H,W,3)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        elif arr.shape[0] == 3 or (arr.shape[0] == 1 and arr.ndim == 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:
            arr = np.concatenate([arr] * 3, axis=-1)
        return arr.astype(np.float32)


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour square resize of a 2D or 3D array (no cv2 dependency)."""
    h, w = arr.shape[-2], arr.shape[-1]
    yi = np.linspace(0, h - 1, size).astype(int)
    xi = np.linspace(0, w - 1, size).astype(int)
    if arr.ndim == 3:
        return arr[yi][:, xi]
    return arr[yi][:, xi]


def sample_pool_frames(wrapper, pool: str, n: int, action_seed: int = 0):
    """Roll `n` frames from a distractor pool.

    Returns (frames (n,H,W,3), rewards (n,)) as np arrays. Samples by stepping
    the env deterministically; for n frames this is n environment steps.
    """
    n = max(int(n), 1)
    rng = np.random.RandomState(action_seed)
    frames = []
    rewards = []
    obs = wrapper.reset(distractor_pool=pool)
    frames.append(obs)
    rewards.append(0.0)
    while len(frames) < n:
        action = rng.uniform(-1.0, 1.0, size=wrapper.action_dim)
        obs, rw, done, _ = wrapper.step(action)
        frames.append(obs)
        rewards.append(rw)
        if done:
            obs = wrapper.reset(distractor_pool=pool)
            frames.append(obs)
            rewards.append(0.0)
    return np.asarray(frames)[:n], np.asarray(rewards)[:n]