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

from typing import Optional

import numpy as np

RESOLUTION_DEFAULT = 64
FRAME_STACK_DEFAULT = 1
DIFFICULTY_DEFAULT = "medium"
VALID_DIFFICULTIES = ("easy", "medium", "hard")


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


def _background_videos(pool: str) -> str:
    """Map a DCS train/eval pool name to a DAVIS video split for the background."""
    return "train" if pool == "train" else "val"


def _build_env(
    domain: str,
    task: str,
    seed: int,
    difficulty: str = DIFFICULTY_DEFAULT,
    resolution: Optional[int] = RESOLUTION_DEFAULT,
    background_dataset_videos: Optional[str] = None,
):
    """Build a distracting environment via ``distracting_control.suite.load``.

    This is the crate's real public API (courtesy of google-research /
    ``sahandrez/distracting_control``): ``suite.load`` wraps a dm_control
    environment with background/camera/colour distractor wrappers and a pixel
    observation wrapper. There is **no** ``DistractingControlEnv`` class to
    construct; the previously-assumed constructor did not exist and import
    would raise ``ImportError``, so DCS environments were never buildable.
    ``difficulty`` selects the DCS distractor pool; ``resolution`` is threaded
    through ``render_kwargs`` so pixels come out at the requested size; and
    ``background_dataset_videos`` makes the DTG train/eval split a real
    distractor change rather than a no-op re-tag.
    """
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            "dcs_difficulty must be one of %s, got %r"
            % (VALID_DIFFICULTIES, difficulty)
        )
    import distracting_control.suite as distractors

    render_kwargs: dict = {}
    if resolution is not None:
        render_kwargs["height"] = int(resolution)
        render_kwargs["width"] = int(resolution)

    return distractors.load(
        domain,
        task,
        difficulty=difficulty,
        dynamic=False,
        task_kwargs={"time_limit": 20.0},
        render_kwargs=render_kwargs or None,
        background_dataset_videos=background_dataset_videos,
    )


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
        difficulty: str = DIFFICULTY_DEFAULT,
    ):
        _require_deps()
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                "dcs_difficulty must be one of %s, got %r"
                % (VALID_DIFFICULTIES, difficulty)
            )
        self.domain = domain
        self.task = task
        self.resolution = resolution
        self.frame_stack = frame_stack
        self.seed = seed
        self.difficulty = difficulty
        self._env = None
        self._env_pool = None
        self.action_dim = 1  # overwritten after env build from the action spec

    def _ensure(self, distractor_pool: str):
        # Rebuild iff the pool changes (train vs eval distractor pool switch);
        # the background video split is baked in at build time.
        if self._env is None or self._env_pool != distractor_pool:
            self._env = _build_env(
                self.domain,
                self.task,
                self.seed,
                difficulty=self.difficulty,
                resolution=self.resolution,
                background_dataset_videos=_background_videos(distractor_pool),
            )
            try:
                spec = self._env.action_spec()
                self.action_dim = spec.shape[-1]
            except Exception:  # pragma: no cover - env-gated, best-effort
                self.action_dim = 1
            self._env_pool = distractor_pool

    def reset(self, distractor_pool: str = "train") -> np.ndarray:
        """Reset into the given distractor pool; returns the first frame (H,W,3)."""
        _require_deps()
        if distractor_pool not in {"train", "eval"}:
            raise ValueError("distractor_pool must be 'train' or 'eval'")
        self._ensure(distractor_pool)
        ts = self._env.reset()
        return self._to_array(ts.observation)

    def step(self, action):
        """Apply `action`, return (obs, reward, terminated, info) gym-style."""
        ts = self._env.step(action)
        obs = self._to_array(ts.observation)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = bool(
            ts.step_type == ts.step_type.LAST
            or (ts.discount is not None and float(ts.discount) == 0.0)
        )
        return obs, reward, terminated, {}

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