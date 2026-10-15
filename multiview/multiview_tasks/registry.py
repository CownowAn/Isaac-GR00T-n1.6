"""Gym registration for multi-view task variants.

IDs look like ``gr1_mv/<Task>_L<level>_<profile>``, e.g.
``gr1_mv/BlindInsertionBottleCabinet_L2_cam3``. Encoding level and camera profile
in the ID lets gr00t/eval/rollout_policy.py run any variant through --env_name.
"""

import cv2
import numpy as np
from gymnasium.envs.registration import register
from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv

from multiview_tasks.cameras import CAMERA_HEIGHT, CAMERA_PROFILES, CAMERA_WIDTH

ENV_PREFIX = "gr1_mv"
ROBOT = "GR1ArmsAndWaistFourierHands"


def square_frame(img: np.ndarray) -> np.ndarray:
    """Centre square crop (full height) resized to 256x256: no padding, no distortion."""
    h, w, _ = img.shape
    x0 = (w - h) // 2
    return cv2.resize(img[:, x0 : x0 + h], (256, 256), interpolation=cv2.INTER_AREA)


def _with_camera_profile(key_converter: type, profile: str) -> type:
    """Return a key converter whose camera config is the given profile."""
    cameras = CAMERA_PROFILES[profile]

    def get_camera_config(cls):
        return (
            [cam.obs_key for cam in cameras],
            [cam.name for cam in cameras],
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )

    return type(
        f"{key_converter.__name__}_{profile}",
        (key_converter,),
        {"get_camera_config": classmethod(get_camera_config)},
    )


class MultiViewGrootEnv(GrootRoboCasaEnv):
    """GrootRoboCasaEnv that renders and returns the cameras of a camera profile.

    RoboCasaEnv reads the camera list from ``self.key_converter`` while it builds the
    robosuite env, so the profile is applied when the converter is assigned.
    """

    def __init__(self, camera_profile: str = "cam1", **kwargs):
        if camera_profile not in CAMERA_PROFILES:
            raise ValueError(
                f"Unknown camera profile {camera_profile!r}; "
                f"expected one of {list(CAMERA_PROFILES)}"
            )
        self.camera_profile = camera_profile
        super().__init__(**kwargs)

    @property
    def key_converter(self):
        return self._key_converter

    @key_converter.setter
    def key_converter(self, converter):
        self._key_converter = _with_camera_profile(converter, self.camera_profile)

    def get_groot_observation(self, raw_obs):
        obs = super().get_groot_observation(raw_obs)
        # The parent letterboxes every camera; re-frame the "square" ones.
        for cam in CAMERA_PROFILES[self.camera_profile]:
            if cam.framing == "square":
                obs[cam.obs_key] = square_frame(raw_obs[f"{cam.name}_image"])
        return obs

    def _episode_signals(self) -> dict:
        # Each task variant defines its own flags (see e.g. BlindInsertionBottleCabinet):
        # phase_signals() for task progress, diagnostic_signals() for failure analysis.
        signals = {}
        for prefix, method in (("phase", "phase_signals"), ("diag", "diagnostic_signals")):
            if hasattr(self.env, method):
                for name, value in getattr(self.env, method)().items():
                    signals[f"{prefix}/{name}"] = bool(value)
        return signals

    def _add_phase_info(self, info: dict) -> None:
        """Report each flag as reached-at-any-point-this-episode (``phase/*``, ``diag/*``)."""
        for key, value in self._episode_signals().items():
            self._phases_reached[key] = self._phases_reached.get(key, False) or value
        info.update(self._phases_reached)

    def reset(self, seed=None, options=None):
        if seed is not None:
            # robocasa samples layout, style and placements from the robosuite env rng, which
            # RoboCasaEnv never seeds. Reseeding it pairs episodes across levels and profiles.
            self.env.rng = np.random.default_rng(seed)
        obs, info = super().reset(seed=seed, options=options)
        self._phases_reached = {}
        self._add_phase_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._add_phase_info(info)
        return obs, reward, terminated, truncated, info


def env_id(env_name: str, profile: str) -> str:
    return f"{ENV_PREFIX}/{env_name}_{profile}"


def register_variants(env_names: dict) -> list:
    """Register one gym ID per (level, camera profile). Returns the registered IDs."""
    ids = []
    for env_name in env_names.values():
        for profile in CAMERA_PROFILES:
            gym_id = env_id(env_name, profile)
            register(
                id=gym_id,
                entry_point="multiview_tasks.registry:MultiViewGrootEnv",
                kwargs=dict(env_name=env_name, robots_name=ROBOT, camera_profile=profile),
            )
            ids.append(gym_id)
    return ids
