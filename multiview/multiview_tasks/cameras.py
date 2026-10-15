"""Single source of truth for the multi-view camera rig.

Every camera used by the multi-view tasks is defined here: where it is mounted,
its pose relative to the parent body, its field of view, how its render is framed
into the 256x256 observation, and the observation key it is exposed under. Env
construction, data export and (later) the GR00T modality config should all read
from this module.

All poses match the RoboCasa-GR1 "Multiview-15views" camera table (checked by
tests/test_tasks.py::test_camera_poses_match_release), which was defined against
robocasa-gr1-tabletop-tasks 4840e67, robosuite 1.5.1 and mujoco 3.2.6. The mounted
cameras are its wrist-under and shoulder cameras (multiview_cameras.py EXTRA_CAMERAS);
egoview and robotview come with robocasa and robosuite. Observation keys follow that
dataset's stream names.
"""

from dataclasses import dataclass
from typing import Optional

# Image size rendered by robosuite for every camera. The GR1 checkpoint expects
# egoview at 1280x800, and RoboCasaEnv only supports a single size for all cameras.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800


@dataclass(frozen=True)
class CameraSpec:
    name: str  # MuJoCo camera name (also the robosuite "<name>_image" obs prefix)
    obs_key: str  # GR00T observation key
    parent_body: Optional[str] = None  # None: camera already defined by robocasa
    pos: tuple = ()  # metres, in the parent body frame
    quat: tuple = ()  # wxyz, in the parent body frame
    fovy: float = 0.0  # vertical field of view (degrees)
    # "pad": robocasa process_img (letterbox the full frame to a square).
    # "square": centre 800x800 crop of the 1280x800 render, INTER_AREA resize to 256x256.
    framing: str = "square"


# Defined by robocasa's Tabletop.set_cameras (mounted on robot0_head_pitch, fovy 77).
# The base GR1 checkpoint was trained on this key and framing.
EGOVIEW = CameraSpec(name="egoview", obs_key="video.ego_view_pad_res256_freq20", framing="pad")

# 5 cm below the wrist, aimed at the fingertips (sees the grasp from underneath).
# Left and right are exact sagittal mirrors. GR1's left hand body really is "..._eefd".
LEFT_WRIST_UNDER = CameraSpec(
    name="left_wrist_under",
    obs_key="video.left_wrist_under_view",
    parent_body="robot0_left_hand_eefd",
    pos=(-0.0221, 0.0411, 0.0109),
    quat=(-0.3277, 0.2035, 0.1928, 0.9022),
    fovy=90.0,
)
RIGHT_WRIST_UNDER = CameraSpec(
    name="right_wrist_under",
    obs_key="video.right_wrist_under_view",
    parent_body="robot0_right_hand_eef",
    pos=(-0.0221, -0.0411, 0.0109),
    quat=(0.9022, 0.1928, 0.2035, -0.3277),
    fovy=90.0,
)

# On the shoulder brackets, rigid to the torso (moves with the waist, not the arm).
LEFT_SHOULDER = CameraSpec(
    name="shoulder_left",
    obs_key="video.shoulder_left",
    parent_body="robot0_left_arm_mount",
    pos=(0.1002, 0.0847, 0.1811),
    quat=(0.7051, 0.0539, -0.3468, -0.6162),
    fovy=77.0,
)
RIGHT_SHOULDER = CameraSpec(
    name="shoulder_right",
    obs_key="video.shoulder_right",
    parent_body="robot0_right_arm_mount",
    pos=(0.1002, -0.0843, 0.1813),
    quat=(0.6162, 0.3468, -0.0539, -0.7050),
    fovy=77.0,
)

# Defined by robosuite's GR1 robot.xml on robot0_head_pitch: pos (0.13, 0, 0.09),
# xyaxes "0 -1 0 0.9 0 0.6", fovy 50. A narrower head camera than egoview.
ROBOTVIEW = CameraSpec(name="robot0_robotview", obs_key="video.robotview")

# Cameras added to every multi-view scene, regardless of which ones are observed.
MOUNTED_CAMERAS = (LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER)

# Which cameras are rendered and returned as observations. The scene is the same for all.
# The egoview spec yields both ego framings (pad, and bg crop added by GrootRoboCasaEnv).
CAMERA_PROFILES = {
    "cam1": (EGOVIEW,),
    "cam3": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER),
    "cam5": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER),
    # Everything stored in demonstration datasets.
    "collect": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER),
}
