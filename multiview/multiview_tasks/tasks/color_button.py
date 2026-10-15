"""Task 2: Side-Panel Color Button (box with colored buttons).

A static box sits on the counter in front of the robot. One face carries three
spring-loaded buttons (red, green, blue) in a random left-to-right order. The
instruction names one color: "press the <color> button on the side of the box".

Knob: the length of a hood (a flat plate on top of the box) that overhangs the
button face. The box pose is randomized per episode (a few cm, +-10 deg yaw) from the
env rng, identically at every level for a given seed, so pressing takes the same arm
motion at every level; only what each camera sees changes. The head looks down steeply and
loses the buttons under a longer hood, while cameras lower than the hood (under-wrist
views) still see them.

Not part of the original RoboCasa GR1 tasks, so the base checkpoint has never seen
the object or the instruction; L0 is the reference point rather than a reproduction.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.variant import VariantMixin, make_level_classes

BOX_PREFIX = "mvbox_"
COLORS = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.65, 0.15, 1.0),
    "blue": (0.10, 0.25, 0.85, 1.0),
}
BOX_HALF_SIZE = (0.07, 0.10, 0.07)  # x is the button-face normal (box frame)
BOX_DISTANCE = 0.42  # box center ahead of the robot base (m)
BOX_LEFT_OFFSET = 0.06  # shift left so the right shoulder camera sees the turned face
BUTTON_RADIUS = 0.018
BUTTON_HALF_LENGTH = 0.008
BUTTON_TRAVEL = 0.012  # joint range (m)
# Joint displacement that counts as pressed (m). A pointing index finger reaches
# 6.3-6.8 mm before its intermediate link meets the box face, so 6 mm left no margin.
PRESS_THRESHOLD = 0.004
# Button spring. The Fourier index finger is weakly actuated: against 150 N/m it bent
# back at ~4-6 mm of travel and the button sprang out.
BUTTON_STIFFNESS = 40.0  # N/m
BUTTON_DAMPING = 1.0  # N s/m
# Yaw of the button face away from the robot, toward the robot's right. A face turned
# 45 deg lets the right index finger point straight at it with the waist fixed.
FACE_YAW_DEG = 45.0
# Per-episode box pose noise (uniform): offsets of the box center along the robot's
# forward and right axes (m) and of the face yaw (deg). Sampled from the env rng
# after the colors, so a seed gives the same box pose at every level.
BOX_POS_NOISE = 0.03
FACE_YAW_NOISE_DEG = 10.0
HOOD_THICKNESS = 0.006
REACH_DISTANCE = 0.03  # fingertip to target face center counted as "reached" (m)


class ColorButtonBox(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "ColorButtonBox"
    VALID_LAYOUTS = [0]
    # Hood overhang in front of the button face (m). In the 256px egoview the colors
    # are fully visible at L0, mostly at L1, partly at L2 and hidden from 3 cm on (L3).
    LEVELS = {
        0: dict(hood=0.0),
        1: dict(hood=0.01),
        2: dict(hood=0.02),
        3: dict(hood=0.04),
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        return []

    # ------------------------------------------------------------------
    # Scene: add the button box to the model XML before it is compiled.
    # ------------------------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        # Tabletop._load_model may recurse on placement failure; add the box only once.
        if self.model.worldbody.find(f"./body[@name='{BOX_PREFIX}body']") is None:
            self._add_button_box()

    def _add_button_box(self):
        if "button_colors" in self._ep_meta:
            # Replaying a recorded episode (robocasa playback calls set_ep_meta first).
            order = list(self._ep_meta["button_colors"])
            self.target_color = self._ep_meta["target_color"]
            self.box_noise = dict(self._ep_meta["box_noise"])
        else:
            order = [str(c) for c in self.rng.permutation(list(COLORS))]
            self.target_color = str(self.rng.choice(order))
            forward_offset, right_offset = self.rng.uniform(-BOX_POS_NOISE, BOX_POS_NOISE, size=2)
            self.box_noise = dict(
                forward=float(forward_offset),
                right=float(right_offset),
                yaw_deg=float(self.rng.uniform(-FACE_YAW_NOISE_DEG, FACE_YAW_NOISE_DEG)),
            )
        self.button_colors = order  # left-to-right as seen when facing the buttons

        base_pos, base_ori = self.compute_robot_base_placement_pose()
        yaw = base_ori[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        counter_top = self.counter.get_ext_sites(relative=False)[3][2]
        noise = self.box_noise
        center = (
            base_pos
            + (BOX_DISTANCE + noise["forward"]) * forward
            + (noise["right"] - BOX_LEFT_OFFSET) * right
        )
        center[2] = counter_top + BOX_HALF_SIZE[2]

        face_yaw = np.deg2rad(FACE_YAW_DEG + noise["yaw_deg"])
        normal = -np.cos(face_yaw) * forward + np.sin(face_yaw) * right
        box_yaw = np.arctan2(normal[1], normal[0])
        qx, qy, qz, qw = Rotation.from_euler("z", box_yaw).as_quat()

        body = ET.Element(
            "body",
            name=f"{BOX_PREFIX}body",
            pos=array_to_string(center),
            quat=array_to_string([qw, qx, qy, qz]),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{BOX_PREFIX}shell",
            type="box",
            size=array_to_string(BOX_HALF_SIZE),
            rgba="0.85 0.85 0.82 1",
            group="1",  # robosuite renders visual group 1; contype/conaffinity keep collisions
        )
        hx, hy, hz = BOX_HALF_SIZE
        hood = self.variant_params["hood"]
        if hood > 0:
            # Plate on the box top, from the back edge to `hood` past the button face.
            ET.SubElement(
                body,
                "geom",
                name=f"{BOX_PREFIX}hood",
                type="box",
                size=array_to_string([hx + hood / 2, hy, HOOD_THICKNESS / 2]),
                pos=array_to_string([hood / 2, 0.0, hz + HOOD_THICKNESS / 2]),
                rgba="0.55 0.55 0.58 1",
                group="1",
            )
        # Facing the buttons, the viewer's right is box +y, so left-to-right runs -y to +y.
        for color, y in zip(order, (-hy * 0.55, 0.0, hy * 0.55)):
            button = ET.SubElement(
                body, "body", name=f"{BOX_PREFIX}button_{color}", pos=array_to_string([hx, y, 0.0])
            )
            # Visual-group geoms are excluded from inertia, so give the moving body its own.
            ET.SubElement(button, "inertial", pos="0 0 0", mass="0.02", diaginertia="1e-5 1e-5 1e-5")
            ET.SubElement(
                button,
                "joint",
                name=f"{BOX_PREFIX}button_{color}_joint",
                type="slide",
                axis="-1 0 0",
                range=f"0 {BUTTON_TRAVEL}",
                limited="true",
                stiffness=str(BUTTON_STIFFNESS),
                damping=str(BUTTON_DAMPING),
            )
            ET.SubElement(
                button,
                "geom",
                name=f"{BOX_PREFIX}button_{color}_geom",
                type="cylinder",
                size=f"{BUTTON_RADIUS} {BUTTON_HALF_LENGTH}",
                pos=f"{BUTTON_HALF_LENGTH * 0.75} 0 0",
                euler="0 1.5708 0",
                rgba=array_to_string(COLORS[color]),
                group="1",
            )
        self.model.worldbody.append(body)

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        self._pressed = {color: False for color in COLORS}

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "target_color"):  # set once the box is added in _load_model
            ep_meta["lang"] = f"press the {self.target_color} button on the side of the box"
            ep_meta["button_colors"] = list(self.button_colors)
            ep_meta["target_color"] = self.target_color
            ep_meta["box_noise"] = dict(self.box_noise)
        return ep_meta

    def _button_depth(self, color):
        return float(self.sim.data.get_joint_qpos(f"{BOX_PREFIX}button_{color}_joint"))

    def _update_pressed(self):
        for color in COLORS:
            self._pressed[color] |= self._button_depth(color) > PRESS_THRESHOLD

    def _check_success(self):
        self._update_pressed()
        wrong = any(v for c, v in self._pressed.items() if c != self.target_color)
        return self._pressed[self.target_color] and not wrong

    def target_face_center(self):
        """Center of the target button's front face (world)."""
        model, data = self.sim.model, self.sim.data
        body = model.body_name2id(f"{BOX_PREFIX}button_{self.target_color}")
        normal = data.body_xmat[body].reshape(3, 3)[:, 0]
        return data.body_xpos[body] + 1.75 * BUTTON_HALF_LENGTH * normal

    def index_tip(self, side):
        """Index fingertip (world): the intermediate link mesh is centered halfway along it."""
        model, data = self.sim.model, self.sim.data
        s = side[0].upper()
        link = model.body_name2id(f"gripper0_{side}_{s}_index_intermediate_link")
        geom = model.geom_name2id(f"gripper0_{side}_{s}_index_intermediate_link_vis")
        offset = 2.0 * model.geom_pos[geom]
        return data.body_xpos[link] + data.body_xmat[link].reshape(3, 3) @ offset

    def _fingertip_to_target_distance(self):
        target = self.target_face_center()
        return min(float(np.linalg.norm(self.index_tip(side) - target)) for side in ("left", "right"))

    def phase_signals(self):
        self._update_pressed()
        return dict(
            reached=self._fingertip_to_target_distance() < REACH_DISTANCE,
            pressed_target=self._pressed[self.target_color],
            pressed_wrong=any(v for c, v in self._pressed.items() if c != self.target_color),
        )


ENV_NAMES = make_level_classes(ColorButtonBox, __name__)
