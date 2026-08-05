from __future__ import annotations

from importlib import import_module
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from Prior_Recon.Masked_Flow.evaluation.errors import EvaluationInputError
from Prior_Recon.Masked_Flow.utils.assets import default_g1_mjcf_xml_path

FloatArray = NDArray[np.float64]
_FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_HAND_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_HAND_OFFSETS = np.array(
    [[0.0415, 0.0030, 0.0], [0.0415, -0.0030, 0.0]],
    dtype=np.float64,
)
_MUJOCO = import_module("mujoco")
_MUJOCO_SYMBOLS = vars(_MUJOCO)
_MJ_MODEL = _MUJOCO_SYMBOLS["MjModel"]
_MJ_DATA = _MUJOCO_SYMBOLS["MjData"]
_MJ_FORWARD = _MUJOCO_SYMBOLS["mj_forward"]
_MJ_NAME2ID = _MUJOCO_SYMBOLS["mj_name2id"]
_HINGE_TYPE = int(vars(_MUJOCO_SYMBOLS["mjtJoint"])["mjJNT_HINGE"])
_SPHERE_TYPE = int(vars(_MUJOCO_SYMBOLS["mjtGeom"])["mjGEOM_SPHERE"])
_BODY_OBJECT = vars(_MUJOCO_SYMBOLS["mjtObj"])["mjOBJ_BODY"]


class KinematicsOutput(NamedTuple):
    body_positions: FloatArray
    hand_positions: FloatArray
    hand_quaternions: FloatArray
    sole_positions: FloatArray


class G1Kinematics:
    def __init__(self) -> None:
        self._model = _MJ_MODEL.from_xml_path(str(default_g1_mjcf_xml_path()))
        self._data = _MJ_DATA(self._model)
        hinge_joint_ids = np.flatnonzero(self._model.jnt_type == _HINGE_TYPE)
        self._body_ids = self._model.jnt_bodyid[hinge_joint_ids]
        self._hand_ids = np.array(
            [self._body_id(name) for name in _HAND_NAMES],
            dtype=np.int32,
        )
        foot_ids = [self._body_id(name) for name in _FOOT_NAMES]
        sole_geom_ids = []
        for foot_id in foot_ids:
            candidates = np.flatnonzero(
                (self._model.geom_bodyid == foot_id)
                & (self._model.geom_type == _SPHERE_TYPE)
            )
            if candidates.size == 0:
                raise EvaluationInputError(
                    "g1_29dof.xml",
                    f"foot body {foot_id} has no sole marker geoms",
                )
            sole_geom_ids.append(candidates)
        if sole_geom_ids[0].size != sole_geom_ids[1].size:
            raise EvaluationInputError(
                "g1_29dof.xml",
                "left and right feet expose different sole marker counts",
            )
        self._sole_geom_ids = np.stack(sole_geom_ids)

    def _body_id(self, name: str) -> int:
        body_id = int(_MJ_NAME2ID(self._model, _BODY_OBJECT, name))
        if body_id < 0:
            raise EvaluationInputError("g1_29dof.xml", f"missing body {name}")
        return body_id

    def forward(self, qpos: FloatArray) -> KinematicsOutput:
        frame_count = qpos.shape[0]
        body_positions = np.empty(
            (frame_count, self._body_ids.size, 3), dtype=np.float64
        )
        hand_positions = np.empty((frame_count, 2, 3), dtype=np.float64)
        hand_quaternions = np.empty((frame_count, 2, 4), dtype=np.float64)
        sole_positions = np.empty(
            (frame_count, 2, self._sole_geom_ids.shape[1], 3),
            dtype=np.float64,
        )
        for frame_index, frame_qpos in enumerate(qpos):
            self._data.qpos[:] = frame_qpos
            _MJ_FORWARD(self._model, self._data)
            body_positions[frame_index] = self._data.xpos[self._body_ids]
            for hand_index, body_id in enumerate(self._hand_ids):
                rotation = self._data.xmat[body_id].reshape(3, 3)
                hand_positions[frame_index, hand_index] = (
                    self._data.xpos[body_id] + rotation @ _HAND_OFFSETS[hand_index]
                )
                hand_quaternions[frame_index, hand_index] = self._data.xquat[body_id]
            sole_positions[frame_index] = self._data.geom_xpos[self._sole_geom_ids]
        return KinematicsOutput(
            body_positions=body_positions,
            hand_positions=hand_positions,
            hand_quaternions=hand_quaternions,
            sole_positions=sole_positions,
        )
