#!/usr/bin/env python3
"""Replay an accepted physical expert action trace into a dual-view video."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import cv2
import numpy as np
import omnigibson as og
import torch as th
from omnigibson.sensors import VisionSensor
from omnigibson import object_states

from run_deltasg_expert import (
    _apply_saved_initial_states,
    _create_expert_env,
    _delta_replay_integrity,
    _physical_added_object_configs,
    _primary_robot_camera,
    _step_control_without_observation,
    _to_numpy,
)


MODEL = "qwen3.8-max"
DEFAULT_VIEW_WIDTH = 960
DEFAULT_VIEW_HEIGHT = 720
DEFAULT_CANVAS_HEIGHT = 900


def _rgb(sensor, width, height):
    obs, _ = sensor.get_obs()
    rgb = _to_numpy(obs["rgb"])[..., :3].astype(np.uint8)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _global_sensor(env, run, width, height):
    cameras = [
        camera
        for camera in ((run.get("task_environment") or {}).get("camera") or run.get("camera") or [])
        if camera.get("camera_type") == "global_camera"
    ]
    if not cameras:
        raise RuntimeError("task has no validated global camera")
    camera = cameras[0]
    pose = camera.get("pose") or {}
    if not pose.get("position") or not pose.get("orientation_xyzw"):
        raise RuntimeError("global camera pose is incomplete")
    sensor = VisionSensor(
        relative_prim_path="/deltasg_video/global_camera",
        name="deltasg_video_global_camera",
        modalities=["rgb"],
        image_height=height,
        image_width=width,
    )
    sensor.load(env.scene)
    sensor.set_position_orientation(
        position=np.asarray(pose["position"], dtype=np.float32),
        orientation=np.asarray(pose["orientation_xyzw"], dtype=np.float32),
    )
    sensor.initialize()
    return camera, sensor


def _frame(robot_sensor, global_sensor, title, progress, view_width, view_height, canvas_height):
    for _ in range(2):
        og.sim.render()
    robot = _rgb(robot_sensor, view_width, view_height)
    external = _rgb(global_sensor, view_width, view_height)
    canvas_width = view_width * 2
    view_top = 70
    canvas = np.full((canvas_height, canvas_width, 3), 23, dtype=np.uint8)
    canvas[view_top : view_top + view_height, :view_width] = robot
    canvas[view_top : view_top + view_height, view_width:] = external
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "ROBOT PRIMARY CAMERA", (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, "GLOBAL CAMERA", (view_width + 18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, progress, (28, canvas_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
    return canvas


def _ffmpeg(output_path, fps, width, height):
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
            "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--expert-result", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--robot", default="R1", choices=("R1", "Tiago"))
    parser.add_argument("--action-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--view-width", type=int, default=DEFAULT_VIEW_WIDTH)
    parser.add_argument("--view-height", type=int, default=DEFAULT_VIEW_HEIGHT)
    parser.add_argument("--canvas-height", type=int, default=DEFAULT_CANVAS_HEIGHT)
    parser.add_argument("--llm-model", required=True)
    args = parser.parse_args()
    if args.llm_model != MODEL:
        parser.error(f"physical replay requires --llm-model {MODEL}")
    if args.action_stride < 1 or args.fps < 1 or args.view_width < 1 or args.view_height < 1:
        parser.error("stride, fps, and view dimensions must be positive")
    if args.canvas_height < args.view_height + 100:
        parser.error("--canvas-height must leave room for the view and labels")

    input_path = Path(args.input_json).resolve()
    result_path = Path(args.expert_result).resolve()
    output_path = Path(args.output_video).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run = json.loads(input_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    backend = result.get("backend") or {}
    te = run.get("task_environment") or {}
    generation_profile = (te.get("generation") or {}).get("solvability_profile")
    assisted = backend.get("assisted_interaction") is True
    supervision_labels_ok = assisted or (
        backend.get("physical_solubility_validation") is True
        and backend.get("low_level_vla_actions_eligible") is True
    )
    if not (
        result.get("accepted") is True
        and backend.get("name") == "physical_control"
        and generation_profile == "physical_control"
        and backend.get("generation_solvability_profile") == generation_profile
        and backend.get("generation_profile_verified") is True
        and backend.get("physical_trajectory_available") is True
        and int(backend.get("physical_action_count") or 0) > 0
        and int(backend.get("physical_nonzero_action_count") or 0) > 0
        and backend.get("complete_action_trace") is True
        and supervision_labels_ok
    ):
        raise RuntimeError("video replay requires an accepted complete physical expert result")
    if Path(str(result.get("input") or "")).resolve() != input_path:
        raise RuntimeError("expert result does not belong to the requested input JSON")
    if result.get("run_id") != run.get("run_id"):
        raise RuntimeError("expert result run_id does not match the input JSON")
    if result.get("robot") != args.robot:
        raise RuntimeError(f"expert robot is {result.get('robot')!r}, not {args.robot!r}")

    action_steps = []
    for record in result.get("steps") or []:
        action_path = Path(record.get("actions_path") or "")
        if not action_path.is_file():
            raise RuntimeError(f"missing saved actions: {action_path}")
        actions = np.load(action_path, allow_pickle=False)
        if actions.ndim != 2 or len(actions) == 0 or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"invalid saved actions: {action_path} shape={actions.shape}")
        if len(actions) != int(record.get("actions_executed") or -1):
            raise RuntimeError(f"action count mismatch: {action_path}")
        interaction = record.get("assisted_interaction")
        if interaction and (
            interaction.get("mode") != "omnigibson_assisted_state_transition"
            or interaction.get("state_set_succeeded") is not True
            or interaction.get("actual_value") != interaction.get("requested_value")
        ):
            raise RuntimeError(f"invalid assisted interaction in {action_path}")
        action_steps.append((record, actions))
    total_actions = sum(len(actions) for _, actions in action_steps)

    scene = (te.get("base_scene") or {}).get("scene_model")
    robot_pose = (te.get("robot") or {}).get("pose") or (run.get("robot") or {}).get("pose") or {}
    env = _create_expert_env(
        scene,
        args.robot,
        "physical_control",
        robot_pose=robot_pose,
        added_objects=_physical_added_object_configs(run),
        camera_resolution=(args.view_width, args.view_height),
    )
    integrity = _delta_replay_integrity(env, run, max_displacement=0.05)
    if not integrity["ok"]:
        raise RuntimeError(f"delta replay integrity failed: {integrity['objects']}")
    _apply_saved_initial_states(env, run)
    _, robot_sensor = _primary_robot_camera(env.robots[0])
    global_camera, global_sensor = _global_sensor(env, run, args.view_width, args.view_height)
    for _ in range(5):
        og.sim.render()
    robot_sensor.get_obs()
    global_sensor.get_obs()

    canvas_width = args.view_width * 2
    encoder = _ffmpeg(output_path, args.fps, canvas_width, args.canvas_height)
    frames = 0
    completed = 0
    replayed_assisted_interactions = []
    try:
        encoder.stdin.write(
            _frame(
                robot_sensor, global_sensor, "INITIAL STATE",
                f"0 / {total_actions} control actions", args.view_width,
                args.view_height, args.canvas_height,
            ).tobytes()
        )
        frames += 1
        for record, actions in action_steps:
            step = record.get("step") or {}
            primitive = str(step.get("primitive") or "ACTION")
            step_id = int(step.get("step_id") or 0)
            for index, action in enumerate(actions, 1):
                _step_control_without_observation(
                    env, th.as_tensor(action, dtype=th.float32)
                )
                completed += 1
                if completed % args.action_stride == 0 or index == len(actions):
                    title = f"STEP {step_id}: {primitive}"
                    progress = f"{completed} / {total_actions} control actions"
                    encoder.stdin.write(
                        _frame(
                            robot_sensor, global_sensor, title, progress,
                            args.view_width, args.view_height, args.canvas_height,
                        ).tobytes()
                    )
                    frames += 1
            interaction = record.get("assisted_interaction")
            if interaction:
                target = env.scene.object_registry(
                    "name", interaction.get("target_object"), None
                )
                state_type = (
                    object_states.Open
                    if primitive in {"OPEN", "CLOSE"}
                    else object_states.ToggledOn
                )
                state = target.states.get(state_type) if target is not None else None
                requested_value = bool(interaction.get("requested_value"))
                if state is None or not bool(state.set_value(requested_value)):
                    raise RuntimeError(
                        f"failed to replay official {primitive} state transition on "
                        f"{interaction.get('target_object')}"
                    )
                actual_value = bool(state.get_value())
                if actual_value != requested_value:
                    raise RuntimeError(
                        f"official {primitive} replay postcondition failed on "
                        f"{interaction.get('target_object')}"
                    )
                replayed_assisted_interactions.append(dict(interaction))
                encoder.stdin.write(
                    _frame(
                        robot_sensor,
                        global_sensor,
                        f"STEP {step_id}: {primitive} (OFFICIAL STATE)",
                        f"{completed} / {total_actions} control actions",
                        args.view_width,
                        args.view_height,
                        args.canvas_height,
                    ).tobytes()
                )
                frames += 1
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")

    metadata = {
        "schema_version": "deltasg_expert_video_replay.v1",
        "input_json": str(input_path),
        "expert_result": str(result_path),
        "source_expert_accepted": True,
        "source_backend": backend,
        "supervision_level": backend.get("supervision_level"),
        "replayed_assisted_interactions": replayed_assisted_interactions,
        "scene": scene,
        "task_name": result.get("task_name"),
        "robot": args.robot,
        "global_camera_id": global_camera.get("camera_id"),
        "total_actions": total_actions,
        "action_stride": args.action_stride,
        "fps": args.fps,
        "view_resolution": [args.view_width, args.view_height],
        "video_resolution": [canvas_width, args.canvas_height],
        "frames": frames,
        "output_video": str(output_path),
        "delta_replay_integrity": integrity,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
