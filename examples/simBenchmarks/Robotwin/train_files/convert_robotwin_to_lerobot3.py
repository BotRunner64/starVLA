#!/usr/bin/env python3
"""Convert RoboTwin aloha_agilex HDF5 demonstrations to LeRobot v3.

Expected input layout::

    <input-root>/<task>/aloha_agilex/data/episode_*.hdf5

The output is one StarVLA-compatible dataset per task::

    <output-root>/Clean/<task>/{data,videos,meta}

State and action use RoboTwin's explicit 14-D joint vectors in this order:
left arm (6), left gripper (1), right arm (6), right gripper (1).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import cv2
    import h5py
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - depends on the conversion environment
    raise SystemExit(
        f"Missing conversion dependency: {exc.name}. "
        "Install h5py opencv-python pandas pyarrow numpy."
    ) from exc


CAMERAS = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}
VECTOR_ORDER = (
    ("left_arm_joint_states", 6),
    ("left_ee_joint_states", 1),
    ("right_arm_joint_states", 6),
    ("right_ee_joint_states", 1),
)


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _instruction(h5_file: h5py.File, task_name: str) -> str:
    if "instruction" in h5_file:
        text = _decode_text(h5_file["instruction"][()]).strip()
        if text:
            return text
    if "instructions" in h5_file:
        for value in h5_file["instructions"][()]:
            text = _decode_text(value).strip()
            if text:
                return text
    return task_name.replace("_", " ")


def _read_vector(h5_file: h5py.File, group_name: str) -> np.ndarray:
    group = h5_file[group_name]
    parts = []
    for key, expected_dim in VECTOR_ORDER:
        if key not in group:
            raise KeyError(f"Missing HDF5 dataset: {group_name}/{key}")
        part = np.asarray(group[key], dtype=np.float32)
        if part.ndim == 1:
            part = part[:, None]
        if part.ndim != 2 or part.shape[1] != expected_dim:
            raise ValueError(
                f"{group_name}/{key} must have shape [T,{expected_dim}], got {part.shape}"
            )
        parts.append(part)
    lengths = {len(part) for part in parts}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched {group_name} trajectory lengths: {sorted(lengths)}")
    vector = np.concatenate(parts, axis=1)

    # RoboTwin also stores the packed vector. Verify our documented ordering.
    if "joint_states" in group:
        packed = np.asarray(group["joint_states"], dtype=np.float32)
        if packed.shape != vector.shape or not np.allclose(packed, vector, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"{group_name}/joint_states does not match the documented "
                "left-arm, left-gripper, right-arm, right-gripper ordering"
            )
    return vector


def _decode_image(value: Any) -> np.ndarray:
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode a RoboTwin camera frame")
    return image


def _write_camera_video(
    frames: h5py.Dataset,
    output_path: Path,
    length: int,
    fps: int,
) -> tuple[int, int]:
    first = _decode_image(frames[0])
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")
    try:
        for frame_index in range(length):
            image = first if frame_index == 0 else _decode_image(frames[frame_index])
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"Camera resolution changed at frame {frame_index}: "
                    f"{image.shape[:2]} != {(height, width)}"
                )
            writer.write(image)
    finally:
        writer.release()
    return height, width


def _vector_feature() -> dict[str, Any]:
    names = [
        *[f"left_joint_{i}" for i in range(6)],
        "left_gripper",
        *[f"right_joint_{i}" for i in range(6)],
        "right_gripper",
    ]
    return {"dtype": "float32", "shape": [14], "names": [names]}


def _video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [3, height, width],
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mpeg4",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _write_metadata(
    output_dir: Path,
    episodes: list[dict[str, Any]],
    tasks: dict[str, int],
    total_frames: int,
    fps: int,
    image_sizes: dict[str, tuple[int, int]],
) -> None:
    meta_dir = output_dir / "meta"
    episode_dir = meta_dir / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(episodes).to_parquet(episode_dir / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": list(tasks.values())}, index=list(tasks)).to_parquet(
        meta_dir / "tasks.parquet"
    )

    modality = {
        "state": {
            "left_joints": {"start": 0, "end": 6, "absolute": True, "dtype": "float32", "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "absolute": True, "dtype": "float32", "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "absolute": True, "dtype": "float32", "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "absolute": True, "dtype": "float32", "original_key": "observation.state"},
        },
        "action": {
            "left_joints": {"start": 0, "end": 6, "absolute": True, "dtype": "float32", "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "absolute": True, "dtype": "float32", "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "absolute": True, "dtype": "float32", "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "absolute": True, "dtype": "float32", "original_key": "action"},
        },
        "video": {
            "cam_high": {"original_key": "observation.images.cam_high"},
            "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
            "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    features: dict[str, Any] = {
        "observation.state": _vector_feature(),
        "action": _vector_feature(),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key, (height, width) in image_sizes.items():
        features[video_key] = _video_feature(height, width, fps)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha_agilex",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(CAMERAS),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def convert_task(args: argparse.Namespace, task_dir: Path) -> Path:
    task_name = task_dir.name
    source_dir = task_dir / args.embodiment / "data"
    episode_files = sorted(source_dir.glob("episode_*.hdf5"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {source_dir}")

    output_dir = args.output_root.resolve() / args.split / task_name
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    tasks: dict[str, int] = {}
    total_frames = 0
    common_fps: int | None = args.fps
    image_sizes: dict[str, tuple[int, int]] = {}

    for episode_index, episode_path in enumerate(episode_files):
        print(f"[{task_name}] {episode_index + 1}/{len(episode_files)} {episode_path.name}")
        with h5py.File(episode_path, "r") as h5_file:
            state = _read_vector(h5_file, "state")
            action = _read_vector(h5_file, "action")
            if state.shape != action.shape:
                raise ValueError(f"state/action shape mismatch in {episode_path}: {state.shape} vs {action.shape}")
            length = len(state)
            source_fps = int(np.asarray(h5_file["additional_info/frequency"]).item())
            fps = args.fps or source_fps
            if common_fps is None:
                common_fps = fps
            if fps != common_fps:
                raise ValueError(f"Mixed FPS is unsupported: {fps} vs {common_fps}")

            instruction = _instruction(h5_file, task_name)
            task_index = tasks.setdefault(instruction, len(tasks))
            video_meta = {}
            for video_key, source_camera in CAMERAS.items():
                camera_path = f"vision/{source_camera}/colors"
                if camera_path not in h5_file:
                    raise KeyError(f"Missing HDF5 camera: {camera_path}")
                frames = h5_file[camera_path]
                if len(frames) < length:
                    raise ValueError(f"Camera {source_camera} has {len(frames)} frames, expected {length}")
                video_path = output_dir / "videos" / video_key / "chunk-000" / f"file-{episode_index:03d}.mp4"
                size = _write_camera_video(frames, video_path, length, fps)
                previous_size = image_sizes.setdefault(video_key, size)
                if previous_size != size:
                    raise ValueError(f"Camera {video_key} resolution changed: {size} vs {previous_size}")
                video_meta.update({
                    f"videos/{video_key}/from_timestamp": 0.0,
                    f"videos/{video_key}/chunk_index": 0,
                    f"videos/{video_key}/file_index": episode_index,
                })

            for frame_index in range(length):
                rows.append({
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": np.float32(frame_index / float(fps)),
                    "task_index": task_index,
                    "index": total_frames + frame_index,
                    "observation.state": state[frame_index],
                    "action": action[frame_index],
                })
            episode_row = {
                "episode_index": episode_index,
                "length": length,
                "tasks": [instruction],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "data/file_from_index": total_frames,
                "data/file_to_index": total_frames + length,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + length,
            }
            episode_row.update(video_meta)
            episode_rows.append(episode_row)
            total_frames += length

    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet", index=False)
    assert common_fps is not None
    _write_metadata(output_dir, episode_rows, tasks, total_frames, common_fps, image_sizes)
    print(f"Wrote {len(episode_rows)} episodes / {total_frames} frames to {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="RoboTwin demo_clean directory")
    parser.add_argument("--output-root", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--task", action="append", help="Task name; repeat to convert multiple tasks (default: all)")
    parser.add_argument("--embodiment", default="aloha_agilex")
    parser.add_argument("--split", default="Clean")
    parser.add_argument("--fps", type=int, help="Override HDF5 additional_info/frequency")
    parser.add_argument("--max-episodes", type=int, help="Convert only the first N episodes")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise SystemExit("--max-episodes must be positive")
    input_root = args.input_root.resolve()
    task_names = args.task or sorted(path.name for path in input_root.iterdir() if path.is_dir())
    for task_name in task_names:
        convert_task(args, input_root / task_name)


if __name__ == "__main__":
    main()
