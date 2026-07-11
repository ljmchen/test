#!/usr/bin/env python3
"""Predictions JSON -> pregrasp/squeeze -> per-task_type dgbench-pami graspdata.

Dispatches each record by its ``task_type`` into four channels and writes one
graspdata directory per channel, named ``<output_root>/<exp_name>_<channel>_
<hand_suffix>`` to align with the downstream eval's
``save_dir=<save_root>/<exp_name>_<hand_name>``:

  channel   hands                   payload      hand_suffix   bench task
  left      pred_left               single-hand  shadow_left   eval
  right     pred_right              single-hand  shadow        eval
  lgbidex   pred_left + pred_right  bimanual     shadow_left   eval3
  bidex     pred_left + pred_right  bimanual     shadow_left   eval4

Pose conversion core (verified 2026-07, GT baseline ~91%): per-hand 28-d
``[trans(3), axis_angle(3), joints(22)]`` -> 29-d bench qpos
``[trans(3), quat wxyz(4), joints(22)]``. The ONLY change is
axis-angle -> wxyz quaternion; translation and the 22 joints pass through
UNCHANGED — no sign flip, no palm offset. Pregrasp/squeeze open/close only the
Shadow FLEX joints (+/-0.1). The bidex channel additionally retreats the
pregrasp palm 1 cm along the hand-local +Y (back of hand); the mirrored
squeeze then presses 1 cm along the palm normal into the object.

Usage:
  python prepare_bench_data.py \\
    --input outputs_v3/predictions_test.json \\
    --mesh-root /home/jiaxuan/data/oakink_obj/processed_data \\
    --output-root /home/jiaxuan/slai/dgbench-pami/output \\
    --exp-name dexvlg_v3 --channels left,right,lgbidex,bidex --clean --numworker 16
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np

SINGLE_HAND_DIM = 28

CHANNELS = ("left", "right", "lgbidex", "bidex")
CHANNEL_SIDES = {
    "left": ("left",),
    "right": ("right",),
    "lgbidex": ("left", "right"),
    "bidex": ("left", "right"),
}
CHANNEL_HAND_SUFFIX = {
    "left": "shadow_left",
    "right": "shadow",
    "lgbidex": "shadow_left",
    "bidex": "shadow_left",
}
# bidex only: pregrasp retreats 1 cm along hand-local +Y (back of hand); the
# mirrored squeeze then presses 1 cm along the palm normal R@[0,-1,0].
BIDEX_MOVE_LOCAL = np.array([0.0, 0.01, 0.0], dtype=np.float64)
NO_MOVE_LOCAL = np.zeros(3, dtype=np.float64)

METADATA_KEYS = (
    "obj_id", "obj_key", "oakink_obj_id", "dexgys_obj_id", "cate_id", "pose_id",
    "scale_id", "action_id", "guidance", "guidence", "contact_area", "grasp_id",
    "index_in_obj", "candidate_idx", "seq_id", "task_type",
)


# ───────────────────────── pre / squeeze (get_pre_squ) ─────────────────────


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def flatten_pose(value, name: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    while pose.ndim > 1 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.shape != (SINGLE_HAND_DIM,):
        raise ValueError(f"{name} must be 28D, got shape {pose.shape}.")
    return pose


def compute_pregrasp(grasp_pose: np.ndarray, move_local: np.ndarray) -> np.ndarray:
    """Open the Shadow FLEX joints by 0.1 and optionally retreat the palm.

    Args:
        grasp_pose: 28-d dex pose ``[trans(3), axis_angle(3), joints(22)]``.
        move_local: Hand-local palm translation, applied as
            ``pre[:3] = grasp[:3] + R @ move_local`` with R from the grasp
            axis-angle. ``NO_MOVE_LOCAL`` for left/right/lgbidex;
            ``BIDEX_MOVE_LOCAL`` (+1 cm along local +Y = back of hand) for
            bidex so the mirrored squeeze presses 1 cm along the palm normal.

    Returns:
        28-d pregrasp pose.
    """
    grasp_pose = flatten_pose(grasp_pose, "grasp_pose")
    pre = grasp_pose.copy()
    rot_mat = axis_angle_to_matrix(grasp_pose[3:6])
    pre[:3] = grasp_pose[:3] + rot_mat @ np.asarray(move_local, dtype=np.float64)
    # pregrasp = grasp - CLOSE_SIGN * 0.1，仅作用在 Shadow 的弯曲(FLEX)关节上。
    # 28维索引 = 6 + joint_idx。CLOSE_SIGN：
    #   四指弯曲 joints{1,2,3,5,6,7,9,10,11,14,15,16} = +1 (张开 → pre 减 0.1)
    #   拇指 THJ1(j20)/THJ0(j21) = -1 (robot range 偏负，张开 → pre 加 0.1)
    #   侧摆/扭转/掌骨(LFJ4 j12)/THJ4(j17)/THJ3(j18)/THJ2(j19 拇指侧摆) 不参与 = 0
    pre[7:10] -= 0.1    # FFJ2,FFJ1,FFJ0  (j1,2,3)
    pre[11:14] -= 0.1   # MFJ2,MFJ1,MFJ0  (j5,6,7)
    pre[15:18] -= 0.1   # RFJ2,RFJ1,RFJ0  (j9,10,11)
    pre[20:23] -= 0.1   # LFJ2,LFJ1,LFJ0  (j14,15,16)
    pre[26:28] += 0.1   # THJ1,THJ0       (j20,21) —— 不含 THJ2(j19,拇指侧摆)
    return pre


def compute_squeeze(grasp_pose: np.ndarray, pre_pose: np.ndarray) -> np.ndarray:
    grasp_pose = flatten_pose(grasp_pose, "grasp_pose")
    pre_pose = flatten_pose(pre_pose, "pre_pose")
    squ = grasp_pose.copy()
    squ[:3] = 2.0 * grasp_pose[:3] - pre_pose[:3]
    squ[6:] = 2.0 * grasp_pose[6:] - pre_pose[6:]
    return squ


def add_pre_squeeze(record: dict, idx: int, channels: set[str]) -> str | None:
    """Compute pre/squeeze for the record's ``task_type`` channel, in place.

    The channel decides which predicted hands are required (left/right = one
    hand, lgbidex/bidex = both) and whether the pregrasp palm retreats
    (bidex only). Records with a ``task_type`` outside ``channels`` or missing
    a required ``pred_*`` hand are skipped (not an error).

    Args:
        record: Prediction record (mutated in place on success).
        idx: Record index, for error messages.
        channels: Enabled channel names (subset of ``CHANNELS``).

    Returns:
        None on success (record now has ``dex_{grasp,pregrasp,squeeze}_<side>``
        for each required side and no ``pred_*``); otherwise a skip-reason
        string (``"task_type"`` or ``"missing_hand"``).
    """
    task_type = str(record.get("task_type", ""))
    if task_type not in channels:
        return "task_type"
    sides = CHANNEL_SIDES[task_type]
    if any(record.get(f"pred_{side}") is None for side in sides):
        return "missing_hand"
    move_local = BIDEX_MOVE_LOCAL if task_type == "bidex" else NO_MOVE_LOCAL
    for side in sides:
        grasp = flatten_pose(record[f"pred_{side}"], f"record[{idx}].pred_{side}")
        pre = compute_pregrasp(grasp, move_local)
        record[f"dex_grasp_{side}"] = grasp.tolist()
        record[f"dex_pregrasp_{side}"] = pre.tolist()
        record[f"dex_squeeze_{side}"] = compute_squeeze(grasp, pre).tolist()
    record.pop("pred_left", None)
    record.pop("pred_right", None)
    return None


# ──────────────────────── bench qpos (prepare_eval3) ───────────────────────


def axis_angle_to_quaternion_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis_angle / angle
    half = 0.5 * angle
    return np.concatenate(
        [np.array([np.cos(half)], dtype=np.float32), axis * np.sin(half)]
    ).astype(np.float32)


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def gys_pose_to_bench_qpos(value: Iterable[float], name: str) -> np.ndarray:
    """28-D dex pose [trans, axis_angle, 22 joints] -> 29-D bench qpos
    [trans, quat wxyz, 22 joints].

    The v1 (robot0/mjcf) data uses the SAME joint convention as eval3, so the
    ONLY change is axis-angle -> wxyz quaternion; translation and the 22 joints
    pass through UNCHANGED — no sign flip, no palm offset. The previous version
    (copied from the lgbidex/GYS `prepare_lgbidex_eval3.py`) flipped joints
    {6,10,-2,-1} and subtracted a palm offset, which targets the OLDER GYS
    convention and CORRUPTS v1 data (GT baseline scored 0%); dexter's verified
    `dex_pose_to_qpos` uses no flip and scores ~75% on GT.
    """
    pose = np.asarray(value, dtype=np.float32)
    if pose.ndim != 1:
        pose = np.squeeze(pose)
    if pose.shape != (28,):
        raise ValueError(f"{name} must be 28D dex pose, got {pose.shape}.")
    pose = pose.copy()
    quaternion = axis_angle_to_quaternion_wxyz(pose[3:6])
    qpos = np.concatenate([pose[:3].astype(np.float32), quaternion, pose[6:]], axis=0)
    if qpos.shape != (29,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"{name} produced invalid qpos {qpos.shape}.")
    return qpos.astype(np.float32)


def resolve_object_path(mesh_root: str | Path, object_id: str) -> Path:
    object_path = Path(mesh_root) / object_id
    required = [
        object_path / "info" / "simplified.json",
        object_path / "mesh" / "simplified.obj",
        object_path / "urdf" / "meshes",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing object assets for {object_id}: {', '.join(missing)}")
    return object_path.resolve()


def sample_key(record: dict) -> tuple:
    return (
        str(record.get("obj_id", "")), str(record.get("scale_id", "")),
        str(record.get("pose_id", "")), str(record.get("action_id", "")),
        str(record.get("grasp_id", "")),
    )


def format_id(value, width: int = 3) -> str:
    try:
        return f"{int(value):0{width}d}"
    except (TypeError, ValueError):
        return str(value)


def relative_sample_dir(record: dict) -> Path:
    obj_id = str(record["obj_id"])
    pose_id = format_id(record.get("pose_id", 0))
    action_id = str(record.get("action_id", ""))
    grasp_id = format_id(record.get("grasp_id", 0))
    scale_id = record.get("scale_id")
    if scale_id is None:
        leaf = f"pose{pose_id}_act{action_id}_grasp{grasp_id}"
    else:
        leaf = f"scale{format_id(scale_id)}_pose{pose_id}_act{action_id}_grasp{grasp_id}"
    return Path(obj_id) / leaf


def make_single_hand_payload(
    record: dict, side: str, variant_idx: int, mesh_root: str, json_path: str
) -> dict:
    """Build the single-hand bench payload (dgbench ``task=eval``) for one grasp.

    Args:
        record: Converted record with ``dex_{pregrasp,grasp,squeeze}_<side>``.
        side: Which hand to export, ``"left"`` or ``"right"``.
        variant_idx: Index of this grasp within its sample group.
        mesh_root: Root directory of processed object assets.
        json_path: Source predictions JSON path, kept for traceability.

    Returns:
        Payload dict with ``obj_path/obj_pose/obj_scale``, the three 29-d
        ``{pregrasp,grasp,squeeze}_qpos`` arrays and metadata.
    """
    object_path = resolve_object_path(mesh_root, str(record["obj_id"]))
    payload = {
        "obj_path": str(object_path),
        "obj_pose": np.asarray(record["obj_pose"], dtype=np.float32),
        "obj_scale": float(record.get("obj_scale", 0.1)),
        "pregrasp_qpos": gys_pose_to_bench_qpos(record[f"dex_pregrasp_{side}"], f"dex_pregrasp_{side}"),
        "grasp_qpos": gys_pose_to_bench_qpos(record[f"dex_grasp_{side}"], f"dex_grasp_{side}"),
        "squeeze_qpos": gys_pose_to_bench_qpos(record[f"dex_squeeze_{side}"], f"dex_squeeze_{side}"),
        "raw_input_path": json_path,
        "candidate_idx": int(record.get("candidate_idx", variant_idx)),
        "hand_side": side,
    }
    for key in METADATA_KEYS:
        if key in record and key not in payload:
            payload[key] = record[key]
    payload["sample_key"] = "__".join(sample_key(record))
    payload["variant_idx"] = int(variant_idx)
    return payload


def make_eval3_payload(record: dict, variant_idx: int, mesh_root: str, json_path: str) -> dict:
    object_path = resolve_object_path(mesh_root, str(record["obj_id"]))
    payload = {
        "obj_path": str(object_path),
        "obj_pose": np.asarray(record["obj_pose"], dtype=np.float32),
        "obj_scale": float(record.get("obj_scale", 0.1)),
        "left_pregrasp_qpos": gys_pose_to_bench_qpos(record["dex_pregrasp_left"], "dex_pregrasp_left"),
        "left_grasp_qpos": gys_pose_to_bench_qpos(record["dex_grasp_left"], "dex_grasp_left"),
        "left_squeeze_qpos": gys_pose_to_bench_qpos(record["dex_squeeze_left"], "dex_squeeze_left"),
        "right_pregrasp_qpos": gys_pose_to_bench_qpos(record["dex_pregrasp_right"], "dex_pregrasp_right"),
        "right_grasp_qpos": gys_pose_to_bench_qpos(record["dex_grasp_right"], "dex_grasp_right"),
        "right_squeeze_qpos": gys_pose_to_bench_qpos(record["dex_squeeze_right"], "dex_squeeze_right"),
        "raw_input_path": json_path,
        "candidate_idx": int(record.get("candidate_idx", variant_idx)),
    }
    for key in METADATA_KEYS:
        if key in record and key not in payload:
            payload[key] = record[key]
    payload["sample_key"] = "__".join(sample_key(record))
    payload["variant_idx"] = int(variant_idx)
    return payload


def save_task(task, *, mesh_root: str, json_path: str) -> None:
    record, variant_idx, save_path, channel = task
    sides = CHANNEL_SIDES[channel]
    if len(sides) == 1:
        payload = make_single_hand_payload(record, sides[0], variant_idx, mesh_root, json_path)
    else:
        payload = make_eval3_payload(record, variant_idx, mesh_root, json_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, payload)


def required_record_keys(channel: str) -> tuple[str, ...]:
    """Keys a converted record must carry to feed the given channel's bench."""
    keys = ["obj_id", "obj_pose"]
    for side in CHANNEL_SIDES[channel]:
        keys += [f"dex_pregrasp_{side}", f"dex_grasp_{side}", f"dex_squeeze_{side}"]
    return tuple(keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="predictions JSON -> pre/squeeze -> per-task_type dgbench graspdata."
    )
    parser.add_argument("-i", "--input", required=True, help="predictions JSON from infer_dataset.py")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--output-root", required=True, help="dgbench-pami output root.")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--channels", type=str, default=",".join(CHANNELS),
        help="Comma-separated channels to convert (subset of "
             f"{','.join(CHANNELS)}). Each channel writes its own "
             "<exp-name>_<channel>_<hand_suffix> directory. Default: all.",
    )
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Keep at most N sample groups per channel (-1 = all).")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output directories of the selected channels first.")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to also dump the intermediate *_4bench.json.")
    parser.add_argument("--num-workers", "--numworker", "--num_worker",
                        dest="num_workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channels {unknown}; choose from {list(CHANNELS)}.")
    if not channels:
        raise ValueError("--channels must select at least one channel.")
    channel_set = set(channels)

    with Path(args.input).open("r", encoding="utf-8") as f:
        all_records = json.load(f)
    if not isinstance(all_records, list):
        raise ValueError("Input JSON root must be a list of records.")

    # Stage 1: compute pre / squeeze poses per channel; skip (and count)
    # records outside the enabled channels or missing a required hand.
    # missing_hand is counted PER CHANNEL so each channel's sidecar can report
    # its intended denominator (kept + missing_hand).
    skipped: dict[str, int] = defaultdict(int)
    missing_by_channel: dict[str, int] = defaultdict(int)
    records_by_channel: dict[str, list[dict]] = {c: [] for c in channels}
    for idx, record in enumerate(all_records):
        reason = add_pre_squeeze(record, idx, channel_set)
        if reason is None:
            records_by_channel[str(record["task_type"])].append(record)
        else:
            skipped[reason] += 1
            if reason == "missing_hand":
                missing_by_channel[str(record.get("task_type", ""))] += 1
                print(f"[warn] record {idx} (task_type={record.get('task_type')}) "
                      f"missing required pred hand; skipped")

    if args.save_json:
        kept = [r for c in channels for r in records_by_channel[c]]
        with Path(args.save_json).open("w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        print(f"[info] wrote intermediate 4bench json -> {args.save_json}")

    # Stage 2: per channel — validate, group, convert to bench qpos, save .npy.
    save_tasks: list[tuple] = []
    channel_summaries: list[str] = []
    for channel in channels:
        records = records_by_channel[channel]
        required = required_record_keys(channel)
        for idx, record in enumerate(records):
            missing = [k for k in required if k not in record]
            if missing:
                raise KeyError(f"[{channel}] record {idx} missing required keys: {missing}")

        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for record in records:
            grouped[sample_key(record)].append(record)
        sample_keys = sorted(grouped)
        if args.max_samples > 0:
            sample_keys = sample_keys[: args.max_samples]

        exp_dir = Path(args.output_root) / f"{args.exp_name}_{channel}_{CHANNEL_HAND_SUFFIX[channel]}"
        grasp_dir = exp_dir / "graspdata"
        if args.clean and exp_dir.exists():
            shutil.rmtree(exp_dir)

        n_missing = int(missing_by_channel.get(channel, 0))
        summary = {
            "channel": channel,
            "kept": len(records),
            "saved": 0,
            "samples": 0,
            "missing_hand": n_missing,
            "task_type_skipped": int(skipped["task_type"]),
            # intended denominator for this channel = input records whose
            # task_type routes here (kept + missing_hand); the eval-side
            # success rate should be double-reported as succ/eval and
            # succ/(eval + missing_hand).
            "intended_total": len(records) + n_missing,
        }

        if sample_keys:
            grasp_dir.mkdir(parents=True, exist_ok=True)
            n_before = len(save_tasks)
            for key in sample_keys:
                sample_records = sorted(grouped[key], key=lambda r: int(r.get("candidate_idx", 0)))
                for variant_idx, record in enumerate(sample_records):
                    save_path = grasp_dir / relative_sample_dir(record) / f"{variant_idx}.npy"
                    save_tasks.append((record, variant_idx, save_path, channel))
            summary["saved"] = len(save_tasks) - n_before
            summary["samples"] = len(sample_keys)
        else:
            exp_dir.mkdir(parents=True, exist_ok=True)

        summary_path = exp_dir / "convert_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if not sample_keys:
            channel_summaries.append(
                f"  [{channel}] kept=0 missing_hand={n_missing} "
                f"intended_total={summary['intended_total']} — nothing to write\n"
                f"      sidecar:   {summary_path}"
            )
            continue
        channel_summaries.append(
            f"  [{channel}] kept={summary['kept']} samples={summary['samples']} "
            f"saved={summary['saved']} missing_hand={n_missing} "
            f"intended_total={summary['intended_total']}\n"
            f"      graspdata: {grasp_dir}\n"
            f"      save_dir:  {exp_dir}\n"
            f"      sidecar:   {summary_path}"
        )

    worker = partial(save_task, mesh_root=args.mesh_root, json_path=str(args.input))
    if args.num_workers <= 1 or len(save_tasks) <= 1:
        for task in save_tasks:
            worker(task)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            for _ in ex.map(worker, save_tasks):
                pass

    total_skipped = sum(skipped.values())
    print(f"Loaded records: {len(all_records)}")
    print(f"Channels: {channels}")
    print(f"Skipped records: {total_skipped} "
          f"(task_type={skipped['task_type']}, missing_hand={skipped['missing_hand']})")
    print(f"Saved bench records: {len(save_tasks)}")
    print("Per-channel summary:")
    for line in channel_summaries:
        print(line)
    print("[hint] 评测请双口径汇报：成功率(evaluated) = succ/eval；"
          "成功率(intended) = succ/(eval + missing_hand)。"
          "各通道 missing_hand/intended_total 见 <exp_dir>/convert_summary.json。")


if __name__ == "__main__":
    main()
