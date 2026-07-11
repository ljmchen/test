#!/usr/bin/env python3
"""Compute relquantile11 translation statistics from a v3 multi-task split.

Works at the record level (no dataset ``__getitem__``, no point clouds): for a
stratified per-task_type random sample it reproduces the dataset's centering
``shift = obj_trans + R @ (bbox_center * scale)`` from the mesh bbox, then
accumulates the scale-relative translation ``rel = (t_world - shift) / L_obj``
per hand side (only for hands present in the record). Outputs the stats JSON
consumed by ``data.pose_normalizer`` (mode ``relquantile11``) plus overlay
histograms, and runs built-in acceptance assertions (PASS/FAIL).

The stored q01/q99 mapping endpoints are the per-task_type envelope of each
type's q01/q99 with 2% span padding, so every task type's central 98%
translation mass maps strictly inside [-1, 1]; the mixed-pool quantiles are
archived as mixed_q01/mixed_q99.

Usage:
  python tools/compute_norm_stats.py \\
    --split-path /home/jiaxuan/slai/new-data/read/test_v3.json \\
    --mesh-root  /home/jiaxuan/data/oakink_obj/processed_data \\
    --out tools/norm_stats_v3.json --per-type-cap 25000 --seed 0 \\
    --hist-dir outputs_norm_stats
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.lgbidex_io import (  # noqa: E402
    axis_angle_to_matrix,
    canonicalize_axis_angle,
    load_mesh_vertices,
    matrix_to_rotation_6d,
    normalize_rotation_6d,
    quaternion_to_matrix,
    resolve_obj_scale,
)
from data.dataset import hands_present  # noqa: E402
from data.pose_normalizer import LgbidexPoseNormalizer  # noqa: E402

SIDES = ("left", "right")
AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-path", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--out", default="tools/norm_stats_v3.json")
    ap.add_argument("--per-type-cap", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist-dir", default=None, help="histogram dir (default: alongside --out)")
    return ap.parse_args()


def load_bbox(mesh_root: Path, obj_id: str, cache: dict) -> tuple[torch.Tensor, float]:
    """Load (bbox_center, bbox_diagonal) of an object's simplified mesh, cached.

    Args:
        mesh_root: Processed-object root directory.
        obj_id: Object id (directory name under mesh_root).
        cache: obj_id -> (bbox_center, diag) cache, mutated in place.

    Returns:
        bbox_center: ``(3,)`` float32 mesh bbox center.
        diag: bbox diagonal length in mesh units.
    """
    if obj_id not in cache:
        mesh_path = mesh_root / obj_id / "mesh" / "simplified.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh for '{obj_id}': {mesh_path}")
        vertices = load_mesh_vertices(mesh_path)
        mn = vertices.min(dim=0).values
        mx = vertices.max(dim=0).values
        cache[obj_id] = (0.5 * (mn + mx), float((mx - mn).norm()))
    return cache[obj_id]


def record_shift_and_L(record: dict, mesh_root: Path, cache: dict) -> tuple[np.ndarray, float]:
    """Centering shift and L_obj of a record (matches dataset._center_hand_pose).

    Args:
        record: Split record with ``obj_id`` / ``obj_pose`` / scale fields.
        mesh_root: Processed-object root directory.
        cache: obj_id -> (bbox_center, diag) cache.

    Returns:
        shift: ``(3,)`` float64 world-frame centering offset.
        L_obj: bbox diagonal x resolved object scale.
    """
    bbox_center, diag = load_bbox(mesh_root, str(record["obj_id"]), cache)
    scale = resolve_obj_scale(record)
    obj_pose = torch.tensor(record["obj_pose"], dtype=torch.float32)
    if obj_pose.shape[-1] != 7:
        raise ValueError(
            f"obj_id={record['obj_id']!r}: expected obj_pose dim 7, got {obj_pose.shape[-1]}."
        )
    rotation = quaternion_to_matrix(obj_pose[3:], order="wxyz")
    shift = obj_pose[:3] + rotation @ (bbox_center * scale)
    return shift.numpy().astype(np.float64), diag * scale


def quantile_block(arr: np.ndarray) -> dict:
    """Per-component q01/q50/q99/mean/std of an ``(N, 3)`` array.

    Args:
        arr: ``(N, 3)`` sample array.

    Returns:
        Dict with q01/q50/q99/mean/std (3-dim lists) and count.
    """
    return {
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q50": np.percentile(arr, 50, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": int(arr.shape[0]),
    }


def build_pose31(raw28: list[float]) -> torch.Tensor:
    """Convert a raw 28-dim grasp to the 31-dim [trans, rot6d, joints] pose.

    Args:
        raw28: ``[trans(3), axis_angle(3), joints(22)]``.

    Returns:
        ``(31,)`` float32 pose tensor (translation NOT centered).
    """
    trans = torch.tensor(raw28[:3], dtype=torch.float32)
    axis_angle = canonicalize_axis_angle(torch.tensor(raw28[3:6], dtype=torch.float32))
    rot6d = normalize_rotation_6d(matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle)))
    joints = torch.tensor(raw28[6:28], dtype=torch.float32)
    return torch.cat([trans, rot6d, joints], dim=-1)


def plot_histograms(
    rel_by_type: dict[str, np.ndarray], L_by_type: dict[str, list[float]], hist_dir: Path
) -> list[str]:
    """Save overlay histograms (rel x/y/z + L_obj) and return the file paths."""
    hist_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for axis_idx, axis in enumerate(AXES):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tt, arr in sorted(rel_by_type.items()):
            ax.hist(arr[:, axis_idx], bins=120, density=True, histtype="step", label=tt)
        ax.set_title(f"translation_rel {axis} (t_centered / L_obj)")
        ax.set_xlabel(f"rel {axis}")
        ax.legend()
        path = hist_dir / f"rel_trans_{axis}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tt, values in sorted(L_by_type.items()):
        ax.hist(values, bins=80, density=True, histtype="step", label=tt)
    ax.set_title("L_obj (bbox diag x obj_scale)")
    ax.set_xlabel("L_obj [m]")
    ax.legend()
    path = hist_dir / "L_obj_hist.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))
    return saved


def main() -> None:
    args = parse_args()
    mesh_root = Path(args.mesh_root).expanduser().resolve()
    out_path = Path(args.out)
    hist_dir = Path(args.hist_dir) if args.hist_dir else out_path.parent

    with open(args.split_path, "r") as f:
        data = json.load(f)
    by_type: dict[str, list[int]] = {}
    for idx, rec in enumerate(data):
        task_type = rec.get("task_type")
        if not task_type:
            raise ValueError(f"Record idx={idx} obj_id={rec.get('obj_id')!r} missing 'task_type'.")
        by_type.setdefault(str(task_type), []).append(idx)

    rng = random.Random(args.seed)
    sampled: dict[str, list[int]] = {}
    for tt, indices in sorted(by_type.items()):
        sampled[tt] = rng.sample(indices, args.per_type_cap) if len(indices) > args.per_type_cap else list(indices)
    print(f"total={len(data)} sampled=" + ", ".join(f"{tt}:{len(v)}" for tt, v in sampled.items()))

    bbox_cache: dict[str, tuple[torch.Tensor, float]] = {}
    rel_by_side: dict[str, list[np.ndarray]] = {s: [] for s in SIDES}
    rel_by_tt_side: dict[str, dict[str, list[np.ndarray]]] = {}
    rel_by_type: dict[str, list[np.ndarray]] = {}
    L_by_type: dict[str, list[float]] = {}
    roundtrip_records: dict[str, list[tuple[dict, float]]] = {tt: [] for tt in sampled}

    for tt, indices in sampled.items():
        rel_by_tt_side[tt] = {s: [] for s in SIDES}
        rel_by_type[tt] = []
        L_by_type[tt] = []
        for n_done, idx in enumerate(indices):
            rec = data[idx]
            shift, L_obj = record_shift_and_L(rec, mesh_root, bbox_cache)
            sides = hands_present(rec)
            if not sides:
                raise ValueError(f"Record idx={idx} obj_id={rec.get('obj_id')!r} has no hands.")
            for side in sides:
                raw = rec[f"dex_grasp_{side}"]
                if len(raw) != 28:
                    raise ValueError(
                        f"Record idx={idx} obj_id={rec.get('obj_id')!r}: "
                        f"dex_grasp_{side} has {len(raw)} dims, expected 28."
                    )
                rel = (np.asarray(raw[:3], dtype=np.float64) - shift) / L_obj
                rel_by_side[side].append(rel)
                rel_by_tt_side[tt][side].append(rel)
                rel_by_type[tt].append(rel)
            L_by_type[tt].append(L_obj)
            if len(roundtrip_records[tt]) < 256:
                roundtrip_records[tt].append((rec, L_obj))
            if (n_done + 1) % 10000 == 0:
                print(f"  [{tt}] {n_done + 1}/{len(indices)}", flush=True)

    rel_side_arr = {s: np.asarray(rel_by_side[s]) for s in SIDES}
    translation_rel: dict[str, dict] = {s: quantile_block(rel_side_arr[s]) for s in SIDES}
    per_task_type = {
        tt: {s: quantile_block(np.asarray(v)) for s, v in sides.items() if len(v) > 0}
        for tt, sides in rel_by_tt_side.items()
    }

    # Mapping endpoints = per-task_type envelope of per-type q01/q99 + 2% span
    # padding, so every task type's central 98% translation mass lands strictly
    # inside [-1, 1] (the mixed-pool quantiles are kept as mixed_q01/mixed_q99).
    for s in SIDES:
        env_lo = np.min(
            [per_task_type[tt][s]["q01"] for tt in per_task_type if s in per_task_type[tt]],
            axis=0,
        )
        env_hi = np.max(
            [per_task_type[tt][s]["q99"] for tt in per_task_type if s in per_task_type[tt]],
            axis=0,
        )
        pad = 0.02 * (env_hi - env_lo)
        translation_rel[s]["mixed_q01"] = translation_rel[s]["q01"]
        translation_rel[s]["mixed_q99"] = translation_rel[s]["q99"]
        translation_rel[s]["q01"] = (env_lo - pad).tolist()
        translation_rel[s]["q99"] = (env_hi + pad).tolist()
    translation_rel["delta"] = {
        key: (
            np.asarray(translation_rel["right"][key]) - np.asarray(translation_rel["left"][key])
        ).tolist()
        for key in ("q01", "q50", "q99", "mean", "std", "mixed_q01", "mixed_q99")
    }

    normalizers = {
        s: LgbidexPoseNormalizer(
            hand_side=s,
            mode="relquantile11",
            rel_quantile={"q01": translation_rel[s]["q01"], "q99": translation_rel[s]["q99"]},
        )
        for s in SIDES
    }

    clip_rate: dict[str, dict[str, list[float]]] = {}
    for tt, sides in rel_by_tt_side.items():
        clip_rate[tt] = {}
        for s, v in sides.items():
            if not v:
                continue
            arr = np.asarray(v)
            q01 = np.asarray(translation_rel[s]["q01"])
            q99 = np.asarray(translation_rel[s]["q99"])
            normed = 2.0 * (arr - q01) / (q99 - q01) - 1.0
            clip_rate[tt][s] = (np.abs(normed) > 1.0).mean(axis=0).tolist()

    roundtrip_max_err = 0.0
    for tt, recs in roundtrip_records.items():
        for rec, L_obj in recs:
            shift, _ = record_shift_and_L(rec, mesh_root, bbox_cache)
            for side in hands_present(rec):
                pose = build_pose31(rec[f"dex_grasp_{side}"])
                pose[:3] -= torch.tensor(shift, dtype=torch.float32)
                cycled = normalizers[side].denormalize_pose(
                    normalizers[side].normalize_pose(pose, L_obj=L_obj), L_obj=L_obj
                )
                roundtrip_max_err = max(roundtrip_max_err, float((cycled - pose).abs().max()))

    a1 = True
    for tt, sides in per_task_type.items():
        for s, block in sides.items():
            q01 = np.asarray(translation_rel[s]["q01"])
            q99 = np.asarray(translation_rel[s]["q99"])
            for key in ("q01", "q99"):
                normed = 2.0 * (np.asarray(block[key]) - q01) / (q99 - q01) - 1.0
                if np.abs(normed).max() > 1.05:
                    a1 = False
    a2 = all(max(rates) < 0.02 for sides in clip_rate.values() for rates in sides.values())
    a3 = roundtrip_max_err < 1e-5

    # Left/right table consistency. The z endpoints genuinely differ between
    # hands (left rel-z is almost entirely positive, matching the legacy v2
    # minmax tables: left z in [0.004, 0.201] vs right z in [-0.158, 0.145]),
    # so z is gated on span ratio only; x/y are gated on endpoint difference
    # too. Full per-axis endpoint diffs are archived in checks.
    span = {
        s: np.asarray(translation_rel[s]["q99"]) - np.asarray(translation_rel[s]["q01"])
        for s in SIDES
    }
    endpoint_diff = {
        key: (
            np.abs(
                np.asarray(translation_rel["left"][key])
                - np.asarray(translation_rel["right"][key])
            )
            / np.maximum(span["left"], span["right"])
        ).tolist()
        for key in ("q01", "q99")
    }
    span_ratio = (np.maximum(span["left"], span["right"]) / np.minimum(span["left"], span["right"])).tolist()
    a4 = max(span_ratio) < 2.0 and all(
        max(endpoint_diff[key][:2]) < 0.20 for key in ("q01", "q99")
    )

    assertions = {
        "a1_per_type_quantiles_within_1.05": bool(a1),
        "a2_per_type_axis_clip_rate_lt_2pct": bool(a2),
        "a3_roundtrip_err_lt_1e-5": bool(a3),
        "a4_left_right_table_consistency": bool(a4),
    }

    hist_files = plot_histograms(
        {tt: np.asarray(v) for tt, v in rel_by_type.items()}, L_by_type, hist_dir
    )

    out = {
        "version": "norm_stats_v3/relquantile11-1",
        "mode": "relquantile11",
        "source": {
            "split": str(args.split_path),
            "seed": args.seed,
            "per_type_cap": args.per_type_cap,
            "total_records": len(data),
            "sampled_counts": {tt: len(v) for tt, v in sampled.items()},
            "note": f"stats computed from {Path(args.split_path).name}",
        },
        "table_rule": (
            "q01/q99 are the normalization mapping endpoints: per-task_type "
            "envelope of per-type q01/q99 with 2% span padding; mixed-pool "
            "quantiles are archived as mixed_q01/mixed_q99."
        ),
        "translation_rel": translation_rel,
        "per_task_type": per_task_type,
        "checks": {
            "clip_rate": clip_rate,
            "roundtrip_max_err": roundtrip_max_err,
            "left_right_endpoint_diff": endpoint_diff,
            "left_right_span_ratio": span_ratio,
            "assertions": assertions,
        },
        "histograms": hist_files,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")
    for path in hist_files:
        print(f"hist  -> {path}")

    for s in SIDES:
        print(
            f"{s:>5}: q01={np.round(translation_rel[s]['q01'], 4).tolist()} "
            f"q99={np.round(translation_rel[s]['q99'], 4).tolist()} "
            f"(n={translation_rel[s]['count']})"
        )
    print(f"roundtrip_max_err={roundtrip_max_err:.3e}")
    ok = True
    for name, passed in assertions.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
