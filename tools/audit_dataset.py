#!/usr/bin/env python3
"""Record-level audit of a v3 multi-task split (no point clouds, CPU, minutes).

Checks, over EVERY record:
  - field presence matrix per task_type (informational);
  - scale rule: |scale_id/100 - obj_scale| < 1e-6 on all dual-field records;
  - resolve_obj_scale succeeds on every record;
  - mesh coverage: every obj_id directory + mesh/simplified.obj exists;
  - hand consistency: task_type left => only dex_grasp_left, right => only
    dex_grasp_right, lgbidex/bidex => both; 28-dim grasps; finite values;
    non-empty guidance; 7-dim obj_pose with unit-norm quaternion;
  - group key (obj_id|pose_id|scale_id|guidance) size distribution summary.

Prints a console table + writes a JSON report; exits non-zero on any FAIL.

Usage:
  python tools/audit_dataset.py \\
    --split-path /home/jiaxuan/slai/new-data/read/test_v3.json \\
    --mesh-root  /home/jiaxuan/data/oakink_obj/processed_data \\
    --out audit_test_v3.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.lgbidex_io import resolve_obj_scale  # noqa: E402
from data.dataset import group_id_of, hands_present  # noqa: E402

TASK_TYPES = ("left", "right", "lgbidex", "bidex")
EXPECTED_HANDS = {
    "left": {"left"},
    "right": {"right"},
    "lgbidex": {"left", "right"},
    "bidex": {"left", "right"},
}
FIELDS = (
    "obj_id", "obj_pose", "obj_scale", "scale_id", "pose_id", "cate_id",
    "action", "action_id", "grasp_id", "cand_idx", "guidance",
    "dex_grasp_left", "dex_grasp_right", "task_type",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-path", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def is_finite_list(values: list) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


def main() -> None:
    args = parse_args()
    mesh_root = Path(args.mesh_root).expanduser().resolve()

    with open(args.split_path, "r") as f:
        data = json.load(f)
    print(f"split={args.split_path} records={len(data)}")

    presence: dict[str, dict[str, int]] = {}
    type_counts: dict[str, int] = {}
    failures: dict[str, list[str]] = {
        "task_type": [], "scale_rule": [], "resolve_scale": [],
        "hand_consistency": [], "grasp_dim": [], "grasp_finite": [],
        "guidance": [], "obj_pose": [],
    }
    dual_scale_records = 0
    obj_ids: set[str] = set()
    group_sizes: dict[str, int] = {}
    groups_per_type: dict[str, set[str]] = {}

    for idx, rec in enumerate(data):
        obj_id = str(rec.get("obj_id", ""))
        obj_ids.add(obj_id)
        tt = rec.get("task_type")
        tt_key = str(tt) if tt else "<missing>"
        type_counts[tt_key] = type_counts.get(tt_key, 0) + 1
        row = presence.setdefault(tt_key, {f: 0 for f in FIELDS})
        for f in FIELDS:
            if rec.get(f) is not None:
                row[f] += 1
        if not tt or tt not in TASK_TYPES:
            failures["task_type"].append(f"idx={idx} obj_id={obj_id} task_type={tt!r}")

        if rec.get("obj_scale") is not None and rec.get("scale_id") is not None:
            dual_scale_records += 1
            if abs(float(rec["scale_id"]) / 100.0 - float(rec["obj_scale"])) >= 1e-6:
                failures["scale_rule"].append(
                    f"idx={idx} obj_id={obj_id} scale_id={rec['scale_id']} obj_scale={rec['obj_scale']}"
                )
        try:
            resolve_obj_scale(rec)
        except ValueError as exc:
            failures["resolve_scale"].append(f"idx={idx} {exc}")

        sides = set(hands_present(rec))
        expected = EXPECTED_HANDS.get(str(tt))
        if expected is not None and sides != expected:
            failures["hand_consistency"].append(
                f"idx={idx} obj_id={obj_id} task_type={tt} hands={sorted(sides)}"
            )
        for side in sides:
            grasp = rec[f"dex_grasp_{side}"]
            if len(grasp) != 28:
                failures["grasp_dim"].append(
                    f"idx={idx} obj_id={obj_id} dex_grasp_{side} len={len(grasp)}"
                )
            elif not is_finite_list(grasp):
                failures["grasp_finite"].append(f"idx={idx} obj_id={obj_id} dex_grasp_{side}")

        guidance = str(rec.get("guidance", rec.get("guidence", ""))).strip()
        if not guidance:
            failures["guidance"].append(f"idx={idx} obj_id={obj_id}")

        obj_pose = rec.get("obj_pose")
        if not isinstance(obj_pose, list) or len(obj_pose) != 7 or not is_finite_list(obj_pose):
            failures["obj_pose"].append(f"idx={idx} obj_id={obj_id} obj_pose={obj_pose!r}")
        else:
            quat_norm = float(np.linalg.norm(obj_pose[3:]))
            if abs(quat_norm - 1.0) >= 1e-3:
                failures["obj_pose"].append(
                    f"idx={idx} obj_id={obj_id} |quat|={quat_norm:.6f}"
                )

        gid = group_id_of(rec)
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
        groups_per_type.setdefault(tt_key, set()).add(gid)

    missing_meshes = [
        obj_id
        for obj_id in sorted(obj_ids)
        if not (mesh_root / obj_id / "mesh" / "simplified.obj").exists()
    ]

    sizes = np.asarray(sorted(group_sizes.values()))
    group_summary = {
        "num_groups": int(sizes.size),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_p90": float(np.percentile(sizes, 90)),
        "size_max": int(sizes.max()),
        "groups_per_task_type": {tt: len(g) for tt, g in sorted(groups_per_type.items())},
    }

    print("\nField presence rate per task_type:")
    header = f"{'field':<16}" + "".join(f"{tt:>10}" for tt in sorted(presence))
    print(header)
    for f in FIELDS:
        row = f"{f:<16}"
        for tt in sorted(presence):
            row += f"{presence[tt][f] / type_counts[tt]:>10.3f}"
        print(row)
    print(f"\ntask_type counts: {dict(sorted(type_counts.items()))}")
    print(f"dual scale-field records checked: {dual_scale_records}")
    print(f"unique obj_ids: {len(obj_ids)}  missing meshes: {len(missing_meshes)}")
    if missing_meshes:
        print("  missing (first 20): " + ", ".join(missing_meshes[:20]))
    print(f"group summary: {group_summary}")

    checks = {
        "task_type_valid": len(failures["task_type"]) == 0,
        "scale_rule": len(failures["scale_rule"]) == 0,
        "resolve_obj_scale": len(failures["resolve_scale"]) == 0,
        "mesh_coverage": len(missing_meshes) == 0,
        "hand_consistency": len(failures["hand_consistency"]) == 0,
        "grasp_dim_28": len(failures["grasp_dim"]) == 0,
        "grasp_finite": len(failures["grasp_finite"]) == 0,
        "guidance_nonempty": len(failures["guidance"]) == 0,
        "obj_pose_7d_unit_quat": len(failures["obj_pose"]) == 0,
    }
    print("\nChecks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    report = {
        "split": str(args.split_path),
        "mesh_root": str(mesh_root),
        "total_records": len(data),
        "task_type_counts": type_counts,
        "field_presence": {
            tt: {f: presence[tt][f] / type_counts[tt] for f in FIELDS} for tt in presence
        },
        "dual_scale_records": dual_scale_records,
        "missing_meshes": missing_meshes,
        "group_summary": group_summary,
        "checks": checks,
        "failure_examples": {k: v[:20] for k, v in failures.items() if v},
        "failure_counts": {k: len(v) for k, v in failures.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {out_path}")

    if not all(checks.values()):
        print("AUDIT: FAIL")
        sys.exit(1)
    print("AUDIT: PASS")


if __name__ == "__main__":
    main()
