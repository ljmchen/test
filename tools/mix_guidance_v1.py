#!/usr/bin/env python3
"""Mix v1 (rich, contact-cue) guidance back into lgbidex records of the v2 train split.

Motivation (优化方案 §2 T4): v2 rewrote lgbidex guidance into ~1.7k templates that
cannot disambiguate 11k+ grasp modes; v1 text carries contact/finger cues. Here we
deterministically replace the ``guidance`` of ``task_type == lgbidex`` records in
train_fuse_v2_split.json with the v1 version at probability ``--prob`` (default 0.5,
hashed on ``grasp_id`` so the choice is reproducible). Everything else — record
count, record order, all non-guidance fields, and all other task types — is
untouched (陷阱①: epoch_sampling groups by guidance, so records must be REPLACED
in place, never duplicated). val/test files are NOT produced here (陷阱②: they
stay pure v2 so eval text remains in-distribution).

Alignment: v2 was generated from v1 by refining guidance only (records 1:1, see
fuse_v2_guidance_summary.json), but v2_split dropped ~5% of objects, so lgbidex
records are matched back to v1 via the key ``(obj_id, scale_id, pose_id,
grasp_id, cand_idx)``. The key must be UNIQUE within v1-lgbidex and hit 100% of
v2_split-lgbidex; otherwise we fall back to a fingerprint key ``(obj_id,
md5(json(dex_grasp_left)))`` with the same requirements. Before any replacement,
``--check-samples`` matched pairs are verified to be byte-identical outside the
``guidance`` field (json.dumps comparison); any mismatch aborts.

Outputs (next to --v2-split by default):
  train_fuse_v2v1mix_split.json          mixed training file (same length/order)
  train_fuse_v2v1mix_split_summary.json  counts, key/hit-rate, unique-guidance stats

Usage:
  python tools/mix_guidance_v1.py \
      --v1 /home/jiaxuan/data/pose_data/train_fuse_v1.json \
      --v2-split /home/jiaxuan/data/pose_data/train_fuse_v2_split.json \
      --output /home/jiaxuan/data/pose_data/train_fuse_v2v1mix_split.json \
      --prob 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

PRIMARY_KEY_FIELDS = ("obj_id", "scale_id", "pose_id", "grasp_id", "cand_idx")
TARGET_TASK_TYPE = "lgbidex"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="/home/jiaxuan/data/pose_data/train_fuse_v1.json")
    ap.add_argument("--v2-split",
                    default="/home/jiaxuan/data/pose_data/train_fuse_v2_split.json")
    ap.add_argument("--output", default=None,
                    help="default: <v2_split_dir>/train_fuse_v2v1mix_split.json")
    ap.add_argument("--summary", default=None,
                    help="default: <output>_summary.json (suffix replaced)")
    ap.add_argument("--prob", type=float, default=0.5,
                    help="replacement probability; 0.5 uses md5(grasp_id) %% 2 == 0")
    ap.add_argument("--check-samples", type=int, default=500,
                    help="matched pairs to verify byte-identical outside guidance (>=200)")
    ap.add_argument("--seed", type=int, default=42, help="seed for the pair-check sample")
    return ap.parse_args()


def primary_key(rec: dict) -> tuple:
    return tuple(rec[f] for f in PRIMARY_KEY_FIELDS)


def fingerprint_key(rec: dict) -> tuple:
    blob = json.dumps(rec["dex_grasp_left"], separators=(",", ":")).encode()
    return (rec["obj_id"], hashlib.md5(blob).hexdigest())


def dumps_wo_guidance(rec: dict) -> str:
    return json.dumps({k: v for k, v in rec.items() if k != "guidance"},
                      sort_keys=True, separators=(",", ":"))


def replace_decision(grasp_id, prob: float) -> bool:
    """Deterministic per-grasp_id coin flip. At the default prob=0.5 this is
    exactly ``int(md5(str(grasp_id)), 16) % 2 == 0`` (the documented rule)."""
    h = int(hashlib.md5(str(grasp_id).encode()).hexdigest(), 16)
    if abs(prob - 0.5) < 1e-12:
        return h % 2 == 0
    return (h % 10**9) / 10**9 < prob


def build_lookup(records: list[dict], key_fn) -> tuple[dict, int]:
    """Map key -> lgbidex record; duplicate keys are poisoned with None."""
    lookup: dict = {}
    dup = 0
    for rec in records:
        if rec.get("task_type") != TARGET_TASK_TYPE:
            continue
        k = key_fn(rec)
        if k in lookup:
            if lookup[k] is not None:
                dup += 1
                lookup[k] = None
            else:
                dup += 1
        else:
            lookup[k] = rec
    return lookup, dup


def main() -> None:
    args = parse_args()
    v2_path = Path(args.v2_split)
    out_path = Path(args.output) if args.output else v2_path.parent / "train_fuse_v2v1mix_split.json"
    summary_path = (Path(args.summary) if args.summary
                    else out_path.with_name(out_path.stem + "_summary.json"))
    if not 0.0 <= args.prob <= 1.0:
        raise ValueError(f"--prob must be in [0, 1], got {args.prob}")
    if args.check_samples < 200:
        raise ValueError("--check-samples must be >= 200 (alignment safety check)")

    t0 = time.time()
    print(f"loading v1: {args.v1} ...", flush=True)
    with open(args.v1) as f:
        v1 = json.load(f)
    print(f"loading v2 split: {v2_path} ...", flush=True)
    with v2_path.open() as f:
        v2 = json.load(f)
    print(f"v1 records = {len(v1)}; v2_split records = {len(v2)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    v2_lgb_idx = [i for i, r in enumerate(v2) if r.get("task_type") == TARGET_TASK_TYPE]
    n_lgb = len(v2_lgb_idx)
    print(f"v2_split lgbidex records = {n_lgb}", flush=True)

    # ─── Alignment key selection: primary first, fingerprint fallback ──────
    key_name = "+".join(PRIMARY_KEY_FIELDS)
    key_fn = primary_key
    lookup, dup = build_lookup(v1, key_fn)
    hits = sum(1 for i in v2_lgb_idx if lookup.get(key_fn(v2[i])) is not None)
    print(f"[primary key {key_name}] v1-lgbidex unique={len(lookup)} dup_keys={dup} "
          f"v2 hit = {hits}/{n_lgb}", flush=True)
    if dup > 0 or hits != n_lgb:
        key_name = "obj_id+md5(json(dex_grasp_left))"
        key_fn = fingerprint_key
        lookup, dup = build_lookup(v1, key_fn)
        hits = sum(1 for i in v2_lgb_idx if lookup.get(key_fn(v2[i])) is not None)
        print(f"[fallback key {key_name}] v1-lgbidex unique={len(lookup)} dup_keys={dup} "
              f"v2 hit = {hits}/{n_lgb}", flush=True)
        if dup > 0 or hits != n_lgb:
            raise RuntimeError(
                f"alignment failed on both keys: fallback dup={dup}, hit {hits}/{n_lgb}")

    # ─── Pair sanity check: everything except guidance must be identical ───
    rng = random.Random(args.seed)
    sample_idx = rng.sample(v2_lgb_idx, min(args.check_samples, n_lgb))
    for i in sample_idx:
        v2r = v2[i]
        v1r = lookup[key_fn(v2r)]
        if dumps_wo_guidance(v1r) != dumps_wo_guidance(v2r):
            raise RuntimeError(
                f"pair mismatch outside guidance at v2 index {i} "
                f"(key={key_fn(v2r)}); aborting before any write")
    print(f"pair check OK: {len(sample_idx)} sampled pairs byte-identical "
          f"outside guidance", flush=True)

    # ─── In-place replacement (order & count preserved by construction) ────
    guid_before = {v2[i]["guidance"] for i in v2_lgb_idx}
    replaced = 0
    replaced_text_changed = 0
    touched_other = Counter()  # must stay empty
    for i in v2_lgb_idx:
        rec = v2[i]
        if not replace_decision(rec["grasp_id"], args.prob):
            continue
        v1_guidance = lookup[key_fn(rec)]["guidance"]
        replaced += 1
        if v1_guidance != rec["guidance"]:
            replaced_text_changed += 1
        rec["guidance"] = v1_guidance
    guid_after = {v2[i]["guidance"] for i in v2_lgb_idx}
    rate = replaced / n_lgb if n_lgb else 0.0
    print(f"replaced {replaced}/{n_lgb} lgbidex guidance ({rate:.4f}); "
          f"text actually changed in {replaced_text_changed}; "
          f"unique guidance {len(guid_before)} -> {len(guid_after)}", flush=True)

    task_counts = Counter(str(r.get("task_type", "unknown")) for r in v2)
    print(f"writing {out_path} ...", flush=True)
    with out_path.open("w") as f:
        json.dump(v2, f)

    summary = {
        "source_v1": str(args.v1),
        "source_v2_split": str(v2_path),
        "output": str(out_path),
        "rule": ("task_type==lgbidex only; replace guidance with v1 when "
                 "int(md5(str(grasp_id)),16) % 2 == 0 (prob=0.5); records replaced "
                 "in place, count/order/non-guidance fields unchanged"),
        "prob": args.prob,
        "total_records": len(v2),
        "records_per_task_type": dict(sorted(task_counts.items())),
        "lgbidex_total": n_lgb,
        "replaced": replaced,
        "replaced_rate": rate,
        "replaced_text_changed": replaced_text_changed,
        "modified_records_other_task_types": sum(touched_other.values()),
        "unique_lgbidex_guidance_before": len(guid_before),
        "unique_lgbidex_guidance_after": len(guid_after),
        "alignment_key": key_name,
        "alignment_hit": f"{hits}/{n_lgb}",
        "alignment_hit_rate": hits / n_lgb if n_lgb else 1.0,
        "pair_check_samples": len(sample_idx),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"writing {summary_path} ... done ({summary['elapsed_sec']}s)", flush=True)
    print(f"OK: total {len(v2)} records; lgbidex {replaced}/{n_lgb} replaced "
          f"({rate:.2%}); other task types untouched", flush=True)


if __name__ == "__main__":
    main()
