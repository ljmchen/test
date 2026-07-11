#!/usr/bin/env python3
"""Build a GT-as-prediction JSON: copy each record's dex_grasp_{left,right} into
pred_{left,right} so the dataset ground truth can be pushed through the same
convert -> eval chain as a sanity baseline (bimanual lgbidex should score ~91%).

Task-type aware: all task_types are kept by default (each feeds its own bench
channel); pass ``--task-type-filter`` to narrow. For a single-hand record only
the hand that exists is written (no fabricated missing hand)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dataset split json (has dex_grasp_*)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=2000,
                    help="Total cap AFTER task_type filtering (-1 = all). Applied "
                         "over the full input scan, never as a head slice.")
    ap.add_argument("--per-type", type=int, default=0,
                    help="Collect at most N records PER task_type (0 = disabled). "
                         "Combines with --n as an overall cap.")
    ap.add_argument(
        "--task-type-filter", default="",
        help="Comma-separated task_types to keep. Empty (default) = all.",
    )
    a = ap.parse_args()
    filter_set = {t.strip() for t in a.task_type_filter.split(",") if t.strip()} or None
    d = json.load(open(a.input))
    out = []
    type_counts: dict[str, int] = {}
    # Scan the FULL input: filter by task_type first, then apply the quotas.
    # (The old version sliced d[:n] up front, so task_types living deeper in the
    # file could never be collected.)
    for i, r in enumerate(d):
        if a.n >= 0 and len(out) >= a.n:
            break
        tt = str(r.get("task_type", ""))
        if filter_set is not None and tt not in filter_set:
            continue
        if a.per_type > 0 and type_counts.get(tt, 0) >= a.per_type:
            # early exit: every requested task_type already met its quota
            if filter_set is not None and all(
                type_counts.get(t, 0) >= a.per_type for t in filter_set
            ):
                break
            continue
        rr = dict(r)
        wrote = False
        for side in ("left", "right"):
            if r.get(f"dex_grasp_{side}"):
                rr[f"pred_{side}"] = r[f"dex_grasp_{side}"]
                wrote = True
        if not wrote:
            continue
        # cand_idx is a synthetic ordering key when the record lacks its own
        # (right/bidex records have no cand_idx); it only groups variants.
        rr["candidate_idx"] = int(r.get("cand_idx", i))
        type_counts[tt] = type_counts.get(tt, 0) + 1
        out.append(rr)
    json.dump(out, open(a.output, "w"), ensure_ascii=False)
    print(f"wrote {len(out)} GT-as-pred records -> {a.output}")
    print("per task_type: "
          + (", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())) or "(none)"))


if __name__ == "__main__":
    main()
