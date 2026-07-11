#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_gays_fid.py  —  深度 FID（复用 GAYS 官方实现, 左右手分别, pred vs GT）

必须在 dexgys 环境运行:
    source /mnt/conda/jiaxuan/miniconda3/etc/profile.d/conda.sh && conda activate dexgys

!!! 硬件 + 权重前提 !!!
1) GPU: HandModel 的 FK/表面点采样(pytorch3d) 与 PointNet++ 前向都在 CUDA 上。
   dexgys 的 torch1.10 在 RTX 5090(sm_120) 上 CUDA 会 hang -> 纯 5090 机器跑不通,
   需 sm<=86 的卡(3090/A100 等)。
2) 预训练权重: scripts/fid/feature_extractor.get_model() 通过
   point_e.models.download.load_checkpoint("pointnet") 加载 PointNet++,
   会从 https://openaipublic.azureedge.net/main/point-e/pointnet.pt 下载到
   `./point_e_model_cache/pointnet.pt`(cwd 相对)。若离线且没缓存则失败。
   >> 本脚本不自动下载。请预先把 pointnet.pt 放到运行目录的 point_e_model_cache/ 下,
      或设置 cache_dir(见下 --pointnet-cache)。缺权重时脚本会报明确错误并退出。

流程(与 scripts/fid/test_fid.py 完全一致, 只是左右手分别 + 用转换后的 per-hand json):
  手表面点(HandModel) 与物体点云(get_obj_points, 1024 点)拼接 -> normalize ->
  PointNet++ 特征 -> point_e.evals.fid_is.compute_statistics -> frechet_distance。

左手: HandModel(flag='left'), 用 shadow_hand_left.xml + *_left 的 contact/penetration json
      (GAYS 官方 HandModel 已内置 flag='left' 分支)。28D 位姿直接喂, 不翻关节。

输入: convert_to_gays.py 产出的 results_left.json / results_right.json。
      预测取每条 predictions(N×28)全部; GT 取每条 dex_grasps(M×28, 若无则该侧无 FID)。
覆盖范围: 仅 left/lgbidex（物体点云经 OakInk 码 obj_id_old 解析）；right(YCB)/
      bidex(sem_*) 无 OakInk mesh —— 开头按 obj_id_old 有无过滤，并打印每
      task_type 覆盖/跳过数（不静默）。

用法:
  CUDA_VISIBLE_DEVICES=<free_gpu> python eval_gays_fid.py \
      -l results_left.json -r results_right.json -o fid.json \
      [--pointnet-cache /path/to/dir_with_pointnet.pt] [--batch-size 60]
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.functional import Tensor

GAYS_ROOT = "/home/jiaxuan/isee2025/Grasp-as-You-Say"
os.chdir(GAYS_ROOT)
sys.path.insert(0, GAYS_ROOT)

from functools import lru_cache  # noqa: E402

from point_e.evals.fid_is import compute_statistics  # noqa: E402
from model.utils.hand_model import HandModel  # noqa: E402
from scripts.fid.feature_extractor import get_model  # noqa: E402
from scripts.fid.test_fid import get_obj_points as _raw_get_obj_points, normalize_point_clouds  # noqa: E402


# 按 obj_id 缓存物体点云：GAYS 的 get_obj_points 每次都重载 mesh + 采样 1024 点；
# 全量 8w×2 条会对同一批物体重复采样上千次，缓存后同物体只采一次。
@lru_cache(maxsize=None)
def get_obj_points(oid):
    return _raw_get_obj_points(oid)


class HandCfgRight:
    mjcf_path = "./data/mjcf/shadow_hand.xml"
    mesh_path = "./data/mjcf/meshes"
    n_surface_points = 1024
    contact_points_path = "./data/mjcf/contact_points.json"
    penetration_points_path = "./data/mjcf/penetration_points.json"
    fingertip_points_path = "./data/mjcf/fingertip.json"


class HandCfgLeft(HandCfgRight):
    # HandModel(flag='left') 读取这些 *_left 字段
    mjcf_path_left = "./data/mjcf/shadow_hand_left.xml"
    contact_points_path_left = "./data/mjcf/contact_points_left.json"
    penetration_points_path_left = "./data/mjcf/penetration_points_left.json"


class PredDataset(Dataset):
    """每条记录展开成多条: predictions(N×28) -> N 个样本, 各配同一 obj 点云。"""
    def __init__(self, results, use_gt=False):
        self.items = []
        for r in results:
            # 物体点云走 OakInk 码 obj_id_old；旧格式 results 无该键时回退 obj_id
            # （旧转换器把 OakInk 码放在 obj_id）。
            oid = r.get("obj_id_old") or r["obj_id"]
            if use_gt:
                # 本项目转换器给 dex_grasps=[M×28]（每组多条 GT）；兼容旧的单条 dex_grasp。
                gts = r.get("dex_grasps")
                if gts is None:
                    gts = [r["dex_grasp"]] if "dex_grasp" in r else []
                for g in gts:
                    self.items.append((g, oid))
            else:
                for p in r["predictions"]:
                    self.items.append((p, oid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pose, oid = self.items[i]
        pose = torch.tensor(pose, dtype=torch.float).squeeze()
        while pose.shape[0] != 28:
            pose = pose[0]
        return {"pose": pose, "obj_pc": get_obj_points(oid)}

    @staticmethod
    def collate_fn(batch):
        return {
            "pose": torch.stack([b["pose"] for b in batch]),
            "obj_pc": torch.stack([b["obj_pc"] for b in batch]),
        }


def extract_features(loader, hand_model, feat, device):
    feats = []
    for batch in loader:
        obj_pc = batch["obj_pc"].to(device)
        hand_pc = hand_model(batch["pose"].to(device), with_surface_points=True)["surface_points"]
        inp = normalize_point_clouds(torch.cat([obj_pc, hand_pc], dim=1)).transpose(-1, -2)
        _, _, f = feat(inp, features=True)
        feats.append(f.cpu().detach())
    return torch.cat(feats, dim=0).numpy()


def filter_mesh_covered(name, results):
    """FID 只能评 OakInk mesh 可解析的记录：按 obj_id_old 有无过滤。

    打印每 task_type 的覆盖/跳过组数与 pred 条数 —— right(YCB)/bidex(sem_*) 不在
    OakInk mesh 集内，显式跳过而非静默丢弃。旧格式 results（无 task_type/
    obj_id_old 键，obj_id 本身就是 OakInk 码）整体原样保留。"""
    if results and all(("task_type" not in r and "obj_id_old" not in r) for r in results):
        print(f"[{name}] legacy results 格式（无 task_type/obj_id_old），"
              f"全部 {len(results)} 组按 obj_id 解析 mesh", flush=True)
        return results

    def _cnt(rs):
        g, p = Counter(), Counter()
        for r in rs:
            tt = str(r.get("task_type", "unknown"))
            g[tt] += 1
            p[tt] += len(r.get("predictions", []))
        return g, p

    tg, tp = _cnt(results)
    kept = [r for r in results if r.get("obj_id_old")]
    kg, kp = _cnt(kept)
    for tt in sorted(tg):
        print(f"[{name}] task_type={tt:<8} 覆盖 {kg.get(tt, 0)}/{tg[tt]} 组 "
              f"({kp.get(tt, 0)}/{tp[tt]} preds)"
              + ("" if kg.get(tt, 0) else "  <- 无 OakInk mesh (obj_id_old 缺失)，跳过"),
              flush=True)
    print(f"[{name}] FID 覆盖范围=仅 left/lgbidex（OakInk mesh）；"
          f"right(YCB)/bidex(sem_*) 的 mesh 构建为后续工作", flush=True)
    return kept


def run_side(name, results, hand_model, feat, device, batch_size):
    pred_ds = PredDataset(results, use_gt=False)
    gt_ds = PredDataset(results, use_gt=True)
    if len(pred_ds) == 0 or len(gt_ds) == 0:
        return {"side": name, "note": f"pred={len(pred_ds)} gt={len(gt_ds)}, cannot compute FID"}
    pl = DataLoader(pred_ds, batch_size=batch_size, collate_fn=PredDataset.collate_fn,
                    num_workers=4, shuffle=True)
    gl = DataLoader(gt_ds, batch_size=batch_size, collate_fn=PredDataset.collate_fn,
                    num_workers=4, shuffle=True)
    fp = extract_features(pl, hand_model, feat, device)
    fg = extract_features(gl, hand_model, feat, device)
    stats_p = compute_statistics(fp)
    stats_g = compute_statistics(fg)
    fid = stats_p.frechet_distance(stats_g)
    return {"side": name, "n_pred": len(pred_ds), "n_gt": len(gt_ds), "fid": float(fid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", default=None)
    ap.add_argument("-r", "--right", default=None)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--pointnet-cache", default=None,
                    help="dir containing pointnet.pt (else point_e downloads to ./point_e_model_cache)")
    args = ap.parse_args()

    device = "cuda"
    feat = get_model(device=device, cache_dir=args.pointnet_cache)
    print("pointnet loaded", flush=True)

    out = {}
    for name, path in [("left", args.left), ("right", args.right)]:
        if path is None:
            continue
        with open(path) as f:
            res = json.load(f)
        if not res:
            print(f"[{name}] empty, skip", flush=True)
            continue
        res = filter_mesh_covered(name, res)
        if not res:
            print(f"[{name}] no mesh-covered entries (obj_id_old), skip", flush=True)
            continue
        cfg = HandCfgLeft() if name == "left" else HandCfgRight()
        flag = "left" if name == "left" else "right"
        hm = HandModel(cfg=cfg, device=device, flag=flag)
        out[name] = run_side(name, res, hm, feat, device, args.batch_size)
        print(f"[{name}] {out[name]}", flush=True)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved ->", args.output, flush=True)


if __name__ == "__main__":
    main()
