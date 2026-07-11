#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_gays_q1.py  —  Q1 / pen / valid_q1（复用 GAYS 官方实现，左右手分别）

必须在 dexgys 环境运行:
    source /mnt/conda/jiaxuan/miniconda3/etc/profile.d/conda.sh && conda activate dexgys

!!! 硬件前提 !!!
本脚本用 csdf.compute_sdf（CUDA-only, 无 CPU 回退）+ pytorch3d GPU 采样, 因此 **必须有
可用 GPU**。dexgys 的 torch1.10/cu113 在 RTX 5090(sm_120) 上任何 CUDA 算子都会 hang,
所以在纯 5090 机器上跑不通(见 docs/指标评测说明.md 的 blocker 说明)。
需要一张 sm<=86 的卡(如 3090/A100)或升级到 cu128 的 torch 才能跑。

直接 import GAYS 的 cal_q1 / cal_pen / KaolinModel(csdf mesh SDF) + ShallowHandModel。
左手: 复用 GAYS 官方双手可视化的做法(scripts/vis/bibodex_vis.py) —— 把 mjcf 换成
      shadow_hand_left.xml(X 轴镜像), 28D 位姿直接喂, 不翻关节符号。

输入: convert_to_gays.py 产出的 results_left.json / results_right.json
      (每条 obj_id / obj_id_old / task_type / guidance / predictions=N×28)。
覆盖范围: 仅 left/lgbidex（mesh 经 data/oakink/<obj_id_old>/coacd 解析）；
      right(YCB)/bidex(sem_*) 无 OakInk mesh —— 开头按 obj_id_old 有无过滤，
      并打印每 task_type 覆盖/跳过数（不静默）。

用法:
  CUDA_VISIBLE_DEVICES=<free_gpu> python eval_gays_q1.py \
      -l results_left.json -r results_right.json -o q1.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean, pstdev

import torch

GAYS_ROOT = "/home/jiaxuan/isee2025/Grasp-as-You-Say"
os.chdir(GAYS_ROOT)          # GAYS 代码用相对路径 ./data/...
sys.path.insert(0, GAYS_ROOT)

import pytorch_kinematics as pk  # noqa: E402
from utils.config_utils import EasyConfig  # noqa: E402
from utils.shadowhand import ShallowHandModel  # noqa: E402
from scripts.q1.eval_utils import cal_q1, cal_pen, KaolinModel  # noqa: E402


def filter_mesh_covered(name, results):
    """Q1/FID 只能评 OakInk mesh 可解析的记录：按 obj_id_old 有无过滤。

    打印每 task_type 的覆盖/跳过组数与 pred 条数 —— right(YCB)/bidex(sem_*) 不在
    OakInk coacd mesh 集内，显式跳过而非静默丢弃。旧格式 results（无 task_type/
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
    print(f"[{name}] Q1/FID 覆盖范围=仅 left/lgbidex（OakInk mesh）；"
          f"right(YCB)/bidex(sem_*) 的 coacd mesh 构建为后续工作", flush=True)
    return kept


def make_hand_model(side, device):
    """左手: 复用 ShallowHandModel 但把 chain 换成 shadow_hand_left.xml。
    ShallowHandModel 硬编码右手 mjcf, 这里构建后替换其 chain（对齐 bibodex_vis 用
    HandModel(mjcf_path=shadow_hand_left.xml) 的做法, 数学一致）。"""
    hm = ShallowHandModel(device=device)
    if side == "left":
        left_mjcf = os.path.join(GAYS_ROOT, "data/mjcf/shadow_hand_left.xml")
        # 用左手 mjcf 重建整个 mesh/chain: 直接再造一个实例最省心, 但 ShallowHandModel
        # 的 mjcf 路径写死。故临时替换全局路径不可行 -> 用 monkeypatch: 重新 build。
        hm_left = _build_left(device, left_mjcf)
        return hm_left
    return hm


def _build_left(device, left_mjcf):
    """按 ShallowHandModel 的构造流程, 但 mjcf 用左手版。逻辑与 ShallowHandModel.__init__
    完全一致(复制其流程只为换 mjcf 路径, 数学不改)。"""
    import trimesh
    import pytorch3d.ops
    import pytorch3d.structures
    from csdf import index_vertices_by_faces

    mesh_path = os.path.join(GAYS_ROOT, "data/mjcf/meshes")
    n_surface_points = 1024
    contact_points_path = os.path.join(GAYS_ROOT, "data/mjcf/contact_points.json")
    penetration_points_path = os.path.join(GAYS_ROOT, "data/mjcf/penetration_points.json")
    fingertip_points_path = os.path.join(GAYS_ROOT, "data/mjcf/fingertip.json")

    hm = ShallowHandModel.__new__(ShallowHandModel)
    hm.device = torch.device(device)
    hm.chain = pk.build_chain_from_mjcf(open(left_mjcf).read()).to(dtype=torch.float, device=device)
    hm.n_dofs = len(hm.chain.get_joint_parameter_names())
    penetration_points = json.load(open(penetration_points_path))
    contact_points = json.load(open(contact_points_path))
    fingertip_points = json.load(open(fingertip_points_path))
    hm.mesh = {}
    areas = {}

    def build_mesh_recurse(body):
        if len(body.link.visuals) > 0:
            link_name = body.link.name
            link_vertices, link_faces, n_link_vertices = [], [], 0
            for visual in body.link.visuals:
                scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                if visual.geom_type == "box":
                    link_mesh = trimesh.load_mesh(os.path.join(mesh_path, "box.obj"), process=False)
                    link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                elif visual.geom_type == "capsule":
                    link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2)
                elif visual.geom_type == "mesh":
                    link_mesh = trimesh.load_mesh(os.path.join(mesh_path, visual.geom_param[0].split(":")[1] + ".obj"), process=False)
                    if visual.geom_param[1] is not None:
                        scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                pos = visual.offset.to(hm.device)
                vertices = vertices * scale
                vertices = pos.transform_points(vertices)
                link_vertices.append(vertices)
                link_faces.append(faces + n_link_vertices)
                n_link_vertices += len(vertices)
            link_vertices = torch.cat(link_vertices, dim=0)
            link_faces = torch.cat(link_faces, dim=0)
            cc = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3)
            pk_ = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3)
            fp = torch.tensor(fingertip_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3)
            lfv = index_vertices_by_faces(link_vertices, link_faces)
            hm.mesh[link_name] = dict(vertices=link_vertices, faces=link_faces,
                                      contact_candidates=cc, penetration_keypoints=pk_,
                                      fingertip_keypoints=fp, face_verts=lfv)
            if link_name not in ["robot0:palm", "robot0:lfmetacarpal_child"]:
                hm.mesh[link_name]["geom_param"] = body.link.visuals[0].geom_param
            areas[link_name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
        for children in body.children:
            build_mesh_recurse(children)

    build_mesh_recurse(hm.chain._root)
    total_area = sum(areas.values())
    num_samples = dict([(ln, int(areas[ln] / total_area * n_surface_points)) for ln in hm.mesh])
    num_samples["robot0:palm"] += n_surface_points - sum(num_samples.values())
    for ln in hm.mesh:
        if num_samples[ln] == 0:
            hm.mesh[ln]["surface_points"] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
            continue
        m = pytorch3d.structures.Meshes(hm.mesh[ln]["vertices"].unsqueeze(0), hm.mesh[ln]["faces"].unsqueeze(0))
        dpc = pytorch3d.ops.sample_points_from_meshes(m, num_samples=100 * num_samples[ln])
        sp = pytorch3d.ops.sample_farthest_points(dpc, K=num_samples[ln])[0][0]
        hm.mesh[ln]["surface_points"] = sp
    return hm


def eval_side(name, results, q1_cfg, hand_model, object_model, device):
    overall = {"q1": [], "pen": [], "valid_q1": []}
    per_obj = {}
    # 按 mesh 码排序让同物体连续 -> 最大化下面 initialize 的 mesh/SDF 缓存命中。
    results = sorted(results, key=lambda r: str(r.get("obj_id_old") or r.get("obj_id")))
    for res in results:
        # mesh 解析走 OakInk 码 obj_id_old（data/oakink/<code>/coacd）；
        # 旧格式 results 无该键时回退 obj_id（旧转换器把 OakInk 码放在 obj_id）。
        oid = res.get("obj_id_old") or res["obj_id"]
        act = res.get("guidance", res.get("action_id", "0"))
        key = f"{oid}_{act}"
        per_obj.setdefault(key, {"q1": [], "pen": [], "valid_q1": []})
        preds = torch.tensor(res["predictions"], dtype=torch.float, device=device)
        if preds.dim() == 3:
            preds = preds.squeeze(1)
        elif preds.dim() == 1:
            preds = preds.unsqueeze(0)
        for i in range(preds.shape[0]):
            q1 = cal_q1(q1_cfg, hand_model, object_model, oid, preds[i], device)
            pen = cal_pen(hand_model, object_model, oid, preds[i], device)
            vq = q1 if pen < q1_cfg["thres_pen"] else 0
            for k, v in [("q1", q1), ("pen", pen), ("valid_q1", vq)]:
                overall[k].append(v)
                per_obj[key][k].append(v)
    return _summ(name, overall, per_obj)


def _summ(name, overall, per_obj):
    def ms(x):
        return {"mean": mean(x) if x else 0.0, "std": pstdev(x) if len(x) > 1 else 0.0}
    obj_means = {k: {kk: mean(vv) for kk, vv in d.items()} for k, d in per_obj.items()}
    return {
        "side": name,
        "n_grasps": len(overall["q1"]),
        "mean_overall": {k: ms(overall[k]) for k in overall},
        "overall_max_pen": max(overall["pen"]) if overall["pen"] else 0.0,
        "mean_object": {
            k: (mean([m[k] for m in obj_means.values()]) if obj_means else 0.0)
            for k in ["q1", "pen", "valid_q1"]
        },
        "per_object": obj_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", default=None)
    ap.add_argument("-r", "--right", default=None)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    device = "cuda"
    cfg = EasyConfig()
    cfg.load("config/test_base.yaml")
    q1_cfg = dict(cfg.q1)
    print("q1 cfg:", q1_cfg, flush=True)

    object_model = KaolinModel("data/oakink", batch_size_each=1, device=device)
    # 按物体缓存 mesh/SDF：cal_q1/cal_pen 每条都会 initialize(object_code) 重载 mesh；
    # 同一物体连续时跳过重载（配合 eval_side 里按 obj_id 排序），大幅提速全量评测。
    _orig_initialize = object_model.initialize
    _init_cache = {"code": None}

    def _cached_initialize(code):
        if code == _init_cache["code"]:
            return
        _orig_initialize(code)
        _init_cache["code"] = code

    object_model.initialize = _cached_initialize

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
        print(f"[{name}] {len(res)} entries -> building {name} hand model", flush=True)
        hm = make_hand_model(name, device)
        out[name] = eval_side(name, res, q1_cfg, hm, object_model, device)
        m = out[name]["mean_overall"]
        print(f"[{name}] Q1={m['q1']['mean']:.5f}+-{m['q1']['std']:.5f} "
              f"pen={m['pen']['mean']:.5f} valid_q1={m['valid_q1']['mean']:.5f} "
              f"(n={out[name]['n_grasps']})", flush=True)

    save = os.path.join(os.getcwd(), args.output) if not os.path.isabs(args.output) else args.output
    with open(args.output if os.path.isabs(args.output) else save, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved ->", args.output, flush=True)


if __name__ == "__main__":
    main()
