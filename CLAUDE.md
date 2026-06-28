# CLAUDE.md

## Project Overview

DexVLG bimanual dexterous grasp generation model. Generates language-conditioned grasp poses for two Shadow Hands (left + right) given a colored point cloud and natural language instruction.

## Tech Stack

- Python 3.10+, PyTorch 2.0+
- HuggingFace Transformers (BERT)
- Trimesh (mesh loading)
- Open3D (visualization, optional)

## Repository Layout

- `models/` — Neural network modules (DexVLG, PointNet++, FlowMatching, PoseDecoder)
- `data/` — Dataset and dataloader
- `utils/` — Rotation math, misc helpers, visualization
- `configs/` — YAML configuration
- `train.py` — Two-stage training with DDP
- `test.py` — Evaluation with metrics
- `inference.py` — Single-sample demo
- `test_sanity.py` — Quick model sanity check (no GPU needed)

## Common Commands

```bash
# Run sanity check (verifies all components work)
python test_sanity.py

# Train stage 1 (full model)
python train.py --config configs/default.yaml --stage 1

# Train stage 2 (fine-tune flow head only)
python train.py --config configs/default.yaml --stage 2 --resume outputs/checkpoints/stage1/best.pt

# Evaluate
python test.py --config configs/default.yaml --checkpoint outputs/checkpoints/stage1/best.pt

# Single inference
python inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/stage1/best.pt --mesh_path /path/to/mesh.obj --instruction "Grasp the mug"
```

## Code Conventions

- Type hints on all function signatures
- Docstrings on public classes and functions (Args/Returns format)
- No comments unless explaining non-obvious WHY
- 6D rotation representation throughout (Gram-Schmidt, not quaternion)
- Quaternion convention: wxyz (real part first)
- Pose vector layout: [trans(3), rot6d(6), joints(22)] = 31D per hand

## Key Design Decisions

- PointNet++ over Uni3D: pure PyTorch, no custom CUDA kernels needed
- BERT over Florence-2: encoder-only is sufficient for instruction encoding, much lighter
- Fusion Transformer: separate module rather than using LLM internals for fusion
- SimpleTokenizer fallback: allows running without downloading BERT weights (testing/CI)
- Flow-matching (not diffusion): deterministic ODE sampling, no noise schedule tuning

## Testing

```bash
python test_sanity.py  # 4 tests: rotation, pointnet2, flow_matching, full_model
```

All tests run on CPU, no GPU required. Tests verify tensor shapes and round-trip correctness.

## Data Convention (LGBiDex format)

Splits are JSON arrays of records, one grasp per record. Input processing in
`data/dataset.py` (and `data/lgbidex_io.py`) mirrors the dexvlm pipeline's
`LgbiDexDataset` so both reproductions share coordinate conventions.

Per-object files under `mesh_root`:
- Colored point cloud: `{mesh_root}/{obj_id}/pc_part/simplified_part_{N}.ply`
- Mesh (fallback / bbox): `{mesh_root}/{obj_id}/mesh/simplified.obj`

Record fields:
- `obj_id`, `obj_pose` (7D = translation + wxyz quaternion), `obj_scale` (~0.1)
- `guidance` (instruction text), `cate_id`, `action`
- `dex_grasp_left` / `dex_grasp_right`: 28D `[trans(3), axis_angle(3), joints(22)]`

Processing: point cloud is bbox-centered and placed by the object rotation/scale
(`center_on_object`); grasp translations are shifted into the same frame and the
**axis-angle** rotation is converted to 6D. Output per hand is the 31D pose vector
`[trans(3), rot6d(6), joints(22)]`; the batch `gt_poses` is `(B, 2, 31)`.

Local paths (this machine):
- mesh_root: `/home/jiaxuan/wowowowo/slai/Dataset/MeshProcess/assets/object/oakink_obj/processed_data`
- splits: `/home/jiaxuan/wowowowo/slai/Dataset/lgbidex/{train_v3,test_v3}.json`

Run training with the `dexvlg` conda env:
`/mnt/conda/jiaxuan/miniconda3/envs/dexvlg/bin/python train.py --config configs/default.yaml --stage 1`
