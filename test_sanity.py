"""Sanity check: verifies the model can run a forward pass and sampling."""

import sys

import torch


def test_rotation_utils():
    from utils.rotation import (
        quaternion_to_rotation_6d,
        rotation_6d_to_quaternion,
        rotation_6d_to_matrix,
    )

    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rot6d = quaternion_to_rotation_6d(quat)
    assert rot6d.shape == (1, 6), f"Expected (1, 6), got {rot6d.shape}"

    quat_back = rotation_6d_to_quaternion(rot6d)
    assert torch.allclose(quat, quat_back, atol=1e-5), "Round-trip quaternion failed"

    mat = rotation_6d_to_matrix(rot6d)
    assert mat.shape == (1, 3, 3), f"Expected (1, 3, 3), got {mat.shape}"
    assert torch.allclose(mat[0], torch.eye(3), atol=1e-5), "Identity rotation failed"

    print("[PASS] Rotation utilities")


def test_matrix_to_quaternion_degenerate():
    """180-degree rotations with tied diagonal maxima (trace <= 0 branch).

    R(180deg, n) = 2 n n^T - I has trace -1; for n = [1,1,0]/sqrt(2) and
    [1,1,1]/sqrt(3) the diagonal has tied maxima, which the old strict-">"
    branch selection mapped to an all-zero quaternion."""
    from utils.rotation import matrix_to_quaternion, quaternion_to_matrix

    axes = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    axes = axes / axes.norm(dim=-1, keepdim=True)
    mats = 2.0 * axes.unsqueeze(-1) * axes.unsqueeze(-2) - torch.eye(3)

    quat = matrix_to_quaternion(mats)
    assert torch.isfinite(quat).all(), "degenerate case produced non-finite quaternion"
    norms = quat.norm(dim=-1)
    assert bool((norms > 0.5).all()), (
        f"degenerate 180-degree rotation produced a (near-)zero quaternion: norms={norms.tolist()}"
    )

    back = quaternion_to_matrix(quat)
    err = (back - mats).abs().max().item()
    assert err < 1e-5, f"180-degree matrix->quat->matrix round-trip error {err:.2e} >= 1e-5"

    print("[PASS] matrix_to_quaternion 180-degree tied-diagonal round-trip")


def test_pointnet2():
    from models.pointnet2 import PointNet2Encoder

    encoder = PointNet2Encoder(in_channels=6, num_output_tokens=32, output_dim=128)

    B, N = 2, 1024
    xyz = torch.randn(B, N, 3)
    rgb = torch.rand(B, N, 3)
    tokens = encoder(xyz, rgb)
    assert tokens.shape == (B, 32, 128), f"Expected (2, 32, 128), got {tokens.shape}"

    print("[PASS] PointNet2Encoder")


def test_flow_matching():
    from models.flow_matching import FlowMatchingTransformer

    pose_dim = 31
    fm = FlowMatchingTransformer(
        pose_dim=pose_dim, num_queries=2, dim=64, depth=2,
        num_heads=4, cond_dim=128,
    )

    B = 2
    x_t = torch.randn(B, 2, pose_dim)
    t = torch.rand(B)
    cond = torch.randn(B, 10, 128)

    v = fm(x_t, t, cond)
    assert v.shape == (B, 2, pose_dim), f"Expected (2, 2, 31), got {v.shape}"

    print("[PASS] FlowMatchingTransformer")


def test_dexvlg_forward():
    from models.dexvlg import DexVLG

    cfg = {
        "pc_in_channels": 6,
        "pc_feature_dim": 64,
        "num_pc_tokens": 16,
        "bert_model": "bert-base-uncased",
        "bert_dim": 768,
        "freeze_bert_embeddings": True,
        "fusion_depth": 1,
        "flow_hidden_dim": 64,
        "flow_depth": 2,
        "flow_heads": 4,
        "joint_dim": 22,
    }

    model = DexVLG(cfg)
    model.eval()

    B, N = 2, 256
    xyz = torch.randn(B, N, 3)
    rgb = torch.rand(B, N, 3)
    texts = ["Grasp the handle of the mug", "Pick up the cup by the rim"]

    pose_dim = 3 + 6 + 22
    gt_poses = torch.randn(B, 2, pose_dim)
    loss_output = model(xyz, rgb, texts, gt_poses=gt_poses)
    assert "loss" in loss_output, "Missing 'loss' key"
    assert loss_output["loss"].dim() == 0, "Loss should be scalar"
    print(f"  Training loss: {loss_output['loss'].item():.4f}")

    with torch.no_grad():
        sample_output = model.sample(xyz, rgb, texts, num_steps=3)
    assert "left_translation" in sample_output
    assert "right_translation" in sample_output
    assert sample_output["left_translation"].shape == (B, 3)
    assert sample_output["right_joints"].shape == (B, 22)
    print("  Sampling output shapes verified")

    print("[PASS] DexVLG forward + sampling")


def _make_latent_ar_model():
    """Tiny CPU latent_ar DexVLG for the multi-task sanity tests."""
    from models.dexvlg import DexVLG

    cfg = {
        "architecture": "latent_ar",
        "pc_in_channels": 3,
        "pc_feature_dim": 32,
        "num_pc_tokens": 8,
        "bert_model": "bert-base-uncased",
        "bert_dim": 96,
        "freeze_bert_embeddings": True,
        "fusion_depth": 1,
        "flow_hidden_dim": 32,
        "flow_depth": 1,
        "flow_heads": 4,
        "joint_dim": 22,
        "reasoner": {
            "dim": 32,
            "depth": 1,
            "num_heads": 4,
            "num_thinking_tokens": 2,
            "max_hands": 2,
            "hand_cond_noise_std": 0.1,
        },
        "loss_weights": {"flow": 1.0, "presence": 0.5, "side": 0.5},
    }
    model = DexVLG(cfg)
    model.eval()
    return model


def test_multitask_collate():
    from data.dataset import collate_fn

    def item(hand_poses, side_ids, task_type, l_obj):
        return {
            "xyz": torch.randn(64, 3),
            "rgb": torch.rand(64, 3),
            "text": f"grasp for {task_type}",
            "hand_poses": hand_poses,
            "hand_side_ids": side_ids,
            "L_obj": l_obj,
            "task_type": task_type,
            "obj_id": "obj",
            "cate_id": "cate",
            "group_id": "g",
        }

    left_pose = torch.full((31,), 1.0)
    right_pose = torch.full((31,), 2.0)
    batch = collate_fn([
        item([left_pose], [0], "left", 0.1),
        item([right_pose], [1], "right", 0.2),
        item([left_pose, right_pose], [0, 1], "lgbidex", 0.3),
    ])

    assert batch["gt_poses"].shape == (3, 2, 31), batch["gt_poses"].shape
    assert batch["hand_mask"].dtype == torch.bool
    assert batch["hand_mask"].tolist() == [[True, False], [True, False], [True, True]]
    assert batch["hand_side_ids"].dtype == torch.long
    assert batch["hand_side_ids"].tolist() == [[0, -1], [1, -1], [0, 1]]
    assert torch.equal(batch["gt_poses"][0, 0], left_pose)
    assert torch.equal(batch["gt_poses"][0, 1], torch.zeros(31)), "pad slot must be zero"
    assert torch.equal(batch["gt_poses"][1, 0], right_pose)
    assert torch.equal(batch["gt_poses"][2, 1], right_pose)
    assert batch["L_obj"].dtype == torch.float32
    assert torch.allclose(batch["L_obj"], torch.tensor([0.1, 0.2, 0.3]))
    assert batch["task_types"] == ["left", "right", "lgbidex"]

    print("[PASS] Multi-task variable-hand collate")


def test_relquantile_roundtrip():
    from data.pose_normalizer import LgbidexPoseNormalizer

    norm = LgbidexPoseNormalizer(
        hand_side="left",
        joint_dim=22,
        mode="relquantile11",
        rel_quantile={"q01": [-1.0, -1.0, -0.2], "q99": [1.0, 1.0, 1.0]},
    )
    torch.manual_seed(0)
    pose = torch.cat(
        [torch.randn(4, 3) * 0.05, torch.randn(4, 6), torch.rand(4, 22) * 0.5],
        dim=-1,
    )
    l_obj = torch.tensor([0.1, 0.2, 0.4, 0.8])

    normalized = norm.normalize_pose(pose, L_obj=l_obj)
    back = norm.denormalize_pose(normalized, L_obj=l_obj)
    assert torch.allclose(pose, back, atol=1e-5), "relquantile round-trip failed"

    scaled = pose.clone()
    scaled[:, :3] *= 2.0
    normalized_scaled = norm.normalize_pose(scaled, L_obj=l_obj * 2.0)
    assert torch.allclose(normalized, normalized_scaled, atol=1e-5), (
        "normalization must be invariant to a joint (translation, L_obj) rescale"
    )

    single = norm.normalize_pose(pose[0], L_obj=float(l_obj[0]))
    assert torch.allclose(
        norm.denormalize_pose(single, L_obj=float(l_obj[0])), pose[0], atol=1e-5
    )

    try:
        norm.normalize_pose(pose)
    except ValueError:
        pass
    else:
        raise AssertionError("relquantile11 without L_obj must raise")

    print("[PASS] relquantile11 round-trip + L_obj scale invariance")


def test_latent_ar_mixed_batch():
    model = _make_latent_ar_model()

    B, N = 2, 256
    torch.manual_seed(1)
    xyz = torch.randn(B, N, 3)
    rgb = torch.rand(B, N, 3)
    texts = ["grasp with both hands", "grasp with the right hand"]
    gt_poses = torch.randn(B, 2, 31)
    gt_poses[1, 1] = 0.0
    hand_mask = torch.tensor([[True, True], [True, False]])
    hand_side_ids = torch.tensor([[0, 1], [1, -1]])

    torch.manual_seed(0)
    out = model(
        xyz, rgb, texts,
        gt_poses=gt_poses, hand_mask=hand_mask, hand_side_ids=hand_side_ids,
    )
    for key in ("loss", "loss_flow", "loss_presence", "loss_side"):
        assert key in out, f"missing '{key}'"
        assert torch.isfinite(out[key]), f"non-finite '{key}'"
    print(f"  Mixed-batch loss: {out['loss'].item():.4f}")

    garbage = gt_poses.clone()
    garbage[1, 1] = 1e3
    torch.manual_seed(0)
    out_garbage = model(
        xyz, rgb, texts,
        gt_poses=garbage, hand_mask=hand_mask, hand_side_ids=hand_side_ids,
    )
    assert torch.allclose(out["loss_flow"], out_garbage["loss_flow"], atol=1e-6), (
        "masked pad slot leaked into the flow loss"
    )
    assert torch.allclose(out["loss"], out_garbage["loss"], atol=1e-6), (
        "masked pad slot leaked into the total loss"
    )
    print("  Pad-slot GT garbage leaves the loss unchanged (mask verified)")

    print("[PASS] latent_ar mixed-batch loss + slot masking")


def test_latent_ar_sample_contract():
    model = _make_latent_ar_model()

    B, N = 2, 256
    torch.manual_seed(2)
    xyz = torch.randn(B, N, 3)
    rgb = torch.rand(B, N, 3)
    texts = ["grasp the box", "hand over the bottle"]

    with torch.no_grad():
        out = model.sample_latent_ar(xyz, rgb, texts, num_steps=2)
    assert out["poses"].shape == (B, 2, 31)
    assert out["hand_mask"].shape == (B, 2) and out["hand_mask"].dtype == torch.bool
    assert out["hand_sides"].shape == (B, 2) and out["hand_sides"].dtype == torch.long
    assert bool(out["hand_mask"][:, 0].all()), "hand slot 0 must always be emitted"
    assert bool(((out["hand_sides"] >= -1) & (out["hand_sides"] <= 1)).all())
    assert bool((out["hand_sides"][~out["hand_mask"]] == -1).all())
    assert bool((out["poses"][~out["hand_mask"]] == 0).all()), "pad poses must be zero"

    force_mask = torch.tensor([[True, False], [True, True]])
    force_sides = torch.tensor([[1, -1], [0, 1]])
    with torch.no_grad():
        forced = model.sample_latent_ar(
            xyz, rgb, texts, num_steps=2,
            force_sides=force_sides, force_mask=force_mask,
        )
    assert torch.equal(forced["hand_mask"], force_mask)
    assert torch.equal(forced["hand_sides"], torch.tensor([[1, -1], [0, 1]]))
    assert bool((forced["poses"][0, 1] == 0).all())

    print("[PASS] latent_ar sample output contract (free + forced)")


def _make_grace_model():
    """Tiny CPU latent_ar DexVLG with GRACE enabled for the sanity test."""
    from models.dexvlg import DexVLG

    cfg = {
        "architecture": "latent_ar",
        "pc_in_channels": 3,
        "pc_feature_dim": 32,
        "num_pc_tokens": 8,
        "bert_model": "bert-base-uncased",
        "bert_dim": 96,
        "freeze_bert_embeddings": True,
        "fusion_depth": 1,
        "flow_hidden_dim": 32,
        "flow_depth": 1,
        "flow_heads": 4,
        "joint_dim": 22,
        "reasoner": {
            "dim": 32,
            "depth": 1,
            "num_heads": 4,
            "num_thinking_tokens": 2,
            "max_hands": 2,
            "hand_cond_noise_std": 0.1,
        },
        "loss_weights": {"flow": 1.0, "presence": 0.5, "side": 0.5},
        "grace": {
            "enabled": True,
            "num_approach_anchors": 4,
            "hidden_dim": 16,
            "condition": True,
            "w_contact": 0.5,
            "w_anchor": 0.5,
            "w_approach": 0.25,
            "w_rel_rot": 0.5,
        },
    }
    model = DexVLG(cfg)
    model.eval()
    return model


def test_grace_latent_ar():
    model = _make_grace_model()

    B, N = 2, 256
    torch.manual_seed(3)
    xyz = torch.randn(B, N, 3)
    rgb = torch.rand(B, N, 3)
    texts = ["grasp with both hands", "grasp with the right hand"]
    gt_poses = torch.randn(B, 2, 31)
    gt_poses[1, 1] = 0.0
    hand_mask = torch.tensor([[True, True], [True, False]])
    hand_side_ids = torch.tensor([[0, 1], [1, -1]])
    grasp_center = torch.randn(B, 2, 3) * 0.1
    approach_dir = torch.nn.functional.normalize(torch.randn(B, 2, 3), dim=-1)
    # pad slot mirrors the collate zero-padding: grasp_center/approach_dir are
    # exactly [0,0,0] there. The zero-norm approach_dir is the critical case —
    # a norm-divided cosine would give 0/0 = NaN and silently poison the loss.
    grasp_center[1, 1] = 0.0
    approach_dir[1, 1] = 0.0
    log_l_obj = torch.randn(B)

    torch.manual_seed(0)
    out = model(
        xyz, rgb, texts,
        gt_poses=gt_poses, hand_mask=hand_mask, hand_side_ids=hand_side_ids,
        log_l_obj=log_l_obj, grasp_center=grasp_center, approach_dir=approach_dir,
    )
    for key in (
        "loss", "loss_flow", "loss_presence", "loss_side",
        "loss_contact", "loss_anchor", "loss_approach", "loss_rel_rot",
    ):
        assert key in out, f"missing '{key}'"
        assert torch.isfinite(out[key]), f"non-finite '{key}' (zero-norm pad NaN?)"
    print(f"  GRACE loss: {out['loss'].item():.4f}")

    # backward: every parameter that receives gradient must stay finite
    # (guards the F.normalize-near-zero exploding-grad path in the HOW head).
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite GRACE grad"

    # pad-slot invariance incl. the zero-norm approach_dir case: garbage in the
    # padded slot must not change the loss (masking verified).
    garbage = gt_poses.clone(); garbage[1, 1] = 1e3
    gc2 = grasp_center.clone(); gc2[1, 1] = 1e3
    torch.manual_seed(0)
    out2 = model(
        xyz, rgb, texts,
        gt_poses=garbage, hand_mask=hand_mask, hand_side_ids=hand_side_ids,
        log_l_obj=log_l_obj, grasp_center=gc2, approach_dir=approach_dir,
    )
    assert torch.isfinite(out2["loss"]), "pad-slot garbage NaN-poisoned the loss"
    assert torch.allclose(out["loss"], out2["loss"], atol=1e-4), (
        "masked pad slot leaked into the GRACE loss"
    )
    print("  Pad-slot garbage (incl. zero-norm approach) leaves the loss unchanged")

    # sampling rebuilds GRACE tokens from the model's own predicted hiddens
    with torch.no_grad():
        s = model.sample_latent_ar(xyz, rgb, texts, num_steps=2)
    assert s["poses"].shape == (B, 2, 31)
    assert bool(s["hand_mask"][:, 0].all()), "hand slot 0 must always be emitted"

    print("[PASS] GRACE latent_ar loss + pad-mask + backward + sampling")


def test_metrics_alignment():
    from utils.metrics import (
        align_hands_by_side,
        relative_pose_error,
        structure_metrics,
        task_type_index,
    )

    D = 31
    torch.manual_seed(3)
    gt_poses = torch.randn(3, 2, D)
    gt_mask = torch.tensor([[True, True], [True, False], [True, False]])
    gt_sides = torch.tensor([[0, 1], [0, -1], [1, -1]])

    pred_poses = torch.randn(3, 2, D)
    pred_poses[0, 1] = gt_poses[0, 0]  # left prediction stored in slot 1
    pred_poses[0, 0] = gt_poses[0, 1]  # right prediction stored in slot 0
    pred_mask = torch.tensor([[True, True], [True, False], [True, True]])
    pred_sides = torch.tensor([[1, 0], [1, -1], [0, 1]])

    aligned, matched, align_idx = align_hands_by_side(
        pred_poses, pred_mask, pred_sides, gt_poses, gt_mask, gt_sides
    )
    assert matched.tolist() == [[True, True], [False, False], [True, False]]
    assert align_idx.tolist() == [[1, 0], [-1, -1], [1, -1]]
    assert torch.equal(aligned[0, 0], gt_poses[0, 0]), "swapped slots must realign"
    assert torch.equal(aligned[0, 1], gt_poses[0, 1])
    assert bool((aligned[1] == 0).all()), "unmatched GT slots must be zero"
    assert torch.equal(aligned[2, 0], pred_poses[2, 1])

    count_ok, side_ok, struct_ok = structure_metrics(
        pred_mask, pred_sides, gt_mask, gt_sides
    )
    assert count_ok.tolist() == [True, True, False]
    assert side_ok.tolist() == [True, False, False]
    assert struct_ok.tolist() == [True, False, False]

    pair = torch.randn(2, 2, D)
    rel_trans, rel_rot = relative_pose_error(pair, pair)
    assert torch.allclose(rel_trans, torch.zeros(2), atol=1e-5)
    assert torch.allclose(rel_rot, torch.zeros(2), atol=1e-3)
    shifted = pair.clone()
    shifted[:, 1, 0] += 0.1
    rel_trans_shift, _ = relative_pose_error(shifted, pair)
    assert torch.allclose(rel_trans_shift, torch.full((2,), 0.1), atol=1e-5)

    ids = task_type_index(["left", "bidex", "lgbidex", "right"])
    assert ids.tolist() == [0, 3, 2, 1]
    try:
        task_type_index(["left", "banana"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown task_type must raise")

    print("[PASS] metrics: side alignment + structure + relative pose")


def main():
    print("=" * 50)
    print("DexVLG Sanity Check")
    print("=" * 50)

    tests = [
        ("Rotation utilities", test_rotation_utils),
        ("matrix_to_quaternion degenerate", test_matrix_to_quaternion_degenerate),
        ("PointNet2 encoder", test_pointnet2),
        ("Flow-Matching Transformer", test_flow_matching),
        ("DexVLG full model", test_dexvlg_forward),
        ("Multi-task collate", test_multitask_collate),
        ("relquantile11 normalizer", test_relquantile_roundtrip),
        ("latent_ar mixed batch", test_latent_ar_mixed_batch),
        ("latent_ar sample contract", test_latent_ar_sample_contract),
        ("GRACE latent_ar", test_grace_latent_ar),
        ("Metrics alignment", test_metrics_alignment),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            print(f"\nTesting {name}...")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 50)

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
