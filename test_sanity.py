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


def main():
    print("=" * 50)
    print("DexVLG Sanity Check")
    print("=" * 50)

    tests = [
        ("Rotation utilities", test_rotation_utils),
        ("PointNet2 encoder", test_pointnet2),
        ("Flow-Matching Transformer", test_flow_matching),
        ("DexVLG full model", test_dexvlg_forward),
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
