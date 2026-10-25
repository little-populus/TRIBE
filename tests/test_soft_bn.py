"""Unit checks for the soft-label Balanced BN statistics update.

Verifies that update_statistics_*_soft with one-hot weights reproduces the
original hard-label update_statistics_*_v5 exactly, and that a genuinely soft
weight matrix produces a smooth interpolation between hard assignments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from core.utils.balanced_bn_pyv import (
    update_statistics_1d_v5,
    update_statistics_2d_v5,
    update_statistics_1d_soft,
    update_statistics_2d_soft,
)


def make_stats(K, C, seed):
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(K, C, generator=g) * 0.5
    var = torch.rand(K, C, generator=g) + 0.5
    return mean, var


def run_case(dim, N=64, C=32, K=10, H=8, W=8, momentum=0.01, gamma=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    label = torch.randint(0, K, (N,), generator=g)
    # make one class absent to exercise the present-class mean
    label[label == K - 1] = 0
    weight = torch.nn.functional.one_hot(label, K).float()

    if dim == 2:
        data = torch.randn(N, C, H, W, generator=g)
        hard_fn, soft_fn = update_statistics_2d_v5, update_statistics_2d_soft
    else:
        data = torch.randn(N, C, 2 * W, generator=g)
        hard_fn, soft_fn = update_statistics_1d_v5, update_statistics_1d_soft

    lm1, lv1 = make_stats(K, C, seed + 1)
    lm2, lv2 = lm1.clone(), lv1.clone()
    gm1 = lm1.mean(0).clone(); gv1 = (lv1 + lm1.var(0)).clone()
    gm2 = gm1.clone(); gv2 = gv1.clone()

    hard_fn(lm1, lv1, gm1, gv1, momentum, data, label, gamma, True)
    soft_fn(lm2, lv2, gm2, gv2, momentum, data, weight, gamma, True)

    assert torch.allclose(lm1, lm2, atol=1e-5), f"[{dim}d] local_mean mismatch: {(lm1-lm2).abs().max():.3e}"
    assert torch.allclose(lv1, lv2, atol=1e-4), f"[{dim}d] local_var mismatch: {(lv1-lv2).abs().max():.3e}"
    assert torch.allclose(gm1, gm2, atol=1e-5), f"[{dim}d] global_mean mismatch"
    assert torch.allclose(gv1, gv2, atol=1e-4), f"[{dim}d] global_var mismatch"
    print(f"[{dim}d] one-hot soft == hard  OK  (max |dmean|={(lm1-lm2).abs().max():.2e}, max |dvar|={(lv1-lv2).abs().max():.2e})")


def run_soft_interpolation(dim, N=64, C=16, K=4, H=8, W=8, momentum=0.01, gamma=0.0, seed=3):
    """Soft weights between two one-hot assignments must interpolate the update."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(N, C, H, W, generator=g) if dim == 2 else torch.randn(N, C, 2 * W, generator=g)
    lab_a = torch.randint(0, K, (N,), generator=g)
    lab_b = torch.randint(0, K, (N,), generator=g)
    oh_a = torch.nn.functional.one_hot(lab_a, K).float()
    oh_b = torch.nn.functional.one_hot(lab_b, K).float()

    alpha = 0.3
    soft_fn = update_statistics_2d_soft if dim == 2 else update_statistics_1d_soft

    lm0, lv0 = make_stats(K, C, seed + 1)
    outs = []
    for w in (oh_a, alpha * oh_a + (1 - alpha) * oh_b, oh_b):
        lm, lv = lm0.clone(), lv0.clone()
        gm = lm.mean(0).clone(); gv = (lv + lm.var(0)).clone()
        soft_fn(lm, lv, gm, gv, momentum, data, w, gamma, True)
        outs.append(lm.clone())
    dist_ab = (outs[0] - outs[2]).abs().max()
    dist_mix_a = (outs[1] - outs[0]).abs().max()
    assert dist_mix_a < dist_ab, f"[{dim}d] mixture should sit between the two hard outcomes"
    print(f"[{dim}d] soft interpolation ordering OK (|mix-a|={dist_mix_a:.3e} < |a-b|={dist_ab:.3e})")


if __name__ == "__main__":
    for gamma in (0.0, 0.1):
        run_case(2, gamma=gamma, seed=1)
        run_case(1, gamma=gamma, seed=2)
    run_soft_interpolation(2)
    run_soft_interpolation(1)
    print("all checks passed")
