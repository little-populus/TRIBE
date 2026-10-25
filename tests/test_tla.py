"""Unit checks for test-time logit adjustment (TLA)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from core.adapter.TRIBE import TRIBE


def test_warmup_and_uniform_prior():
    K = 4
    logits = torch.randn(16, K, generator=torch.Generator().manual_seed(0))
    cnt = torch.zeros(K)
    # during warmup: identity, counts collected
    adj, cnt = TRIBE.tla_adjust(logits, cnt, tau=1.0, alpha=1.0, warmup=100)
    assert torch.equal(adj, logits)
    assert cnt.sum() == 16
    print("warmup identity + count collection OK")


def test_head_class_suppressed():
    # prior strongly favours class 0; equal logits must flip towards the tail classes
    K = 4
    logits = torch.zeros(1, K)
    cnt = torch.tensor([1000.0, 10.0, 10.0, 10.0])
    adj, _ = TRIBE.tla_adjust(logits, cnt, tau=1.0, alpha=1.0, warmup=0)
    assert adj[0, 0] < adj[0, 1], "head class should be penalised"
    assert adj.argmax(dim=-1).item() != 0
    print(f"head suppression OK (adjusted logits {adj[0].tolist()})")


def test_uniform_prior_is_identity():
    K = 4
    logits = torch.randn(8, K, generator=torch.Generator().manual_seed(1))
    cnt = torch.full((K,), 250.0)
    adj, _ = TRIBE.tla_adjust(logits, cnt, tau=1.0, alpha=1.0, warmup=0)
    assert torch.allclose(adj, logits, atol=1e-5), "uniform prior must leave logits unchanged"
    print("uniform prior identity OK")


def test_strength_tau():
    K = 4
    logits = torch.zeros(1, K)
    cnt = torch.tensor([1000.0, 10.0, 10.0, 10.0])
    adj1, _ = TRIBE.tla_adjust(logits, cnt, tau=1.0, alpha=1.0, warmup=0)
    adj05, _ = TRIBE.tla_adjust(logits, cnt, tau=0.5, alpha=1.0, warmup=0)
    assert abs(adj1[0, 0].item()) > abs(adj05[0, 0].item())
    print("tau controls correction strength OK")


if __name__ == "__main__":
    test_warmup_and_uniform_prior()
    test_head_class_suppressed()
    test_uniform_prior_is_identity()
    test_strength_tau()
    print("all TLA checks passed")
