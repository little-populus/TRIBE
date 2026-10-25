"""Diagnostic: accuracy vs within-domain batch index.

Runs the default TRIBE pipeline on the GLI stream and reports how accuracy
evolves after each domain (corruption) switch, to quantify the cold-start
loss that domain-switch statistics pollution causes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch

from core.configs import cfg
from core.utils import *
from core.utils.utils import get_device, set_random_seed
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-acfg', default='configs/adapter/TRIBE.yaml')
    parser.add_argument('-dcfg', default='configs/dataset/cifar10.yaml')
    parser.add_argument('-pcfg', default='configs/protocol/gli_tta.yaml')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    cfg.merge_from_file(args.acfg)
    cfg.merge_from_file(args.dcfg)
    cfg.merge_from_file(args.pcfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.SEED)

    device = get_device()
    model = build_model(cfg)
    optimizer = build_optimizer(cfg)
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.to(device)

    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)

    correct_by_pos = {}   # within-domain batch index -> [correct_count, total]
    correct_by_pos_last_domain = {}
    prev_domain = None
    pos = 0
    max_pos = 0
    domain_names = []

    tta_model.eval()
    for data_package in loader:
        data = data_package["image"]
        label = data_package['label']
        domain = data_package['domain']
        if len(label) == 1:
            continue
        dom = int(domain[0].item())
        if dom != prev_domain:
            prev_domain = dom
            pos = 0
            domain_names.append(dom)
        data, label = data.to(device), label.to(device)
        output = tta_model(data)
        predict = torch.argmax(output, dim=1)
        correct = (predict == label).float()

        key = min(pos, 30)  # bucket everything beyond 30 into the last bin
        if key not in correct_by_pos:
            correct_by_pos[key] = [0.0, 0]
        correct_by_pos[key][0] += correct.sum().item()
        correct_by_pos[key][1] += len(label)
        max_pos = max(max_pos, key)
        pos += 1

    print("\n=== accuracy vs within-domain batch index (bucketed, 30+ merged) ===")
    print("pos\tacc\tnum")
    for key in sorted(correct_by_pos):
        c, n = correct_by_pos[key]
        print(f"{key}\t{100.0 * c / n:.2f}\t{n}")

    # aggregate buckets
    def bucket(lo, hi):
        c = sum(correct_by_pos[k][0] for k in correct_by_pos if lo <= k < hi)
        n = sum(correct_by_pos[k][1] for k in correct_by_pos if lo <= k < hi)
        return 100.0 * c / max(n, 1), n

    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 31)]:
        acc, n = bucket(lo, hi)
        print(f"batches[{lo}:{hi}) -> acc {acc:.2f} over {n} samples")


if __name__ == "__main__":
    main()
