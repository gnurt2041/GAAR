"""Train one attribute classifier.

    python train.py --config configs/m1_eva02l_clip.yaml --fold loo_Market1501   # epoch selection
    python train.py --config configs/m1_eva02l_clip.yaml --fold final            # all three datasets

`loo_<Dataset>` trains on the two other datasets, validates on <Dataset> after every epoch
(EMA weights, Challenge_Avg at threshold 0.5), keeps the best epoch and stops after
`early_stop_patience` epochs without improvement. `final` trains on all datasets and saves
every epoch. Any config value can be overridden with `--set key.subkey=value`.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, ROOT)
from model_def import ATTRIBUTE_NAMES, PARModel, get_backbone_normalization  # noqa: E402
from par.data import UPARDataset, balanced_sampler, get_fold  # noqa: E402
from par.metrics import attribute_metrics  # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")


def build_transforms(cfg):
    h, w = cfg["data"]["image_size"]
    mean, std = get_backbone_normalization(cfg["model"]["backbone"])
    aug = cfg["train"]["augment"]
    ops = [transforms.Resize((h, w))]
    if aug.get("random_crop"):
        j = aug.get("rrc_ratio_jitter")
        ratio = (w / h * (1 - j), w / h * (1 + j)) if j else (0.75 * w / h, (4 / 3) * w / h)
        ops.append(transforms.RandomResizedCrop((h, w), scale=(float(aug.get("rrc_scale_min", 0.8)), 1.0), ratio=ratio))
    if aug.get("horizontal_flip"):
        ops.append(transforms.RandomHorizontalFlip())
    if aug.get("augmix"):
        ops.append(transforms.AugMix())
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if aug.get("random_erasing", 0) > 0:
        ops.append(transforms.RandomErasing(p=aug["random_erasing"]))
    val = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops), val


def pos_weight_of(labels: np.ndarray, clamp: float = 50.0) -> torch.Tensor:
    """Weighted BCE: positives of attribute a are weighted by #neg_a / #pos_a (clamped)."""
    pos, neg = labels.sum(0), (1 - labels).sum(0)
    pw = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(pw, dtype=torch.float32).clamp(max=clamp)


def param_groups(model: PARModel, lr_head, lr_backbone, weight_decay, llrd):
    """Layer-wise LR decay: block i of n gets lr_backbone * llrd^(n-1-i); the embeddings get llrd^n."""
    head = list(model.head.parameters())
    if model.region_mode:
        head += [model.region_weight, model.region_bias]
    head_group = {"params": head, "lr": lr_head, "weight_decay": weight_decay}
    bb = model.backbone
    if not llrd or llrd >= 1.0 or not hasattr(bb, "blocks"):
        return [{"params": list(bb.parameters()), "lr": lr_backbone, "weight_decay": weight_decay}, head_group]
    n = len(bb.blocks)
    blocks, embed, final = [[] for _ in range(n)], [], []
    for name, p in bb.named_parameters():
        if name.startswith("blocks."):
            blocks[int(name.split(".")[1])].append(p)
        elif name.startswith(("norm.", "fc_norm.", "head.")) or name in ("norm", "fc_norm"):
            final.append(p)
        else:
            embed.append(p)
    groups = [{"params": embed, "lr": lr_backbone * llrd ** n, "weight_decay": weight_decay}]
    groups += [{"params": ps, "lr": lr_backbone * llrd ** (n - 1 - i), "weight_decay": weight_decay}
               for i, ps in enumerate(blocks)]
    groups += [{"params": final, "lr": lr_backbone, "weight_decay": weight_decay}, head_group]
    return groups


def mix_batch(cfg, images, target, valid):
    """Optional batch mixing. regionmix: upper part of image i + lower part of image j (cut at 45-55 % of
    the height), LowerBody-* labels from j, bag/backpack ignored. mixup: convex combination of images and labels."""
    rm, mx = cfg["train"].get("regionmix") or {}, cfg["train"].get("mixup") or {}
    if float(rm.get("prob", 0)) > 0 and np.random.rand() < float(rm["prob"]):
        r = int(round(images.shape[2] * float(np.random.uniform(0.45, 0.55))))
        perm = torch.randperm(images.shape[0], device=images.device)
        images = torch.cat([images[:, :, :r], images[perm][:, :, r:]], dim=2)
        lower = torch.tensor([n.startswith("LowerBody-") for n in ATTRIBUTE_NAMES], device=images.device)
        bags = torch.tensor([n in ("Accessory-Backpack", "Accessory-Bag") for n in ATTRIBUTE_NAMES], device=images.device)
        target = torch.where(lower, target[perm], target)
        valid = torch.where(lower, valid[perm], valid) * (~bags).to(valid.dtype)
    elif float(mx.get("alpha", 0)) > 0 and np.random.rand() < float(mx.get("prob", 1.0)):
        lam = float(np.random.beta(mx["alpha"], mx["alpha"]))
        perm = torch.randperm(images.shape[0], device=images.device)
        images = lam * images + (1 - lam) * images[perm]
        target = lam * target + (1 - lam) * target[perm]
        valid = valid * valid[perm]
    return images, target, valid


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    probs, targets = [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(batch["image"].to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(batch["target"].numpy())
    return attribute_metrics(np.concatenate(probs), np.concatenate(targets))


def train(cfg, fold, device, limit_batches=None):
    train_tf, val_tf = build_transforms(cfg)
    csvs = [os.path.join(cfg["data"]["root"], p) for p in cfg["data"]["gt_csvs"]]
    train_ds = UPARDataset(csvs, cfg["data"]["root"], fold.train_datasets, train_tf)
    tc = cfg["train"]
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], num_workers=cfg["data"]["num_workers"], drop_last=True,
                        sampler=balanced_sampler(train_ds) if tc.get("balanced_dataset_sampling") else None,
                        shuffle=not tc.get("balanced_dataset_sampling"))
    val_loader = None
    if fold.val_dataset:
        val_ds = UPARDataset(csvs, cfg["data"]["root"], [fold.val_dataset], val_tf)
        val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], num_workers=cfg["data"]["num_workers"])
    print(f"[{fold.name}] train {fold.train_datasets}: {len(train_ds)} images; val {fold.val_dataset}", flush=True)

    mcfg = cfg["model"]
    model = PARModel(mcfg["backbone"], len(ATTRIBUTE_NAMES), pretrained=mcfg.get("pretrained", True),
                     drop_rate=mcfg.get("drop_rate", 0.2), img_size=mcfg.get("img_size"),
                     region_head=mcfg.get("region_head", False), grad_checkpointing=mcfg.get("grad_checkpointing", False),
                     pretrained_path=mcfg.get("pretrained_path")).to(device)
    pos_weight = pos_weight_of(train_ds.labels, cfg["loss"].get("pos_weight_clamp", 50.0)).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, tc["lr_head"], tc["lr_backbone"], tc["weight_decay"],
                                               tc.get("llrd_decay")))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)
    ema = copy.deepcopy(model).requires_grad_(False) if tc["ema"]["enable"] else None

    best, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(tc["epochs"]):
        model.train()
        total, steps = 0.0, 0
        for bi, batch in enumerate(tqdm(loader, desc=f"[{fold.name}] epoch {epoch}")):
            if limit_batches and bi >= limit_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            images, target, valid = mix_batch(cfg, images, target, torch.ones_like(target))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda" and tc.get("amp", True)):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
                loss = (loss * valid).sum() / valid.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    d = tc["ema"]["decay"]
                    for pe, p in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(p.detach(), alpha=1 - d)
                    for be, b in zip(ema.buffers(), model.buffers()):
                        be.copy_(b)
            total, steps = total + loss.item(), steps + 1
        eval_model = ema if ema is not None else model
        state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        log = {"epoch": epoch, "train_loss": total / max(steps, 1)}
        if tc.get("save_all_epochs") or val_loader is None:
            torch.save(state, os.path.join(cfg["output_dir"], f"{fold.name}_epoch{epoch}.pt"))
        if val_loader is not None:
            m = validate(eval_model, val_loader, device)
            scheduler.step(1 - m["Challenge_Avg"])
            log.update(m)
            if m["Challenge_Avg"] > best:
                best, best_state, stale = m["Challenge_Avg"], state, 0
            else:
                stale += 1
        print(json.dumps(log), flush=True)
        history.append(log)
        if val_loader is not None and stale >= tc.get("early_stop_patience", 3):
            print(f"[{fold.name}] early stop at epoch {epoch} (best Challenge_Avg {best:.4f})", flush=True)
            break
    if best_state is not None:
        torch.save(best_state, os.path.join(cfg["output_dir"], f"{fold.name}_best.pt"))
    with open(os.path.join(cfg["output_dir"], f"{fold.name}_history.json"), "w") as f:
        json.dump(history, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="final", help="final | loo_Market1501 | loo_PA100k | loo_PETA")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit_batches", type=int, default=None, help="debug: batches per epoch")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for ov in args.overrides:
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    train(cfg, get_fold(args.fold), "cuda" if torch.cuda.is_available() else "cpu", args.limit_batches)


if __name__ == "__main__":
    main()
