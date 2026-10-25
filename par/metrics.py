"""Attribute recognition metrics used for epoch selection on the held-out dataset of a LOO fold.

mA = mean over attributes of (TPR + TNR) / 2; Instance F1 = 2 P R / (P + R) with P and R averaged
over images; Challenge_Avg = harmonic mean of mA and Instance F1 (the Track 1 ranking metric).
"""
from __future__ import annotations

import numpy as np


def attribute_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    pred = probs > threshold
    gt = targets > 0.5
    per_attr = []
    for a in range(gt.shape[1]):
        pos, neg = gt[:, a], ~gt[:, a]
        rates = ([pred[pos, a].mean()] if pos.any() else []) + ([(~pred[neg, a]).mean()] if neg.any() else [])
        if rates:
            per_attr.append(np.mean(rates))
    mA = float(np.mean(per_attr))
    inter, n_pred, n_gt = (pred & gt).sum(1), pred.sum(1), gt.sum(1)
    both_empty = (n_pred == 0) & (n_gt == 0)
    prec = np.where(both_empty, 1.0, np.where(n_pred > 0, inter / np.maximum(n_pred, 1), 0.0)).mean()
    rec = np.where(both_empty, 1.0, np.where(n_gt > 0, inter / np.maximum(n_gt, 1), 0.0)).mean()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    avg = 2 * mA * f1 / (mA + f1) if mA + f1 > 0 else 0.0
    return {"Challenge_Avg": float(avg), "mA": mA, "Instance_F1": float(f1)}
