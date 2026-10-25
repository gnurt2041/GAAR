"""Fit the colour label-transition prior (submission/assets/label_transition.json).

    python tools/make_label_transition.py --weights runs/m1_eva02l_clip/loo_Market1501_best.pt \
        --data_root ./data --out submission/assets/label_transition.json

The M1 model of the loo_Market1501 fold (not trained on Market1501) predicts the Market1501 images of the
labelled retrieval split. For the upper- and lower-body colour groups, on images with exactly one colour label:
    A[c', c] = sum_i p_i(c') y_i(c) + I,  rows normalized,  p_i normalized within the group,
i.e. A[c', c] ~ P(label c | predicted c'). Colours with fewer than 20 labelled images keep an identity row
and column, so the prior does not learn that a colour is missing from Market1501's label vocabulary.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submission"))
from model_def import build_from_config, get_backbone_normalization  # noqa: E402
from par.data import read_retrieval_split  # noqa: E402

GROUPS = {"U-Color": list(range(8, 20)), "L-Color": list(range(21, 33))}


def fit_A(P, Y, cols, smooth=1.0, min_n=20):
    m = Y[:, cols].sum(1) == 1
    pt = P[m][:, cols]
    pt = pt / (pt.sum(1, keepdims=True) + 1e-8)
    A = pt.T @ Y[m][:, cols].astype(np.float64) + smooth * np.eye(len(cols))
    A = A / A.sum(1, keepdims=True)
    absent = Y[m][:, cols].sum(0) < min_n
    A[absent] = np.eye(len(cols))[absent]
    A[:, absent] = 0
    A[np.ix_(absent, absent)] = np.eye(len(cols))[np.ix_(absent, absent)]
    return A / A.sum(1, keepdims=True)


@torch.no_grad()
def predict(weights, model_config, data_root, paths, batch=128):
    mc = json.load(open(model_config))
    model = build_from_config(mc)
    model.load_state_dict({k: v.float() for k, v in torch.load(weights, map_location="cpu").items()})
    model = model.eval().cuda().half()
    mean, std = get_backbone_normalization(mc["backbone"])
    tf = transforms.Compose([transforms.Resize(tuple(mc["image_size"])), transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    out = []
    for s in range(0, len(paths), batch):
        x = torch.stack([tf(Image.open(os.path.join(data_root, p)).convert("RGB")) for p in paths[s:s + batch]])
        out.append(torch.sigmoid(model(x.cuda().half()).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model_config", default=os.path.join(ROOT, "submission/assets/model_config.json"))
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--split", default="annotations/task2/val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names, gal, labels, *_ = read_retrieval_split(os.path.join(a.data_root, a.split), "Market1501")
    P = predict(a.weights, a.model_config, a.data_root, gal)
    groups = []
    for g, cols in GROUPS.items():
        A = fit_A(P, labels, cols)
        groups.append({"group": g, "names": [names[j] for j in cols], "A": np.round(A, 5).tolist()})
    with open(a.out, "w") as f:
        json.dump({"source": "Market1501 (M1 of the loo_Market1501 fold, vocab_only)", "groups": groups}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
