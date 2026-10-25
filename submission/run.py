"""UPAR Challenge 2027, attribute-based person retrieval: code submission entry point.

Interface of the challenge's ingestion program:
    load_model()                 called once
    rank_gallery(sample) -> {"distances": (n_queries, n_gallery) array}, smaller = better match

Pipeline (every gallery image passes through each model once):
  1. Ensemble: weighted mean of the attribute probabilities of up to three classifiers
     (assets/weights.pt, weights2.pt, weights3.pt). The first model also gives the
     pooled feature f used to build the gallery graph.
  2. Label transition: the upper- and lower-body colour probabilities are mixed with
     a transition prior learned on training data (assets/label_transition.json).
  3. Duplicate merging: average p over neighbours with cos(f) >= theta.
  4. kNN smoothing: p <- sum_j w_ij p_j over the k nearest neighbours,
     w = softmax((sim - sim_max) / tau), repeated `iters` times.
  5. Scoring: S = -log Sinkhorn(-NLL / T) + mu * L1, where the Sinkhorn step normalizes
     fully over queries and partially (exponent beta) over the gallery.

All settings are read from assets/scoring_config.json.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# The ingestion program passes the whole gallery in one call, so images are processed in batches.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_IO_THREADS = 8
LOG_EVERY_BATCHES = 20

_state = {"model": None, "members": [], "device": "cpu", "transform": None, "prior": {}, "cfg": {},
          "t0": None, "pool": None, "n_images": 0, "n_bad": 0, "n_batch_err": 0, "n_batches": 0}


def _log(msg):
    print(f"[run] {msg}", file=sys.stderr, flush=True)


def _asset(name):
    if name == "scoring_config.json" and os.environ.get("SCORING_CONFIG"):
        return os.environ["SCORING_CONFIG"]
    return os.path.join(os.environ.get("ASSETS_DIR", os.path.join(_THIS_DIR, "assets")), name)


def _load_json(name):
    with open(_asset(name), encoding="utf-8") as f:
        return json.load(f)


def _load_classifier(config_name, weights_name, device):
    from model_def import build_from_config
    mc = _load_json(config_name)
    model = build_from_config(mc)
    sd = torch.load(_asset(weights_name), map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return model.eval().to(device), mc


def load_model() -> None:
    _state["t0"] = time.time()
    _state["pool"] = ThreadPoolExecutor(max_workers=NUM_IO_THREADS)
    try:
        prior = _load_json("attribute_prior.json")
        _state["prior"] = dict(zip(prior["attribute_names"], prior["positive_rate"]))
        from model_def import get_backbone_normalization

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, mc = _load_classifier("model_config.json", "weights.pt", device)
        h, w = mc["image_size"]
        mean, std = get_backbone_normalization(mc["backbone"])
        _state["transform"] = transforms.Compose([
            transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        cfg = _load_json("scoring_config.json")
        _state["cfg"] = cfg
        _state["half"] = bool(cfg.get("half", True)) and device == "cuda"
        lt = cfg.get("label_transition")
        if lt:
            _state["label_transition"] = dict(_load_json(lt.get("file", "label_transition.json")),
                                              lam=float(lt.get("lam", 0.15)))

        ens = cfg.get("ensemble")
        if ens:
            ws = [float(v) for v in ens["weights"]]
            _state["w0"], _state["budget_s"] = ws[0], float(ens.get("budget_s", 240))
            for k in range(2, len(ws) + 1):
                try:
                    mk, mck = _load_classifier(f"model{k}_config.json", f"weights{k}.pt", device)
                    # all members share the image tensor of the first model
                    assert list(mck["image_size"]) == list(mc["image_size"])
                    assert get_backbone_normalization(mck["backbone"]) == (mean, std)
                    _state["members"].append((mk, ws[k - 1], mck["backbone"]))
                except Exception:
                    _log(f"ensemble: model {k} could not be loaded, skipped:\n{traceback.format_exc()}")
            _log(f"ensemble: weights {ws}, prediction budget {_state['budget_s']:.0f} s")

        if _state["half"]:
            model.half()
            for mk, _, _ in _state["members"]:
                mk.half()
        _state.update(model=model, device=device)
        _log(f"ready on {device}: backbone={mc['backbone']} size={mc['image_size']} "
             f"load={time.time() - _state['t0']:.1f}s")
    except Exception:
        _state["model"] = None
        _log(f"load_model failed, every image gets the attribute prior:\n{traceback.format_exc()}")


def _read_one(g):
    try:
        with Image.open(g["image_path"]) as im:
            return _state["transform"](im.convert("RGB"))
    except Exception as e:
        _state["n_bad"] += 1
        _log(f"unreadable image: {g.get('image_path')} ({e!r})")
        return None


def _to_device(x):
    x = x.to(_state["device"])
    return x.half() if _state["half"] else x


@torch.no_grad()
def _forward_main(x):
    """Logits (B, 40) and pooled feature (B, D) of the first model."""
    model = _state["model"]
    try:
        xd = _to_device(x)
        if model.region_mode:
            lg, f = model(xd, return_features=True)
        else:
            f = model.pooled_features(xd)
            lg = model.head(model.dropout(f))
        return lg.float().cpu(), f.float().cpu()
    except Exception:
        if x.shape[0] <= 1:
            raise
        _state["n_batch_err"] += 1
        _log(f"batch {tuple(x.shape)} failed, splitting in two:\n{traceback.format_exc()}")
        h = x.shape[0] // 2
        a, fa = _forward_main(x[:h])
        b, fb = _forward_main(x[h:])
        return torch.cat([a, b], 0), torch.cat([fa, fb], 0)


@torch.no_grad()
def _forward_member(model, x):
    return model(_to_device(x)).float().cpu()


def _predict(gallery, names):
    """Attribute probabilities (G, 40) and features (G, D) of the gallery."""
    out = np.tile(np.array([_state["prior"].get(n, 0.5) for n in names], dtype=np.float32), (len(gallery), 1))
    feats, ok_mask = None, np.zeros(len(gallery), dtype=bool)
    if _state["model"] is None or not gallery:
        return out, feats, ok_mask
    pool, t_start, members = _state["pool"], time.time(), _state["members"]
    nxt = [pool.submit(_read_one, g) for g in gallery[:BATCH_SIZE]]
    for s0 in range(0, len(gallery), BATCH_SIZE):
        chunk = gallery[s0:s0 + BATCH_SIZE]
        cur = nxt                                   # the next batch is decoded while this one runs on the GPU
        nxt = [pool.submit(_read_one, g) for g in gallery[s0 + BATCH_SIZE:s0 + 2 * BATCH_SIZE]]
        try:
            tensors = [f.result() for f in cur]
            ok = [i for i, t in enumerate(tensors) if t is not None]
            if ok:
                x = torch.stack([tensors[i] for i in ok])
                lg, ft = _forward_main(x)
                probs = torch.sigmoid(lg).clamp(0.0, 1.0).numpy()
                if members:
                    acc, wsum = _state["w0"] * probs, _state["w0"]
                    for mk, w, name in list(members):
                        try:
                            pk = torch.sigmoid(_forward_member(mk, x)).clamp(0.0, 1.0).numpy()
                            if pk.shape == probs.shape and np.all(np.isfinite(pk)):
                                acc, wsum = acc + w * pk, wsum + w
                        except Exception:
                            members.remove((mk, w, name))
                            _log(f"ensemble: {name} failed and is disabled:\n{traceback.format_exc()}")
                    probs = acc / wsum
                    # time guard: drop the last member if the projected prediction time exceeds the budget
                    done = s0 + len(chunk)
                    if members and done >= 8 * BATCH_SIZE and done % (8 * BATCH_SIZE) == 0:
                        projected = (time.time() - t_start) / done * len(gallery)
                        if projected > _state["budget_s"]:
                            _, _, name = members.pop()
                            _log(f"ensemble: projected {projected:.0f} s > budget, {name} disabled "
                                 f"from image {done}/{len(gallery)}")
                if feats is None:
                    feats = np.zeros((len(gallery), ft.shape[1]), dtype=np.float16)
                for j, i in enumerate(ok):
                    row = probs[j]
                    if row.shape[0] == len(names) and np.all(np.isfinite(row)):
                        out[s0 + i] = row
                        if np.all(np.isfinite(ft[j].numpy())):
                            feats[s0 + i] = ft[j].numpy().astype(np.float16)
                            ok_mask[s0 + i] = True
        except Exception:
            _state["n_batch_err"] += 1
            _log(f"chunk {s0} ({len(chunk)} images) failed:\n{traceback.format_exc()}")
        _state["n_batches"] += 1
        _state["n_images"] += len(chunk)
        if _state["n_batches"] % LOG_EVERY_BATCHES == 0:
            dt = time.time() - _state["t0"]
            _log(f"progress {_state['n_images']}/{len(gallery)} images, {dt:.0f} s "
                 f"({_state['n_images'] / max(dt, 1e-9):.1f} img/s), bad={_state['n_bad']} "
                 f"batch_err={_state['n_batch_err']}")
    _log(f"prediction done in {time.time() - t_start:.0f} s with {1 + len(members)} model(s) at the end")
    return out, feats, ok_mask


# ---------------------------------------------------------------- gallery-side refinement

def apply_label_transition(probs, names, lt):
    """Per colour group G: p'_G = s_G * normalize(p_G) @ A_G with s_G = sum(p_G),
    A_G[c', c] = P(label c | predicted c'); then p <- (1 - lam) p + lam p'."""
    out, lam = probs.copy(), float(lt["lam"])
    for g in lt["groups"]:
        cols = [names.index(n) for n in g["names"]]
        A = np.asarray(g["A"], dtype=np.float32)
        pg = probs[:, cols]
        sm = pg.sum(1, keepdims=True)
        pa = np.clip(sm * ((pg / (sm + 1e-8)) @ A), 0.0, 1.0)
        out[:, cols] = (1 - lam) * pg + lam * pa
    return out


def _normalized(feats, idxs, dev):
    F = torch.from_numpy(feats[idxs]).to(dev).float()
    F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    return F.half() if dev == "cuda" else F


def merge_duplicates(probs, feats, ok, theta=0.9, kmax=64, chunk=2048):
    """p'(g) = mean of p over {g' among the kmax nearest : cos(g, g') >= theta} (g itself included)."""
    idxs = np.where(ok)[0]
    m = len(idxs)
    if m < 3:
        return probs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Fh = _normalized(feats, idxs, dev)
    P = torch.from_numpy(probs[idxs]).to(dev).float()
    out_p, cnt = torch.empty_like(P), []
    for s0 in range(0, m, chunk):
        sv, si = (Fh[s0:s0 + chunk] @ Fh.T).float().topk(int(min(kmax, m)), dim=1)
        w = (sv >= float(theta)).float()
        w[:, 0] = 1.0
        out_p[s0:s0 + chunk] = (w.unsqueeze(-1) * P[si]).sum(1) / w.sum(1, keepdim=True)
        cnt.append(w.sum(1))
    cnt = torch.cat(cnt)
    _log(f"duplicate merge theta={theta}: mean group size {cnt.mean().item():.1f}, median {cnt.median().item():.0f}")
    out = probs.copy()
    out[idxs] = out_p.clamp(0, 1).cpu().numpy()
    return out


def knn_smooth(probs, feats, ok, k=8, tau=0.05, iters=3, chunk=2048):
    """p'(g) = sum_{g' in kNN(g)} w(g, g') p(g'), w = softmax((sim - sim_max) / tau), applied `iters` times.
    The neighbour list contains g itself, so its weight adapts to the local density."""
    idxs = np.where(ok)[0]
    m = len(idxs)
    if m < 3:
        return probs
    k = int(min(k, m))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Fh = _normalized(feats, idxs, dev)
    I = torch.zeros(m, k, dtype=torch.long, device=dev)
    W = torch.zeros(m, k, device=dev)
    for s0 in range(0, m, chunk):
        sv, si = (Fh[s0:s0 + chunk] @ Fh.T).float().topk(k, dim=1)
        I[s0:s0 + chunk] = si
        W[s0:s0 + chunk] = torch.softmax((sv - sv[:, :1]) / float(tau), dim=1)
    P = torch.from_numpy(probs[idxs]).to(dev).float()
    for _ in range(int(iters)):
        P = (W.unsqueeze(-1) * P[I]).sum(1)
    out = probs.copy()
    out[idxs] = P.clamp(0, 1).cpu().numpy()
    return out


# ---------------------------------------------------------------- scoring

def score_l1(probs, queries):
    """sum_k |q_k - p_k| = sum_k p_k + q . (1 - 2p) for binary q."""
    p = probs.astype(np.float64)
    return (p.sum(1)[None, :] + queries.astype(np.float64) @ (1.0 - 2.0 * p).T).astype(np.float32)


@torch.no_grad()
def score_sinkhorn(probs, queries, temperature=2.5, beta=0.4, iters=3, eps=1e-6):
    """L = -NLL / T, normalized over queries (log-softmax per gallery column), then `iters` rounds of
    L <- L - beta * logsumexp_gallery(L);  L <- log_softmax_queries(L).  Returns -L."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.from_numpy(np.clip(probs.astype(np.float32), eps, 1.0 - eps)).to(dev)
    q = torch.from_numpy(queries.astype(np.float32)).to(dev)
    nll = (-torch.log1p(-p)).sum(1)[None, :] + q @ (torch.log1p(-p) - torch.log(p)).T     # (Q, G)
    L = -nll / float(temperature)
    L = L - torch.logsumexp(L, 0, keepdim=True)
    for _ in range(int(iters)):
        L = L - float(beta) * torch.logsumexp(L, 1, keepdim=True)
        L = L - torch.logsumexp(L, 0, keepdim=True)
    return (-L).cpu().numpy()


def rank_gallery(sample: dict) -> dict:
    if _state["t0"] is None:
        load_model()
    names = list(sample.get("attribute_names") or _state["prior"].keys())
    gallery = list(sample["gallery"])
    queries = np.asarray(sample["queries"], dtype=np.float32)
    cfg = _state["cfg"]
    t0 = time.time()
    probs, feats, ok = _predict(gallery, names)

    if _state.get("label_transition"):
        try:
            probs = apply_label_transition(probs, names, _state["label_transition"])
        except Exception:
            _log(f"label transition failed, skipped:\n{traceback.format_exc()}")
    rr = cfg.get("rerank")
    if rr and feats is not None and ok.sum() >= 3:
        try:
            if rr.get("dup"):
                probs = merge_duplicates(probs, feats, ok, theta=rr["dup"].get("theta", 0.9),
                                         kmax=rr["dup"].get("kmax", 64))
            probs = knn_smooth(probs, feats, ok, k=rr.get("k", 8), tau=rr.get("tau", 0.05), iters=rr.get("iters", 3))
        except Exception:
            _log(f"gallery refinement failed, skipped:\n{traceback.format_exc()}")

    try:
        sk = cfg.get("sinkhorn") or {}
        distances = score_sinkhorn(probs, queries, float(cfg.get("temperature", 2.5)),
                                   beta=sk.get("beta", 0.4), iters=sk.get("iters", 3))
        mu = float(cfg.get("l1_mix", 0.0) or 0.0)
        if mu > 0:
            distances = distances + mu * score_l1(probs, queries)
        if not np.isfinite(distances).all():
            raise RuntimeError("non-finite distances")
    except Exception:
        _log(f"Sinkhorn scoring failed, falling back to L1:\n{traceback.format_exc()}")
        distances = score_l1(probs, queries)
    distances = np.nan_to_num(distances, nan=1e9, posinf=1e9, neginf=-1e9).astype(np.float32)
    _log(f"distances {distances.shape} (queries x gallery) in {time.time() - t0:.0f} s")
    return {"distances": distances}


@atexit.register
def _summary():
    if _state["t0"] is not None:
        _log(f"summary: images={_state['n_images']} elapsed={time.time() - _state['t0']:.0f}s "
             f"bad_images={_state['n_bad']} batch_errors={_state['n_batch_err']}")
