"""Retrieval metrics of the UPAR Challenge (mADM, mAP, mINP, Rank-k).

For a query with R exact matches at ranks r_1 < ... < r_R:
    AP  = (1/R) sum_k k / r_k
    INP = R / r_R
    dom(g)  = fraction of the 40 attributes of g that agree with the query
    ndom(g) = max(0, (dom(g) - mean dom) / (max dom - mean dom))   (over the whole gallery)
    ADM = (1/R) sum_k (ndom(g_1) + ... + ndom(g_{r_k})) / r_k
Metrics are averaged over the queries that have at least one match in the gallery.
"""
from __future__ import annotations

import numpy as np


def evaluate_retrieval(distances: np.ndarray, gallery_labels: np.ndarray, gallery_ids: np.ndarray,
                       query_vecs: np.ndarray, query_ids: np.ndarray, rank_ks=(1, 5, 10), eps=1e-8) -> dict:
    """distances: (Q, G), smaller = better. gallery_ids / query_ids: semantic ids; a gallery image is a
    match of a query when the ids are equal (equivalently, all 40 attributes agree)."""
    gallery_labels = np.asarray(gallery_labels)
    aps, adms, inps, hits = [], [], [], {k: [] for k in rank_ks}
    for qi in range(query_vecs.shape[0]):
        rel = gallery_ids == query_ids[qi]
        R = int(rel.sum())
        if R == 0:
            continue
        order = np.argsort(distances[qi], kind="stable")
        ranks = np.where(rel[order])[0] + 1
        for k in rank_ks:
            hits[k].append(float(ranks[0] <= k))
        aps.append(float(np.sum(np.arange(1, R + 1) / ranks) / R))
        inps.append(float(R / ranks[-1]))
        dom = (gallery_labels == query_vecs[qi][None, :]).sum(1) / gallery_labels.shape[1]
        denom = dom.max() - dom.mean()
        ndom = np.zeros_like(dom) if denom < eps else np.clip((dom - dom.mean()) / denom, 0.0, None)
        cum = np.cumsum(ndom[order])
        adms.append(float(np.sum(cum[ranks - 1] / ranks) / R))
    out = {"mADM": float(np.mean(adms)), "mAP": float(np.mean(aps)), "mINP": float(np.mean(inps))}
    out.update({f"Rank-{k}": float(np.mean(v)) for k, v in hits.items()})
    out["n_queries"] = len(aps)
    return out
