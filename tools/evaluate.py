"""Run submission/run.py on the gallery of one dataset of a labelled retrieval split and report mADM.

    python tools/evaluate.py --data_root ./data --dataset Market1501

Same protocol as the test phase: one rank_gallery() call with the whole gallery; queries are the
attribute vectors that have at least one match in the gallery. Note that M1 is trained on all three
datasets, so this run checks the installation, not generalization (use LOO-fold weights for that,
e.g. ASSETS_DIR pointing to assets with loo_Market1501 weights).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from par.data import read_retrieval_split  # noqa: E402
from retrieval_metrics import evaluate_retrieval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--split", default="annotations/task2/val")
    ap.add_argument("--dataset", default="Market1501", choices=["Market1501", "PA100k", "PETA"])
    ap.add_argument("--max_images", type=int, default=0, help="subsample the gallery (quick test)")
    ap.add_argument("--save", default=None, help="save the distance matrix (.npy)")
    a = ap.parse_args()
    names, gal, labels, g_ids, queries, q_ids = read_retrieval_split(os.path.join(a.data_root, a.split), a.dataset)
    if a.max_images:
        sel = np.arange(0, len(gal), max(1, len(gal) // a.max_images))[:a.max_images]
        gal, labels, g_ids = [gal[i] for i in sel], labels[sel], g_ids[sel]
        keep = np.isin(q_ids, g_ids)
        queries, q_ids = queries[keep], q_ids[keep]
    print(f"{a.dataset}: gallery {len(gal)}, queries {len(queries)}", flush=True)

    sys.path.insert(0, os.path.join(ROOT, "submission"))
    import run
    t0 = time.time()
    run.load_model()
    sample = {"queries": queries.astype(np.float32).tolist(), "attribute_names": names,
              "gallery": [{"image_path": os.path.join(a.data_root, p)} for p in gal]}
    dist = run.rank_gallery(sample)["distances"]
    elapsed = time.time() - t0
    if a.save:
        np.save(a.save, dist)
    res = evaluate_retrieval(dist, labels, g_ids, queries, q_ids)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
          + f"  ({elapsed:.0f} s incl. model loading)")


if __name__ == "__main__":
    main()
