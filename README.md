# Gallery-Aware Attribute Refinement and Sinkhorn Normalization for Cross-Domain Person Retrieval

Code of team **gnurt2041** for the UPAR Challenge 2027 (WACV 2027 RWS workshop), track *Attribute-based Person Retrieval*.
Given a query as a binary vector of 40 attributes, the system ranks a gallery of pedestrian images.

| Test set (Codabench) | mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
|---|---|---|---|---|---|---|
| This repository | **0.6831** | 0.4057 | 0.4169 | 0.5808 | 0.6442 | 0.2829 |

The released code and weights reproduce this submission exactly (verified: identical distance matrices).

## Method

```
gallery images ──► ensemble of 3 attribute classifiers ──► p (G×40), f (G×D)
                    label transition ─► duplicate merging ─► kNN smoothing      (gallery only, no query)
queries (Q×40) ──► S = −log Sinkhorn(−NLL(q, p) / T) + μ·L1(q, p)  ──► distances (Q×G)
```

1. **Ensemble.** Weighted mean of the attribute probabilities of three classifiers trained with a weighted BCE loss:

   | | Backbone (timm) | Head | Training data | Weight |
   |---|---|---|---|---|
   | M1 | `eva02_large_patch14_clip_224.merged2b` | linear | Market1501 + PA100k + PETA | 0.51 |
   | M2 | `eva02_large_patch14_clip_224.merged2b` | region | PA100k + PETA | 0.34 |
   | M3 | `eva02_base_patch14_224.mim_in22k` | linear | PA100k + PETA | 0.15 |

   All models take a 224×112 image. The pooled feature `f` of M1 defines the gallery graph.
   The region head pools the token grid over four soft horizontal bands (head, upper body, lower body, full body)
   and each attribute reads the band it belongs to.
2. **Label transition** (λ = 0.15). Upper- and lower-body colour probabilities are mixed with
   `s · normalize(p) · A`, where `A[c', c] ≈ P(label c | predicted c')` is estimated on training data
   (`tools/make_label_transition.py`).
3. **Duplicate merging.** Each image averages `p` with its near-duplicates (cos(f) ≥ 0.9 among its 64 nearest neighbours).
4. **kNN smoothing.** `p ← Σ_j w_ij p_j` over the 8 nearest neighbours including the image itself,
   `w = softmax((sim − sim_max)/0.05)`, 3 passes. The self-weight is large in sparse neighbourhoods and small in dense ones.
5. **Scoring.** `NLL(q, p) = −Σ_k [q_k log p_k + (1−q_k) log(1−p_k)]`. The matrix `−NLL/T` (T = 2.5) is normalized
   with a partial Sinkhorn iteration: full normalization over queries, exponent β = 0.4 over the gallery, 3 rounds.
   An L1 term `Σ_k |q_k − p_k|` with weight μ = 0.15 is added.

All hyper-parameters are in `submission/assets/scoring_config.json`. They were selected on training data
(leave-one-dataset-out folds and simulated galleries of different density). Test images were never used for
training, pseudo-labelling or parameter selection.

## Repository layout

```
submission/                 content of the Codabench code submission
  run.py                    load_model() / rank_gallery(): ensemble, refinement, scoring
  model_def.py              classifier definition (shared with training)
  assets/                   model and scoring configs, label-transition prior, attribute prior; weights*.pt after download
train.py                    training of one classifier (LOO fold or final)
par/                        dataset, folds and attribute metrics used by train.py
configs/                    training configs of M1, M2, M3
tools/
  build_submission.py       builds the code-submission zip (with timm vendored) and checks that it loads
  evaluate.py               runs submission/run.py on a labelled split and reports mADM / mAP / Rank-k / mINP
  make_label_transition.py  fits the colour label-transition prior
  retrieval_metrics.py      challenge metrics
scripts/
  download_weights.sh       downloads the released weights into submission/assets/ and checks SHA-256
  train_all.sh              trains M1, M2, M3 and rebuilds the label-transition prior
Dockerfile
```

## Setup

**Data.** Place the UPAR challenge data under `./data` (or pass `--data_root` / `--set data.root=`):

```
data/annotations/task2/{train,val}/...   data/Market1501/...   data/PA100k/...   data/PETA/...
```

**Weights** (1.38 GB, fp16 state dicts):

```bash
bash scripts/download_weights.sh
```

| File | Size | Model |
|---|---|---|
| `m1_eva02l_clip.pt` → `submission/assets/weights.pt` | 607 MB | M1 |
| `m2_eva02l_clip_region.pt` → `submission/assets/weights2.pt` | 607 MB | M2 |
| `m3_eva02b_mim.pt` → `submission/assets/weights3.pt` | 171 MB | M3 |

**Environment.** Docker (recommended):

```bash
docker build -t upar-retrieval .
```

or pip with PyTorch 2.4.1 / torchvision 0.19.1 installed: `pip install -r requirements.txt`.
`timm` must be **1.0.9**; other versions change parameter names of the EVA02 models.

## Inference

Build the submission archive (the same archive that was evaluated on Codabench) and check that it loads offline:

```bash
docker run --rm --gpus all -v $PWD/out:/workspace/out upar-retrieval
# or: python tools/build_submission.py --out out/submission.zip --check
```

Evaluate on a labelled split with the challenge protocol (one `rank_gallery` call over a whole gallery):

```bash
docker run --rm --gpus all -v $PWD/data:/workspace/data upar-retrieval \
    python tools/evaluate.py --data_root data --dataset Market1501
```

Prediction of the 29,248 test images takes about 230 s on an NVIDIA A10 (fp16, batch 64), within the 300 s limit.
If the projected prediction time exceeds `budget_s` (240 s), the last ensemble member is dropped for the remaining images.
Set `BATCH_SIZE=32` for GPUs with less memory.

## Training

```bash
DATA=./data bash scripts/train_all.sh
```

| Model | Command | Released checkpoint |
|---|---|---|
| M1 | `--config configs/m1_eva02l_clip.yaml --fold loo_Market1501` to select the epoch, then `--fold final` | `final_epoch0.pt` |
| M2 | `--config configs/m2_eva02l_clip_region.yaml --fold loo_Market1501` | `loo_Market1501_best.pt` |
| M3 | `--config configs/m3_eva02b_mim.yaml --fold loo_Market1501` | `loo_Market1501_best.pt` |

Common recipe: AdamW (head LR 1e-4, backbone LR 1e-5, layer-wise decay 0.9, weight decay 5e-4), bfloat16 autocast,
EMA 0.9998, gradient clipping 5, random resized crop (scale 0.8–1, aspect ratio ±15 %), horizontal flip, AugMix,
random erasing 0.25, equal sampling of the source datasets, weighted BCE with `pos_weight = #neg / #pos` (≤ 50).
On a `loo_<Dataset>` fold the EMA model is validated on the held-out dataset after every epoch
(harmonic mean of mA and instance F1 at threshold 0.5), with early stopping after 3 epochs without improvement.
For M1 the held-out score peaks at epoch 0, so the final model is the first epoch on all three datasets
(≈ 94 min on one V100, batch 6). Pre-trained backbones are downloaded from the Hugging Face hub by timm;
on an offline machine set `model.pretrained=false` and `model.pretrained_path=<backbone state dict>`.

## Rules compliance

- Training data: Market1501, PA100k and PETA as provided by the challenge. Pre-trained models: EVA02 backbones (timm),
  which are generic image models and not trained for pedestrian attributes.
- The refinement steps use only the gallery passed to `rank_gallery` at inference time, with parameters fixed beforehand.
  Test images were not used for training, pseudo-labelling, annotation or hyper-parameter selection.

## Citation

If you use this code, please cite the workshop paper:

```
Gallery-Aware Attribute Refinement and Sinkhorn Normalization for Cross-Domain Person Retrieval.
WACV 2027 Workshops (RWS), UPAR Challenge.
```
