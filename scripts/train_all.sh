#!/usr/bin/env bash
# Train the three classifiers of the released system (one GPU each, run sequentially here).
# The epoch of M1 is chosen on the loo_Market1501 fold; its final model is trained on all datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-./data}

python train.py --config configs/m1_eva02l_clip.yaml --fold loo_Market1501 --set data.root=$DATA   # epoch selection
python train.py --config configs/m1_eva02l_clip.yaml --fold final --set data.root=$DATA --set train.epochs=1
python train.py --config configs/m2_eva02l_clip_region.yaml --fold loo_Market1501 --set data.root=$DATA
python train.py --config configs/m3_eva02b_mim.yaml --fold loo_Market1501 --set data.root=$DATA

cp runs/m1_eva02l_clip/final_epoch0.pt            submission/assets/weights.pt
cp runs/m2_eva02l_clip_region/loo_Market1501_best.pt submission/assets/weights2.pt
cp runs/m3_eva02b_mim/loo_Market1501_best.pt         submission/assets/weights3.pt

# colour label-transition prior (needs the M1 model of the loo_Market1501 fold)
python tools/make_label_transition.py --weights runs/m1_eva02l_clip/loo_Market1501_best.pt \
  --data_root $DATA --out submission/assets/label_transition.json
