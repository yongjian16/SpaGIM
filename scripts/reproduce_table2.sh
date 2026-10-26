#!/bin/bash
# Refit the selected configuration of every dataset on all official splits with seeds 101 and 202 (Table 2).
# Usage: bash scripts/reproduce_table2.sh [DEVICE]   (run from the repository root; default device: cuda)
set -e
DEVICE=${1:-cuda}
for DATASET in Cora CiteSeer Chameleon-filtered Squirrel-filtered Actor Roman-empire Amazon-ratings Minesweeper; do
  python train.py --config "configs/selected/${DATASET}.json" --device "$DEVICE"
done
