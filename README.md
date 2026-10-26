# SpaGIM

Code for the WWW 2027 submission "Node-Adaptive Implicit Graph Neural Networks via Bilevel Optimization".

SpaGIM computes the equilibrium Z\* of an implicit GNN and learns sparse, node-specific departures U\* from it. A
small allocator sets each node's residual target from label-free node statistics, and a group-lasso-regularized
lower-level problem fits these targets. The classifier reads Z\* − U\*. The model is trained end to end by implicit
differentiation through both the equilibrium and the lower-level solution.

## Contents

| Path | Contents |
|---|---|
| `spagim/model.py` | Implicit layers (MIGNN, mean aggregation, personalized PageRank) with implicit gradients, the budget allocator, the FISTA lower-level solver, and the implicit hypergradient (Algorithm 1) |
| `spagim/data.py` | Datasets, official splits, preprocessing, accuracy and ROC-AUC |
| `spagim/protocol.py` | The training and evaluation protocol shared by all experiments |
| `train.py` | Fits a configuration on every split and seed and summarizes the results |
| `tune.py` | Runs the 30-trial search on a dataset, then the final refits |
| `configs/selected/` | The selected configuration of each dataset (Table 6) |
| `configs/search/` | The search space of each dataset (Appendix C) |
| `scripts/reproduce_table2.sh` | Refits every selected configuration |

## Setup

The experiments used Python 3.10, PyTorch 2.1.0 (CUDA 11.8), PyTorch Geometric 2.7.0 and Optuna 4.8.0 on NVIDIA A30
GPUs.

    pip install -r requirements.txt

Datasets are downloaded into `data/` on first use:

- Roman-empire, Amazon-ratings and Minesweeper come through PyTorch Geometric.
- Actor and the filtered Chameleon and Squirrel graphs come from the repository of Platonov et al. (2023).
- Cora and CiteSeer use the Planetoid files with the public split.

## Main results (Table 2)

    python train.py --config configs/selected/Cora.json            # public split, seeds 101 and 202
    python train.py --config configs/selected/Roman-empire.json    # 10 official splits, seeds 101 and 202
    bash scripts/reproduce_table2.sh                               # all eight datasets

Each fit writes a JSON record to `results/<dataset>/`. It holds the validation and test scores at the best validation
epoch, the share of departing nodes, and the solver statistics. `summary.json` reports the mean ± population
standard deviation across splits of the per-split mean over the two seeds; on Cora and CiteSeer it is over the two
seeds.

| Roman-empire | Amazon-ratings | Minesweeper | Chameleon | Squirrel | Actor | Cora | CiteSeer |
|---|---|---|---|---|---|---|---|
| 91.35 ± 0.34 | 54.34 ± 0.41 | 96.60 ± 0.30 | 48.95 ± 2.64 | 44.47 ± 1.91 | 37.36 ± 0.75 | 82.50 ± 0.10 | 71.65 ± 0.05 |

The scores are accuracy, or ROC-AUC on Minesweeper. GPU runs are not bit-for-bit reproducible, because the sparse
kernels accumulate in a varying order, so single fits can differ slightly from the paper's.

On one A30, an epoch takes about 1 s on the three larger graphs (a 1,000-epoch Roman-empire fit takes about 20 minutes)
and much less on the small graphs.

## Protocol

These settings are in `spagim/protocol.py`:

- Hidden width 512.
- Adam, at most 1,000 epochs, patience 200, and a one-hour limit per fit.
- The validation score (accuracy, or ROC-AUC on Minesweeper) picks the checkpoint; test is read only at that epoch.
- Every model is tuned with the same budget: 30 trials of the tree-structured Parzen estimator, with 8 random start-up
  trials.
- A trial is scored by its mean validation score over seeds 11 and 29 on split 0.
- The selected configuration is refitted on every official split with seeds 101 and 202.

## Tuning

    python tune.py --dataset Cora

The search space is read from `configs/search/<dataset>.json`. Trials are logged to `results/tune/<dataset>/`, and
the best trial is refitted as above. In the spaces of Chameleon and Actor, the paper's search also had an alternative
destination variant that was never selected. It is not part of this release, so those spaces have 8 instead of 12
model configurations.

## Ablations and other experiments

Each experiment changes settings of a selected configuration with `--set KEY=VALUE`. Values are parsed as JSON,
e.g. `true`, `null` or `0.1`.

| Experiment | Settings |
|---|---|
| Departure-layer variants, each tuned (Table 3) | `python tune.py --dataset Chameleon-filtered --set departure=gate`; likewise `departure=thresholded_gate`, `lam=0`, `shared_budget=true`, and `lam=100` (no departures) |
| The same variants at the selected configuration (Table 10) | `python train.py --config configs/selected/Chameleon-filtered.json --set departure=gate --out results/gate/Chameleon-filtered` |
| Penalty strength (Figure 2, Table 10) | `--set lam=1` and the other values of the sweep; `lam=100` turns departures off |
| Backbone ablation (Table 4) | Roman-empire: `batch_norm=false`, `input_map=false`, `layers=2`; Amazon-ratings: `backbone=mignn stack_backbone=null`, `batch_norm=false`; Minesweeper: `layers=6`, `batch_norm=false` |
| Faster solver settings (Table 9) | `--set check_every=3 backward_tol=1e-4 separate_warm_starts=true` |

Use a different `--out` directory for each variant. Existing records in an output directory are reused.

## Notes

- Model options are documented in the `SpaGIM` constructor in `spagim/model.py`.
- Option names follow the paper: the injection is `propagated` (ÂXΩ), `own` (XΩ) or `combined` (XΩ + ÂXΩ′).
  `backbone="mean"` is the mean-aggregation layer of Section 3.3, and `backbone="ppr"` with `width="classes"` is the
  decoupled citation variant of Appendix B.
- Baselines were run under the same protocol (Appendix C). Their code is not part of this release.
