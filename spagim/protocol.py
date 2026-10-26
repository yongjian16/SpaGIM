"""The evaluation protocol shared by training and tuning (Section 4.1 and Appendix C).

One fit trains with Adam for at most 1,000 epochs, with patience 200 on the validation score and a one-hour limit.
The validation score is accuracy, or ROC-AUC on Minesweeper. Test is read at the best validation epoch only. Tuning
scores a configuration by its mean validation score over seeds 11 and 29 on split 0. The selected configuration is
refitted on every official split with seeds 101 and 202, and results are reported as the mean and population standard
deviation across splits of the per-split seed mean (on single-split datasets, over the seeds).
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from .data import DATASETS, classification_metric, load_dataset
from .model import SpaGIM

HIDDEN = 512
EPOCHS, PATIENCE, MAX_SECONDS = 1000, 200, 3600
TUNING_FITS = [(0, 11), (0, 29)]  # (split, seed)
FINAL_SEEDS = [101, 202]
TRAINING_KEYS = ("lr", "weight_decay", "feature_preprocessing")


def load_config(path: str | Path) -> dict:
    """A configuration file: {"dataset": ..., "training": {lr, weight_decay, feature_preprocessing},
    "model": {SpaGIM keyword arguments}}."""
    config = json.loads(Path(path).read_text())
    if set(config) != {"dataset", "training", "model"}:
        raise ValueError(f"{path}: expected the keys dataset, training and model")
    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides (values parsed as JSON when possible) to the training or model section."""
    config = {"dataset": config["dataset"], "training": dict(config["training"]), "model": dict(config["model"])}
    for item in overrides or []:
        key, _, text = item.partition("=")
        if not key or not _:
            raise ValueError(f"override {item!r} is not key=value")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        config["training" if key in TRAINING_KEYS else "model"][key] = value
    return config


def fit(config: dict, split: int, seed: int, device: str = "cuda", data_root: str | Path = "data",
        with_test: bool = True, epochs: int = EPOCHS, patience: int = PATIENCE,
        max_seconds: float = MAX_SECONDS) -> dict:
    """Train one model at one (split, seed) and return its record."""
    started = time.time()
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.manual_seed(seed)
    training, dataset = config["training"], config["dataset"]
    data = load_dataset(dataset, root=data_root, split=split,
                        feature_preprocessing=training["feature_preprocessing"]).to(device)
    S = data.propagation.to_sparse_csr()
    model = SpaGIM(data.num_features, data.num_classes, data.num_nodes, hidden=HIDDEN, **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    score = lambda logits, mask: classification_metric(logits[mask], data.labels[mask], data.metric)
    best, stale, epoch, deadline = {"val": -math.inf, "epoch": -1}, 0, -1, started + max_seconds
    for epoch in range(epochs):
        model.train_step(data.features, S, data.labels, data.train_mask, optimizer)
        adjoint_steps = model.stats.get("adjoint_steps")
        logits = model.predict(data.features, S)
        val = score(logits, data.val_mask)
        if val > best["val"]:
            best = {"val": val, "epoch": epoch, **model.stats, "adjoint_steps": adjoint_steps,
                    "contraction": model.backbone[-1].lipschitz()}
            if with_test:
                best["test"] = score(logits, data.test_mask)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        if time.time() > deadline:
            break
    return {"dataset": dataset, "split": split, "seed": seed, "config": config, "metric": data.metric, **best,
            "best_epoch": best["epoch"], "epochs_completed": epoch + 1, "timed_out": time.time() > deadline,
            "seconds": time.time() - started, "parameters": sum(p.numel() for p in model.parameters())}


def summarize(records: list[dict], key: str = "test") -> dict:
    """Mean and population standard deviation (in %) across splits of the per-split mean over seeds; with a single
    split, over the seeds."""
    by_split: dict[int, list[float]] = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record[key])
    split_means = [statistics.fmean(scores) for scores in by_split.values()]
    flat = [score for scores in by_split.values() for score in scores]
    spread = statistics.pstdev(split_means) if len(by_split) > 1 else (
        statistics.pstdev(flat) if len(flat) > 1 else 0.0)
    return {"mean": round(100 * statistics.fmean(split_means), 2), "std": round(100 * spread, 2),
            "splits": len(by_split), "fits": len(records)}


def official_splits(dataset: str) -> list[int]:
    return list(range(DATASETS[dataset][2]))
