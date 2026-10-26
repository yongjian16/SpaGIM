"""Datasets, official splits, deterministic preprocessing and metrics.

Every model sees the same tensors: the undirected, unweighted, deduplicated graph with one self loop per node and its
symmetric normalization A = (D + I)^-1/2 (A + I) (D + I)^-1/2, the node features (optionally L1-normalized per node),
and the official masks of the requested split. Raw files are downloaded on first use into `root`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

# name -> (source, primary metric, number of official splits)
DATASETS: dict[str, tuple[str, str, int]] = {
    "Roman-empire": ("heterophilous", "accuracy", 10),
    "Amazon-ratings": ("heterophilous", "accuracy", 10),
    "Minesweeper": ("heterophilous", "roc_auc", 10),
    "Chameleon-filtered": ("platonov_npz", "accuracy", 10),
    "Squirrel-filtered": ("platonov_npz", "accuracy", 10),
    "Actor": ("platonov_npz", "accuracy", 10),
    "Cora": ("planetoid", "accuracy", 1),
    "CiteSeer": ("planetoid", "accuracy", 1),
    "synthetic": ("synthetic", "accuracy", 1),  # a 30-node offline graph for smoke tests
}

PLATONOV_URL = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"
PLATONOV_FILES = {"Actor": "actor.npz", "Chameleon-filtered": "chameleon_filtered.npz",
                  "Squirrel-filtered": "squirrel_filtered.npz"}
PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_PARTS = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")


@dataclass
class GraphData:
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    propagation: torch.Tensor  # A, sparse COO
    metric: str

    @property
    def num_nodes(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item()) + 1

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    def to(self, device) -> "GraphData":
        return GraphData(*(getattr(self, name).to(device) for name in
                           ("features", "labels", "train_mask", "val_mask", "test_mask", "propagation")),
                         metric=self.metric)


def _fetch(url: str, path: Path) -> None:
    """Download one file if absent, writing it atomically."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        payload = response.read()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def normalized_propagation(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """(D + I)^-1/2 (A + I) (D + I)^-1/2 of the undirected, unweighted, deduplicated graph."""
    edge_index = edge_index.long().cpu()
    edges = edge_index[:, edge_index[0] != edge_index[1]]
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (n_nodes, n_nodes)).coalesce()
    diagonal = torch.arange(n_nodes)
    indices = torch.cat((adj.indices(), torch.stack((diagonal, diagonal))), dim=1)
    values = torch.ones(indices.shape[1])
    degree = torch.zeros(n_nodes).scatter_add_(0, indices[0], values)
    values = degree[indices[0]].rsqrt() * degree[indices[1]].rsqrt()
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


def _mask_for_split(mask: torch.Tensor, split: int) -> torch.Tensor:
    if mask.ndim == 1:
        if split != 0:
            raise ValueError("This dataset has only split 0")
        return mask.bool()
    if mask.ndim != 2 or not 0 <= split < mask.shape[1]:
        raise ValueError(f"Invalid split {split} for mask shape {tuple(mask.shape)}")
    return mask[:, split].bool()


def _synthetic():
    generator = torch.Generator().manual_seed(731)
    n = 30
    labels = torch.arange(n) % 3
    features = torch.randn(n, 8, generator=generator) * 0.15
    features[:, :3] += torch.nn.functional.one_hot(labels, 3).float()
    vertices = torch.arange(n)
    edges = torch.stack((vertices, (vertices + 3) % n))
    edges = torch.cat((edges, torch.stack((vertices, (vertices + 1) % n))), dim=1)
    return features, labels, edges, vertices < 12, (vertices >= 12) & (vertices < 21), vertices >= 21


def _load_platonov(name: str, root: Path, split: int):
    """One npz file of the heterophilous-graphs repository (Platonov et al., 2023)."""
    path = root / "platonov" / "raw" / PLATONOV_FILES[name]
    _fetch(f"{PLATONOV_URL}/{PLATONOV_FILES[name]}", path)
    raw = np.load(path)
    features = torch.from_numpy(raw["node_features"]).float()
    labels = torch.from_numpy(raw["node_labels"]).reshape(-1).long()
    edges = torch.from_numpy(raw["edges"]).t().contiguous().long()
    masks = tuple(_mask_for_split(torch.from_numpy(raw[f"{part}_masks"]).t().contiguous(), split)
                  for part in ("train", "val", "test"))
    return (features, labels, edges, *masks)


def load_dataset(name: str, root: str | Path = "data", split: int = 0,
                 feature_preprocessing: str = "row_l1") -> GraphData:
    """One dataset at one official split. feature_preprocessing: "row_l1" (per-node L1 normalization) or "none"."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    source, metric, splits = DATASETS[name]
    if not 0 <= split < splits:
        raise ValueError(f"{name} has splits 0 through {splits - 1}")
    if feature_preprocessing not in ("row_l1", "none"):
        raise ValueError('feature_preprocessing must be "row_l1" or "none"')
    root = Path(root)
    if source == "synthetic":
        features, labels, edges, train, val, test = _synthetic()
    elif source == "platonov_npz":
        features, labels, edges, train, val, test = _load_platonov(name, root, split)
    else:
        from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
        if source == "planetoid":
            raw_dir = root / "planetoid" / name / "raw"
            for part in PLANETOID_PARTS:
                filename = f"ind.{name.lower()}.{part}"
                _fetch(f"{PLANETOID_URL}/{filename}", raw_dir / filename)
            dataset = Planetoid(str(root / "planetoid"), name, split="public")
        else:
            dataset = HeterophilousGraphDataset(str(root / "heterophilous"), name=name)
        raw = dataset[0]
        features, labels, edges = raw.x.float(), raw.y.reshape(-1).long(), raw.edge_index
        train, val, test = (_mask_for_split(getattr(raw, f"{part}_mask"), split) for part in ("train", "val", "test"))
    if ((train & val) | (train & test) | (val & test)).any():
        raise ValueError("Train, validation, and test masks overlap")
    if feature_preprocessing == "row_l1":  # per node; uses no label and no cross-split statistic
        features = features / features.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return GraphData(features, labels, train, val, test, normalized_propagation(edges, len(labels)), metric)


def classification_metric(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> float:
    """Accuracy, or ROC-AUC of the positive-class score logit_1 - logit_0 (ties averaged, as sklearn)."""
    if metric == "accuracy":
        return float((logits.argmax(dim=1) == labels).float().mean())
    if metric != "roc_auc":
        raise ValueError(f"Unknown metric {metric!r}")
    if logits.shape[1] != 2:
        raise ValueError("ROC-AUC is defined here for binary tasks only")
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC is undefined when the evaluated mask holds one class")
    scores = (logits[:, 1] - logits[:, 0]).double()
    order = scores.argsort()
    ordered = scores[order]
    count = ordered.numel()
    positions = torch.arange(1, count + 1, dtype=torch.float64, device=scores.device)
    starts_group = torch.ones(count, dtype=torch.bool, device=scores.device)
    starts_group[1:] = ordered[1:] != ordered[:-1]
    group = starts_group.cumsum(0) - 1
    totals = torch.zeros(int(group[-1]) + 1, dtype=torch.float64, device=scores.device)
    sizes = torch.zeros_like(totals)
    totals.scatter_add_(0, group, positions)
    sizes.scatter_add_(0, group, torch.ones_like(positions))
    ranks = torch.empty_like(positions)
    ranks[order] = (totals / sizes)[group]
    return (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2) / (positives * negatives)
