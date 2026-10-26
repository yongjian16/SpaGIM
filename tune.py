"""Tune SpaGIM on one dataset under the protocol's budget, then refit the selected configuration.

    python tune.py --dataset Cora
    python tune.py --dataset Chameleon-filtered --set departure=gate   # tune an ablation variant in the same space

Search: 30 trials of Optuna's tree-structured Parzen estimator (8 random start-up trials) over the space in
configs/search/<dataset>.json. A trial is scored by its mean validation score over seeds 11 and 29 on split 0; no
search fit reads test. Final: the best trial (ties go to the lower trial number) is refitted on every official split
with seeds 101 and 202, and the test score is summarized as in train.py.

A search file holds "fixed" settings, a "space" of categorical choices, and optional "bundles": a choice named in a
bundle expands into several settings (for example a backbone name into its depth, injection, and head).
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import optuna

from spagim.protocol import (FINAL_SEEDS, TRAINING_KEYS, TUNING_FITS, apply_overrides, fit, official_splits,
                             summarize)


def make_config(dataset: str, search: dict, params: dict) -> dict:
    flat = dict(search.get("fixed", {}))
    for name, choice in params.items():
        if name in search.get("bundles", {}):
            flat.update(search["bundles"][name][choice])
        else:
            flat[name] = choice
    return {"dataset": dataset, "training": {k: v for k, v in flat.items() if k in TRAINING_KEYS},
            "model": {k: v for k, v in flat.items() if k not in TRAINING_KEYS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--search", help="search-space file (default: configs/search/<dataset>.json)")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--startup", type=int, default=8)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="settings fixed for every trial")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", help="output directory (default: results/tune/<dataset>)")
    parser.add_argument("--search-only", action="store_true", help="skip the final refits")
    args = parser.parse_args()

    search = json.loads(Path(args.search or Path("configs/search") / f"{args.dataset}.json").read_text())
    out = Path(args.out or Path("results/tune") / args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "trials.jsonl"
    seed = int(hashlib.sha256(f"spagim/{args.dataset}".encode()).hexdigest()[:8], 16)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True,
                                         n_startup_trials=args.startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {name: trial.suggest_categorical(name, choices) for name, choices in search["space"].items()}
        config = apply_overrides(make_config(args.dataset, search, params), args.set)
        records = [fit(config, split, seed_, device=args.device, data_root=args.data_root, with_test=False)
                   for split, seed_ in TUNING_FITS]
        value = statistics.fmean(r["val"] for r in records)
        with log.open("a") as handle:
            handle.write(json.dumps({"trial": trial.number, "params": params, "config": config, "value": value})
                         + "\n")
        print(f"trial {trial.number}: validation {100 * value:.2f} {params}", flush=True)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, catch=(RuntimeError,))
    done = [t for t in study.trials if t.value is not None]
    best = sorted(done, key=lambda t: (-t.value, t.number))[0]
    config = apply_overrides(make_config(args.dataset, search, best.params), args.set)
    (out / "selected.json").write_text(json.dumps({**config, "search_value": best.value}, indent=1) + "\n")
    print(f"selected trial {best.number}: validation {100 * best.value:.2f}")
    if args.search_only:
        return
    records = []
    for split in official_splits(args.dataset):
        for seed_ in FINAL_SEEDS:
            path = out / "final" / f"s{split:02d}_r{seed_}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = fit(config, split, seed_, device=args.device, data_root=args.data_root)
            path.write_text(json.dumps(record) + "\n")
            records.append(record)
    summary = summarize(records, "test")
    (out / "final" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{args.dataset}: test {summary['mean']:.2f} ± {summary['std']:.2f} over {summary['fits']} fits")


if __name__ == "__main__":
    main()
