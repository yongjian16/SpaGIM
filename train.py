"""Fit SpaGIM with a configuration file on every official split and seed (or a chosen subset), and summarize.

    python train.py --config configs/selected/Cora.json
    python train.py --config configs/selected/Chameleon-filtered.json --splits 0 1 --seeds 101
    python train.py --config configs/selected/Chameleon-filtered.json --set departure=gate   # an ablation

Each fit writes a JSON record to OUT/s<split>_r<seed>.json; existing records are reused. The summary (mean and
population standard deviation across splits of the per-split seed mean, in %) goes to OUT/summary.json.
"""
import argparse
import json
from pathlib import Path

from spagim.protocol import FINAL_SEEDS, apply_overrides, fit, load_config, official_splits, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a configuration file, e.g. configs/selected/Cora.json")
    parser.add_argument("--splits", type=int, nargs="*", help="official splits (default: all)")
    parser.add_argument("--seeds", type=int, nargs="*", default=FINAL_SEEDS)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override training or model settings, e.g. lam=1 batch_norm=false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", help="output directory (default: results/<config name>)")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    out = Path(args.out or Path("results") / Path(args.config).stem)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    records = []
    for split in args.splits if args.splits is not None else official_splits(config["dataset"]):
        for seed in args.seeds:
            path = out / f"s{split:02d}_r{seed}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = fit(config, split, seed, device=args.device, data_root=args.data_root)
                path.write_text(json.dumps(record) + "\n")
            records.append(record)
            print(f"{config['dataset']} split {split} seed {seed}: val {100 * record['val']:.2f} "
                  f"test {100 * record['test']:.2f} (epoch {record['best_epoch']}, departing "
                  f"{100 * record.get('departing', 0):.0f}%, {record['seconds']:.0f} s)", flush=True)
    summary = {"test": summarize(records, "test"), "val": summarize(records, "val")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{config['dataset']}: test {summary['test']['mean']:.2f} ± {summary['test']['std']:.2f} "
          f"(validation {summary['val']['mean']:.2f}) over {summary['test']['fits']} fits")


if __name__ == "__main__":
    main()
