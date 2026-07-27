#!/usr/bin/env python3
"""Train per-layer classifiers on extracted NIAH hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = np.load(args.input_npz, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    layer_count = features.shape[1]
    indices = np.arange(features.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=labels if len(set(labels.tolist())) > 1 else None,
    )

    rows = []
    for layer_idx in range(layer_count):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=args.max_iter, multi_class="auto"),
        )
        clf.fit(features[train_idx, layer_idx, :], labels[train_idx])
        pred = clf.predict(features[test_idx, layer_idx, :])
        rows.append(
            {
                "layer_idx": layer_idx,
                "accuracy": float(accuracy_score(labels[test_idx], pred)),
                "num_train": int(len(train_idx)),
                "num_test": int(len(test_idx)),
            }
        )

    output = {
        "input_npz": str(args.input_npz),
        "condition": str(data["condition"]) if "condition" in data.files else "unknown",
        "layers": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    best = max(rows, key=lambda row: row["accuracy"])
    print(f"best layer {best['layer_idx']} accuracy={best['accuracy']:.3f}")


if __name__ == "__main__":
    main()
