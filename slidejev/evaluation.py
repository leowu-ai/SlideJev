"""Evaluate original-index predictions from a labeled JSON Lines file."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from .orders import ORDER_NAMES


def _check_index(value, count, field, *, allow_invalid=False):
    if value is None and allow_invalid:
        return
    if type(value) is not int or not 0 <= value < count:
        suffix = " or null for an invalid prediction" if allow_invalid else ""
        raise ValueError(f"{field} must be a zero-based candidate index{suffix}")


def evaluate(rows):
    """Return percentage metrics; control gains use paired rows and percentage points."""
    seen = set()
    correct = Counter()
    consistent = 0
    class_counts = Counter()
    class_hits = {name: Counter() for name in ORDER_NAMES}
    control_hits = {name: Counter() for name in ("mismatched", "image_free")}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each example must be a JSON object")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError(f"expected a unique nonempty string id, got {item_id!r}")
        seen.add(item_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"{item_id}: expected at least two candidates")
        target = row.get("target")
        _check_index(target, len(candidates), f"{item_id}: target")
        predictions = row.get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != set(ORDER_NAMES):
            raise ValueError(f"{item_id}: predictions require {', '.join(ORDER_NAMES)}")
        label = row.get("class_label")
        if label is not None:
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{item_id}: class_label must be a semantic class name")
            class_counts[label] += 1
        for name in ORDER_NAMES:
            prediction = predictions[name]
            _check_index(prediction, len(candidates), f"{item_id}: {name}", allow_invalid=True)
            hit = prediction == target
            correct[name] += hit
            if label is not None:
                class_hits[name][label] += hit
        original = predictions["original"]
        consistent += original is not None and all(v == original for v in predictions.values())
        controls = row.get("controls", {})
        if not isinstance(controls, dict) or controls.keys() - control_hits.keys():
            raise ValueError(f"{item_id}: controls accepts mismatched and image_free")
        for name, prediction in controls.items():
            _check_index(prediction, len(candidates), f"{item_id}: {name}", allow_invalid=True)
            control_hits[name]["n"] += 1
            control_hits[name]["correct_image"] += original == target
            control_hits[name]["control"] += prediction == target

    n = len(seen)
    if not n:
        raise ValueError("predictions file contains no examples")
    acc = {name: 100 * correct[name] / n for name in ORDER_NAMES}
    result = {
        "n_examples": n,
        "accuracy_pct": acc,
        "four_order_mean_accuracy_pct": fmean(acc.values()),
        "answer_consistency_pct": 100 * consistent / n,
    }
    if class_counts:
        if sum(class_counts.values()) != n:
            raise ValueError("supply class_label for every row or omit it for every row")
        balanced = {
            name: 100 * fmean(class_hits[name][k] / v for k, v in class_counts.items())
            for name in ORDER_NAMES
        }
        result["class_support"] = dict(class_counts)
        result["balanced_accuracy_pct"] = balanced
        result["four_order_mean_balanced_accuracy_pct"] = fmean(balanced.values())
    if any(control_hits.values()):
        result["image_controls"] = {
            name: {
                "n_paired": counts["n"],
                "correct_image_accuracy_pct": 100 * counts["correct_image"] / counts["n"],
                "control_accuracy_pct": 100 * counts["control"] / counts["n"],
                "gain_pp": 100 * (counts["correct_image"] - counts["control"]) / counts["n"],
            }
            for name, counts in control_hits.items() if counts
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Labeled predictions in JSONL format")
    args = parser.parse_args()
    try:
        with args.predictions.open(encoding="utf-8") as file:
            result = evaluate(json.loads(line) for line in file if line.strip())
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
