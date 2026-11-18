"""Score questions from precomputed patch features."""
import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .orders import option_orders


def read_features(directory, name):
    import numpy as np
    import torch

    if not isinstance(name, str) or not name:
        raise ValueError("features must be a path to a .npy file")
    array = np.load(directory / name, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[1] != 512:
        raise ValueError("expected a feature array of shape [N, 512]")
    if not len(array) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("features must contain finite floating-point patch vectors")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def predict(row, models, directory, seed, with_image_free):
    from .scoring import score_candidates

    uid, question, options = row["id"], row["question"], row["candidates"]
    if not isinstance(uid, str) or not uid.strip():
        raise ValueError("id must be a nonempty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(options, list) or not all(isinstance(s, str) and s.strip() for s in options):
        raise ValueError("candidates must be a list of nonempty strings")
    orders = option_orders(len(options), uid, seed, random_order=row.get("random_order"))
    prefix, patches = models.encode_features(read_features(directory, row["features"]))

    def score(candidates, prefix, patches):
        return score_candidates(models.backbone, models.tokenizer, models.plugin,
                                question, candidates, prefix, patches)

    result = {k: row[k] for k in ("id", "candidates", "target", "class_label") if k in row}
    result["predictions"] = {}
    result["orders"] = orders
    result["readouts"] = {}
    for name, mapping in orders.items():
        output = score([options[i] for i in mapping], prefix, patches)
        result["predictions"][name] = mapping[output["prediction"]]
        # Store each score vector in the same order as the input candidates.
        readout = {}
        for field in ("scores", "probabilities"):
            values = [None] * len(options)
            for displayed, original in enumerate(mapping):
                values[original] = output[field][displayed]
            readout[field] = values
        result["readouts"][name] = readout
    controls = {}
    if row.get("mismatched_features") is not None:
        other = read_features(directory, row["mismatched_features"])
        controls["mismatched"] = score(options, *models.encode_features(other))["prediction"]
    if with_image_free:
        controls["image_free"] = score(options, None, None)["prediction"]
    if controls:
        result["controls"] = controls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Questions in JSONL format")
    parser.add_argument("--output", type=Path, required=True, help="Predictions in JSONL format")
    parser.add_argument("--slidechat-checkpoint", required=True, help="Official model state file")
    parser.add_argument("--trust-slidechat-checkpoint", action="store_true",
                        help="Allow loading a trusted official SlideChat PyTorch checkpoint")
    parser.add_argument("--qwen-dir", required=True, help="Local Qwen model and tokenizer directory")
    parser.add_argument("--plugin-weights", required=True, help="SlideJev inference weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fixed random order")
    parser.add_argument("--image-free", action="store_true", help="Also score each question without an image")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must refer to different files")

    import torch
    from .loading import load_models

    models = load_models(args.slidechat_checkpoint, args.qwen_dir, args.plugin_weights, device=args.device,
                         trust_slidechat_checkpoint=args.trust_slidechat_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    temporary = None
    try:
        with args.input.open(encoding="utf-8") as source, NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as sink, torch.inference_mode():
            temporary = Path(sink.name)
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row["id"] in seen:
                        raise ValueError("duplicate question id")
                    result = predict(row, models, args.input.parent, args.seed, args.image_free)
                    seen.add(row["id"])
                    sink.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                except (KeyError, ValueError, TypeError, OSError) as exc:
                    raise ValueError(f"input line {line_number}: {exc}") from exc
            if not seen:
                raise ValueError("the input contains no questions")
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
