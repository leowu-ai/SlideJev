# SlideJev

Inference and evaluation code for **SlideJev: Order-Stable and Tissue-Grounded Jev Decisions for Whole-Slide Pathology**.

SlideJev verifies each candidate answer in a separate prompt. A shared evidence module reads patch features, compares candidate responses on the same tissue regions, and updates visual tokens in the decoder. The language-model head scores each candidate by the difference between the log-likelihoods of the support and rejection responses.

## Evaluate predictions

From the repository root, run the evaluator on two hand-written example predictions:

```bash
python -m slidejev.evaluation examples/predictions.jsonl
```

This example requires only Python 3.10 or later. The expected four-order mean accuracy is **87.5%** and the answer consistency is **50%**. The example illustrates the file format.

Each JSONL row describes one question. Use a unique string `id`, the list of `candidates`, a zero-based `target`, and a `predictions` object with the keys `original`, `reversed`, `cyclic`, and `random`. All prediction indices refer to the **original** candidate list. An invalid answer is represented by `null`. It counts as incorrect and prevents that question from counting as consistent.

The evaluator reports per-order accuracy and its four-order mean. A question is consistent when all four valid predictions select the same candidate, including when that candidate is incorrect. For a fixed-class task, provide the ground-truth semantic `class_label` for each row to obtain balanced accuracy (mean recall over the observed classes). Evaluate each fixed-class task separately. Optional `controls.mismatched` and `controls.image_free` are original-order predictions. Each gain is the correct-image accuracy minus the respective control accuracy in percentage points, calculated on rows with that control.

## Run inference

**Weight availability.** This release provides the implementation. Reproducing the trained model's predictions requires the SlideJev plugin checkpoint, which is not included in this release. Inference also uses the official SlideChat Stage 2 checkpoint and the matching Qwen2.5-7B-Instruct model/tokenizer directory.

1. Set up a CUDA environment with the [official SlideChat repository](https://github.com/uni-medical/SlideChat). Use its bundled `xtuner` package. The reference implementation uses SlideChat revision `fea92b61f46e0a3786b9674712454bb994c17eb7` and Transformers 4.42.4.
2. Obtain the [SlideChat Stage 2 model state](https://huggingface.co/General-Medical-AI/SlideChat_Weight/tree/main/stage2_pth) and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Install this package with `python -m pip install -e ".[inference]"` in the SlideChat environment.
3. Supply a SlideJev plugin state dictionary in safetensors format and one JSONL row per question. The command below uses example paths. The trust flag permits deserialization of the official PyTorch checkpoint. Use it only with a checkpoint from a trusted source.

```bash
python -m slidejev.infer \
  --slidechat-checkpoint models/slidechat/mp_rank_00_model_states.pt \
  --trust-slidechat-checkpoint \
  --qwen-dir models/Qwen2.5-7B-Instruct \
  --plugin-weights models/slidejev_plugin.safetensors \
  --input examples/questions.jsonl \
  --output predictions.jsonl
```

Supply precomputed CONCH patch features as a floating-point NumPy array of shape `[N, 512]` in slide order. The `features` path is relative to the input JSONL file. The hand-written question in `examples/questions.jsonl` references a feature file that you supply. Candidate strings contain answer text without position labels.

Each output row includes scores and probabilities in the original candidate order, the selected original candidate index for each ordering, and the exact displayed-to-original permutations. The cyclic order shifts the first candidate to the end. A supplied zero-based `random_order` replays a specific permutation. Otherwise the CLI generates a fixed permutation from the question ID and seed.

To also score the question with a different slide, provide `mismatched_features` in its input row. Add `--image-free` to compute the original-order image-free prediction. The resulting JSONL can be passed directly to the evaluator when ground-truth labels are supplied.
