"""Parallel binary verification with the frozen SlideChat language-model head."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


_VERIFY = (
    "Determine whether the candidate answer correctly answers the pathology question "
    "using the whole-slide image.\n\nQuestion: {question}\nCandidate answer: {candidate}"
    "\n\nVerification options:\n[1] The candidate answer is correct.\n"
    "[2] The candidate answer is incorrect.\n"
    "Select one option. Answer only with its bracketed numeric identifier."
)


@dataclass
class _Branches:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    answer_positions: torch.Tensor
    suffix_tokens: torch.Tensor
    visual_start: int
    visual_length: int


def _branches(llm, tokenizer, question, candidates, visual):
    from xtuner.utils import DEFAULT_IMAGE_TOKEN, PROMPT_TEMPLATE

    if any(DEFAULT_IMAGE_TOKEN in text for text in (question, *candidates)):
        raise ValueError("Question and candidates cannot contain the image placeholder")

    embed = llm.get_input_embeddings()
    device = embed.weight.device
    sequences, positions, targets = [], [], []
    visual_start = None
    for candidate in candidates:
        text = _VERIFY.format(question=question.strip(), candidate=candidate)
        prompt = PROMPT_TEMPLATE.qwen_chat["INSTRUCTION"].format(
            input=DEFAULT_IMAGE_TOKEN + "\n" + text, round=1
        ) + "Best answer: ["
        before, after = prompt.split(DEFAULT_IMAGE_TOKEN)
        before_ids = tokenizer.encode(before, add_special_tokens=True)
        after_ids = tokenizer.encode(after, add_special_tokens=False)
        if visual_start is not None and visual_start != len(before_ids):
            raise ValueError("Candidate prompts must share the same visual prefix boundary")
        visual_start = len(before_ids)
        prefix = embed(torch.tensor(before_ids, device=device, dtype=torch.long))
        suffix = embed(torch.tensor(after_ids, device=device, dtype=torch.long))
        prefill = torch.cat((prefix, visual, suffix))
        if len(prefill) + 1 > 4096:
            raise ValueError("The verification prompt exceeds the 4096-token limit")

        for ending in ("1]", "2]"):
            complete = tokenizer.encode(after + ending, add_special_tokens=False)
            tokens = complete[len(after_ids):]
            if complete[:len(after_ids)] != after_ids or len(tokens) != 2:
                raise ValueError("Expected a SlideChat Qwen tokenizer with two-token verdicts")
            digit = embed(torch.tensor(tokens[:1], device=device))
            sequences.append(torch.cat((prefill, digit)))
            positions.append(len(prefill) - 1)
            targets.append(tokens)

    embeddings = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    mask = torch.arange(embeddings.shape[1], device=device)[None, :] < lengths[:, None]
    return _Branches(embeddings, mask.long(), torch.tensor(positions, device=device),
                     torch.tensor(targets, device=device), visual_start, len(visual))


def _decode(llm, plugin, batch, patches):
    decoder = llm.model
    h = batch.embeddings
    pos = torch.arange(h.shape[1], device=h.device)
    causal = decoder._update_causal_mask(batch.attention_mask, h, pos, None, False)
    rows = torch.arange(0, h.shape[0], 2, device=h.device)
    start, end = batch.visual_start, batch.visual_start + batch.visual_length
    if patches is not None:
        memory = plugin.memory(patches.to(device=h.device, dtype=torch.float32))
        cache = plugin.empty_cache(len(rows), h.device)

    for layer, block in enumerate(decoder.layers, 1):
        if patches is not None:
            # The two verdict branches share the same candidate-conditioned visual update.
            visual = h[rows, start:end]
            state = h[rows, batch.answer_positions[rows]]
            evidence, cache, gate = plugin(
                state, memory, cache, layer / len(decoder.layers), context_states=visual
            )
            delta = plugin.visual_residual(visual, evidence, gate, layer)
            h = h.clone()
            h[:, start:end] += delta.repeat_interleave(2, dim=0).to(h.dtype)

        h = block(h, attention_mask=causal, position_ids=pos[None, :],
                  past_key_value=None, output_attentions=False, use_cache=False,
                  cache_position=pos)[0]

    h = decoder.norm(h)
    all_rows = torch.arange(len(h), device=h.device)
    p = batch.answer_positions
    readouts = torch.stack((h[all_rows, p], h[all_rows, p + 1]), dim=1)
    logits = F.linear(readouts.float(), llm.lm_head.weight.float())
    token_scores = logits.log_softmax(-1).gather(-1, batch.suffix_tokens.unsqueeze(-1))
    verdicts = token_scores.reshape(-1, 2, 2).sum(dim=-1)
    return verdicts[:, 0] - verdicts[:, 1]


@torch.inference_mode()
def score_candidates(backbone, tokenizer, plugin, question, candidates,
                     visual_prefix=None, patches=None):
    """Return scores, probabilities and a zero-based prediction in the caller's order.

    Provide both the projected prefix and raw features for an image, or omit both
    to measure the text-only control. Inputs contain candidate text without position labels.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide a nonempty question")
    if isinstance(candidates, str) or len(candidates) < 2 or any(
        not isinstance(c, str) or not c.strip() for c in candidates
    ):
        raise ValueError("Provide at least two nonempty candidate strings")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidate text is ambiguous")
    if (visual_prefix is None) != (patches is None):
        raise ValueError("Pass visual_prefix and patches together")
    llm = backbone.llm
    weight = llm.get_input_embeddings().weight
    if visual_prefix is None:
        visual_prefix = weight.new_empty((0, weight.shape[1]))
    else:
        if visual_prefix.ndim != 2 or visual_prefix.shape[1] != weight.shape[1] or not len(visual_prefix):
            raise ValueError("Expected a nonempty projected prefix of shape [V, hidden_size]")
        visual_prefix = visual_prefix.to(device=weight.device, dtype=weight.dtype)

    order = sorted(range(len(candidates)), key=candidates.__getitem__)
    batch = _branches(llm, tokenizer, question, [candidates[i] for i in order], visual_prefix)
    canonical = _decode(llm, plugin, batch, patches)
    scores = torch.empty_like(canonical)
    scores[order] = canonical
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite candidate scores")
    tied = (scores.max() - scores <= 1e-6).nonzero().flatten().tolist()
    prediction = min(tied, key=candidates.__getitem__)
    return {"scores": scores.tolist(), "probabilities": scores.softmax(-1).tolist(),
            "prediction": prediction}
