"""Candidate-conditioned patch reading, comparison and visual-token updates."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PatchMemory:
    keys: torch.Tensor
    values: tuple
    match_keys: torch.Tensor


@dataclass
class EvidenceCache:
    evidence: torch.Tensor
    valid: torch.Tensor


def _partitioned_mlp(network, features, first, second):
    """Evaluate two channel groups in the same accumulation order as the model."""
    layer, activation, output = network
    hidden = F.linear(features[:, first].contiguous(), layer.weight[:, first].contiguous(), layer.bias)
    hidden = hidden + F.linear(features[:, second].contiguous(), layer.weight[:, second].contiguous())
    return output(activation(hidden))


class EvidencePlugin(nn.Module):
    """Shared reader and writer used before each language decoder block.

    Load the plugin state dictionary strictly before inference. The reader
    operates on float32 features and casts its output to the decoder dtype.
    """

    def __init__(self, hidden_size=3584, patch_dim=512, dimension=128,
                 value_dimension=512, top_m=8, region_budget=32,
                 visual_writer_layers=28, visual_write_max_ratio=0.25):
        super().__init__()
        if not 0 < dimension < value_dimension:
            raise ValueError("The value width must exceed the positive query width.")
        if min(top_m, region_budget, visual_writer_layers) < 1:
            raise ValueError("Region budgets and layer count must be positive.")
        self.dimension = dimension
        self.value_dimension = value_dimension
        self.top_m = top_m
        self.region_budget = region_budget
        self.visual_write_max_ratio = visual_write_max_ratio

        self.query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dimension))
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.key = nn.Linear(patch_dim, dimension, bias=False)
        self.value = nn.Linear(patch_dim, value_dimension, bias=False)
        self.match_query = nn.Linear(dimension, dimension, bias=False)
        self.match_key = nn.Linear(dimension, dimension, bias=False)
        self.router = nn.Sequential(
            nn.Linear(2 * dimension + 2 * value_dimension + 3, dimension),
            nn.GELU(), nn.Linear(dimension, 1),
        )
        self.region_weight = nn.Sequential(
            nn.Linear(2 * dimension + value_dimension + 4, dimension),
            nn.GELU(), nn.Linear(dimension, 1),
        )
        self.self_out = nn.Linear(value_dimension, hidden_size, bias=False)
        self.compare_out = nn.Linear(value_dimension, hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_key = nn.Linear(hidden_size, dimension, bias=False)
        self.context_value = nn.Linear(hidden_size, dimension, bias=False)
        self.context_to_raw = nn.Linear(dimension, dimension, bias=False)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.visual_down = nn.Linear(hidden_size, dimension, bias=False)
        self.visual_condition_norm = nn.LayerNorm(hidden_size)
        self.visual_condition = nn.Linear(hidden_size, dimension, bias=False)
        self.visual_up = nn.Linear(dimension, hidden_size, bias=False)
        self.visual_gain_logits = nn.Parameter(torch.zeros(visual_writer_layers))

        # Inference reads the key-width and remaining value channels separately.
        d, v = dimension, value_dimension
        self._router_first = list(range(3*d)) + list(range(2*d+v, 3*d+v)) + list(range(2*d+2*v, 2*d+2*v+3))
        self._router_second = list(range(3*d, 2*d+v)) + list(range(3*d+v, 2*d+2*v))
        self._region_first = list(range(3*d)) + list(range(2*d+v, 2*d+v+4))
        self._region_second = slice(3*d, 2*d+v)

    def memory(self, patches):
        """Project a single slide's [patches, patch_dim] feature bank once."""
        if patches.ndim != 2 or not len(patches) or patches.size(1) != self.patch_norm.normalized_shape[0]:
            raise ValueError("Expected a nonempty [patches, patch_dim] feature bank.")
        x = self.patch_norm(patches.float())
        keys = self.key(x)
        d = self.dimension
        values = (F.linear(x, self.value.weight[:d].contiguous()),
                  F.linear(x, self.value.weight[d:].contiguous()))
        return PatchMemory(keys, values, self.match_key(keys))

    def empty_cache(self, count, device):
        return EvidenceCache(torch.zeros(count, self.value_dimension, device=device, dtype=torch.float32),
                             torch.zeros(count, device=device, dtype=torch.float32))

    def _gates(self, queries, cache, depth):
        pairs = torch.triu_indices(len(queries), len(queries), offset=1, device=queries.device)
        left, right = pairs
        a, b = queries[left], queries[right]
        e, f = cache.evidence[left], cache.evidence[right]
        v, w = cache.valid[left, None], cache.valid[right, None]
        features = torch.cat((a+b, (a-b).abs(), e+f, (e-f).abs(),
                              v+w, (v-w).abs(), torch.full_like(v, depth)), dim=-1)
        logits = _partitioned_mlp(self.router, features, self._router_first, self._router_second)
        z = logits.squeeze(-1).sigmoid()
        pair_gates = queries.new_zeros((len(queries), len(queries)))
        pair_gates[left, right] = z
        pair_gates[right, left] = z
        candidate_gate = 1 - (1 - pair_gates).prod(dim=-1)
        return pairs, pair_gates, candidate_gate

    def _regions(self, attention):
        top = torch.argsort(attention, dim=-1, descending=True, stable=True)[:, :self.top_m]
        union = torch.unique(top.flatten(), sorted=True)
        if len(union) > self.region_budget:
            priority = attention[:, union].sum(dim=0)
            keep = torch.argsort(priority, descending=True, stable=True)[:self.region_budget]
            union = union[keep].sort().values
        return union

    def _value_output(self, layer, value):
        d = self.dimension
        return (F.linear(value[:, :d].contiguous(), layer.weight[:, :d].contiguous())
                + F.linear(value[:, d:].contiguous(), layer.weight[:, d:].contiguous()))

    def forward(self, states, memory, cache, depth, context_states):
        """Return one delta and gate per candidate, and the updated evidence cache.

        states are [candidates, hidden], context_states are [candidates, tokens,
        hidden], and depth is the one-based layer index divided by layer count.
        Return order: (delta, cache, gate).
        """
        if states.ndim != 2 or len(states) < 2:
            raise ValueError("Expected at least two candidate states.")
        if context_states.ndim == 2:
            context_states = context_states.unsqueeze(0).expand(len(states), -1, -1)
        if (context_states.ndim != 3 or context_states.size(0) != len(states)
                or context_states.size(1) == 0 or context_states.size(2) != states.size(1)):
            raise ValueError("Expected nonempty [candidates, visual_tokens, hidden] context.")

        queries = self.query(states.float())
        pairs, pair_gates, gate = self._gates(queries, cache, depth)
        scale = math.sqrt(self.dimension)
        context = self.context_norm(context_states.float())
        context_scores = torch.bmm(queries[:, None], self.context_key(context).transpose(1, 2)) / scale
        context_read = torch.bmm(context_scores.softmax(-1), self.context_value(context)).squeeze(1)
        raw_queries = queries + self.context_to_raw(context_read)
        attention = (raw_queries @ memory.keys.T / scale).softmax(-1)
        evidence = torch.cat([attention @ part for part in memory.values], dim=-1)
        updated = EvidenceCache((1 - gate[:, None]) * cache.evidence + gate[:, None] * evidence,
                                1 - (1 - cache.valid) * (1 - gate))

        union = self._regions(attention)
        scores = self.match_query(raw_queries) @ memory.match_keys[union].T / scale
        region_values = torch.cat([part[union] for part in memory.values], dim=-1)
        messages = torch.zeros_like(evidence)
        for left, right in pairs.T.tolist():
            symmetric_query = torch.cat((queries[left] + queries[right],
                                         (queries[left] - queries[right]).abs()))
            sa, sb = scores[left], scores[right]
            aa, ab = attention[left, union], attention[right, union]
            stats = torch.stack((sa+sb, (sa-sb).abs(), aa+ab, (aa-ab).abs()), dim=-1)
            features = torch.cat((symmetric_query.expand(len(union), -1), region_values, stats), dim=-1)
            logits = _partitioned_mlp(self.region_weight, features, self._region_first, self._region_second)
            weights = logits.squeeze(-1).softmax(0)
            direction = weights[:, None] * (sa-sb).tanh()[:, None]
            d = self.dimension
            message = torch.cat([(direction * part.contiguous()).sum(0)
                                 for part in (region_values[:, :d], region_values[:, d:])])
            contribution = pair_gates[left, right] * message
            sign = queries.new_zeros(len(states))
            sign[left], sign[right] = 1, -1
            messages = messages + sign[:, None] * contribution
        delta = self._value_output(self.compare_out, messages / (len(states)-1))
        delta = delta + self._value_output(self.self_out, evidence)
        return delta.to(states.dtype), updated, gate

    def visual_residual(self, visual, evidence_delta, gate, layer):
        """Produce a per-token residual using the gain for a one-based layer."""
        if not 1 <= layer <= len(self.visual_gain_logits):
            raise ValueError("Invalid visual writer layer.")
        local = self.visual_down(self.visual_norm(visual.float()))
        condition = self.visual_condition(self.visual_condition_norm(evidence_delta.float()))
        raw = evidence_delta[:, None].float() + self.visual_up(F.gelu(local + condition[:, None]))
        reference_rms = visual.float().square().mean(-1, keepdim=True).sqrt()
        update_rms = raw.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        strength = self.visual_write_max_ratio * self.visual_gain_logits[layer-1].sigmoid()
        return raw * (reference_rms / update_rms) * strength * gate[:, None, None]
