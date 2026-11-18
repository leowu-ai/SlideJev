"""Load local SlideChat and SlideJev inference weights and encode patch features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .evidence import EvidencePlugin


def _evenly_spaced(length: int, limit: int):
    if length <= limit:
        return slice(None)
    return np.linspace(0, length - 1, limit, dtype=np.int64)


@dataclass
class LoadedModels:
    backbone: torch.nn.Module
    tokenizer: Any
    plugin: EvidencePlugin
    device: torch.device

    @torch.inference_mode()
    def encode_features(self, features):
        """Return visual prefix [min(N,512),3584] and original patch memory [N,512]."""
        patches = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if patches.ndim != 2 or patches.shape[1] != 512 or not len(patches):
            raise ValueError("Expected a nonempty [N, 512] matrix of CONCH patch features")
        if not torch.isfinite(patches).all():
            raise ValueError("Patch features must be finite")
        image = patches[_evenly_spaced(len(patches), 38_400)].to(torch.bfloat16)
        outputs = self.backbone.LongNet_encoder(
            src_tokens=None, token_embeddings=image.unsqueeze(1)
        )
        encoded = outputs["encoder_out"].transpose(0, 1)
        selected = encoded[:, _evenly_spaced(encoded.shape[1], 512)]
        prefix = self.backbone.projector(selected).squeeze(0)
        if not torch.isfinite(prefix).all():
            raise ValueError("The visual encoder returned nonfinite values")
        return prefix, patches


def _load_backbone_state(path: Path, trust_checkpoint: bool):
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    if not trust_checkpoint:
        raise ValueError(
            "The official SlideChat .pt includes pickled metadata. Only enable "
            "trust_slidechat_checkpoint for a checkpoint obtained from the official "
            "publisher, or supply a tensor-only .safetensors file."
        )
    # Older NumPy installations need the newer module names from the publisher's pickle.
    import sys
    try:
        import numpy._core.multiarray  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected a SlideChat model checkpoint dictionary")
    return payload.get("module", payload)


def load_models(slidechat_checkpoint, qwen_dir, plugin_weights, device="cuda", *,
                trust_slidechat_checkpoint=False) -> LoadedModels:
    """Restore the official backbone and the strictly matching SlideJev plugin."""
    checkpoint = Path(slidechat_checkpoint)
    qwen_dir = Path(qwen_dir)
    weights = Path(plugin_weights)
    for file in (checkpoint, weights):
        if not file.is_file():
            raise FileNotFoundError(file)
    if not qwen_dir.is_dir():
        raise NotADirectoryError(qwen_dir)
    if weights.suffix != ".safetensors":
        raise ValueError("Provide a tensor-only SlideJev plugin .safetensors file")

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The official SlideChat LongNet requires CUDA for inference")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xtuner.model.llava import LLaVAModel

    plugin = EvidencePlugin()
    plugin.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    with torch.cuda.device(device):
        # Initializing from Qwen also initializes rotary buffers omitted from the checkpoint.
        llm = AutoModelForCausalLM.from_pretrained(
            str(qwen_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        backbone = LLaVAModel(
            llm=llm, pretrained_pth=None, freeze_llm=False, train_stage="2",
            hidden_size=512, enable_long_net=True, use_activation_checkpointing=False,
        )
        if len(llm.model.layers) != 28 or llm.config.hidden_size != 3584:
            raise ValueError("Expected the Qwen2.5-7B SlideChat architecture")
        state = _load_backbone_state(checkpoint, trust_slidechat_checkpoint)
        backbone.load_state_dict(state, strict=True)
        del state
        backbone.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
        backbone.llm.config.use_cache = False
        plugin.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(qwen_dir), local_files_only=True, encode_special_tokens=True,
    )
    return LoadedModels(backbone, tokenizer, plugin, device)
