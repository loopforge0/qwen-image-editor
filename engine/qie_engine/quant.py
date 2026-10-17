"""INT8 layers backed by comfy-kitchen kernels.

The checkpoints store each quantized linear as three tensors: ``<name>.weight`` (int8),
``<name>.weight_scale`` (float32, one per output row) and ``<name>.comfy_quant`` (a JSON
blob, e.g. ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``).
ConvRot weights are stored rotated; comfy-kitchen rotates the activations to match.
"""

import json

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from torch import nn


def parse_quant_config(blob: torch.Tensor) -> dict:
    return json.loads(blob.numpy().tobytes().decode())


class Int8Linear(nn.Module):
    """Linear layer whose weight stays int8; activations are quantized per row on the fly."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        convrot: bool,
        convrot_groupsize: int = 256,
        input_act: str | None = None,
    ):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.convrot = convrot
        self.convrot_groupsize = convrot_groupsize
        self.input_act = input_act
        self.out_features, self.in_features = weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ck.int8_linear(
            x,
            self.weight,
            self.weight_scale,
            None,
            x.dtype,
            convrot=self.convrot,
            convrot_groupsize=self.convrot_groupsize,
            input_act=self.input_act,
        )

    def extra_repr(self) -> str:
        act = f", input_act={self.input_act}" if self.input_act else ""
        return f"in={self.in_features}, out={self.out_features}, convrot={self.convrot}{act}"


class Int8Embedding(nn.Module):
    """Embedding table kept in int8; only the looked-up rows are dequantized."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, config: dict, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.config = config
        self.dtype = dtype
        self.num_embeddings, self.embedding_dim = weight.shape

    def _params(self) -> TensorWiseINT8Layout.Params:
        return TensorWiseINT8Layout.Params(
            scale=self.weight_scale,
            orig_dtype=self.dtype,
            orig_shape=tuple(self.weight.shape),
            convrot=self.config.get("convrot", False),
            convrot_groupsize=self.config.get("convrot_groupsize", 256),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return TensorWiseINT8Layout.dequantize_embedding(self.weight, self._params(), indices)


class FusedSwiGLU(nn.Module):
    """``out(silu(gate) * up)`` with gate and up in one GEMM, and the SwiGLU folded into the
    quantizer of the ``out`` GEMM so the activated tensor never round-trips through memory."""

    def __init__(self, gate_up: Int8Linear, out: Int8Linear):
        super().__init__()
        self.gate_up = gate_up
        self.out = out
        self.out.input_act = "swiglu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.gate_up(x))
