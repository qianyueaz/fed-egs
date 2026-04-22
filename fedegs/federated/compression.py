from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CompressedTensor:
    values: torch.Tensor
    dtype_name: str
    quantized: bool
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class CompressedStateDict:
    tensors: Dict[str, CompressedTensor]
    raw_nbytes: int
    compressed_nbytes: int
    bits: int


def estimate_state_dict_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in state_dict.values())


def compress_state_dict(state_dict: Dict[str, torch.Tensor], bits: int = 8) -> CompressedStateDict:
    safe_bits = max(int(bits), 1)
    compressed_tensors: Dict[str, CompressedTensor] = {}
    raw_nbytes = estimate_state_dict_nbytes(state_dict)
    compressed_nbytes = 0

    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu()
        if not cpu_tensor.is_floating_point():
            payload = CompressedTensor(
                values=cpu_tensor.clone(),
                dtype_name=str(cpu_tensor.dtype),
                quantized=False,
            )
        else:
            payload = _quantize_tensor(cpu_tensor, safe_bits)
        compressed_tensors[name] = payload
        compressed_nbytes += int(payload.values.numel() * payload.values.element_size())

    return CompressedStateDict(
        tensors=compressed_tensors,
        raw_nbytes=raw_nbytes,
        compressed_nbytes=compressed_nbytes,
        bits=safe_bits,
    )


def decompress_state_dict(payload: CompressedStateDict) -> Dict[str, torch.Tensor]:
    return {name: _decompress_tensor(tensor_payload) for name, tensor_payload in payload.tensors.items()}


def _quantize_tensor(tensor: torch.Tensor, bits: int) -> CompressedTensor:
    dtype_name = str(tensor.dtype)
    min_value = float(tensor.min().item()) if tensor.numel() > 0 else 0.0
    max_value = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
    levels = max((1 << bits) - 1, 1)

    if tensor.numel() == 0 or math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        quantized_values = torch.zeros_like(tensor, dtype=_quantized_dtype(bits))
        return CompressedTensor(
            values=quantized_values,
            dtype_name=dtype_name,
            quantized=True,
            scale=0.0,
            offset=min_value,
        )

    scale = (max_value - min_value) / float(levels)
    quantized = torch.round((tensor - min_value) / scale).clamp_(0, levels).to(_quantized_dtype(bits))
    return CompressedTensor(
        values=quantized,
        dtype_name=dtype_name,
        quantized=True,
        scale=float(scale),
        offset=min_value,
    )


def _decompress_tensor(payload: CompressedTensor) -> torch.Tensor:
    target_dtype = _dtype_from_name(payload.dtype_name)
    if not payload.quantized:
        return payload.values.clone().to(dtype=target_dtype)

    restored = payload.values.to(torch.float32) * float(payload.scale) + float(payload.offset)
    return restored.to(dtype=target_dtype)


def _quantized_dtype(bits: int) -> torch.dtype:
    if bits <= 8:
        return torch.uint8
    if bits <= 16:
        return torch.int32
    return torch.int32


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported dtype name in compressed payload: {name}")
    dtype = getattr(torch, normalized)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Resolved attribute is not a torch dtype: {name}")
    return dtype
