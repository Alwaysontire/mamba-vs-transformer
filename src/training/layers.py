"""Реестр кастомных слоев для сборки модели"""

import math

from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import torch 
import torch.nn as nn
from torch.nn import functional as F

from src.config.schemas import ModelConfig, MambaBlockConfig, TransformerBlockConfig


LAYERS: dict[str, type[nn.Module]] = {}


def register_layer(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Декоратор регистрации слоев по имени name"""

    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in LAYERS:
            raise ValueError(f"Мы уже реализовали такой слой: {name!r}")
        LAYERS[name] = cls
        return cls
    return deco



@register_layer("mamba_block")
class MambaBlock(nn.Module):
    """Мамба блок"""
    def __init__(self, config: MambaBlockConfig):
        pass
    def forward(self):
        pass


@register_layer("rmsnorm")
class RMSnorm(nn.Module):
    """
    Вход: [batch, sequence, d_model]
    Выход: [batch, sequence, d_model]
    """

    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-6
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        norm = x_float / torch.sqrt(mean_square + self.eps)

        return (norm * self.weight.float()).to(dtype=input_dtype)


@register_layer("rope")
class RotaryEmbedding(nn.Module):
    """
        Вход: [batch, n_heads, sequence, head_dim]
        Выход: [batch, n_heads, sequence, head_dim]
    """
    
    def __init__(
            self, 
            head_dim: int, 
            base: int = 10_000
    ):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(
            self, 
            x: torch.Tensor, 
            cos: torch.Tensor,
            sin: torch.Tensor
    ) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)

        return rotated.flatten(start_dim=-2)

    def forward(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = q.size(-2)
        positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
        inv_freq = self.inv_freq.float()
        angles = torch.outer(positions, inv_freq)

        cos = angles.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=q.dtype)

        return self.rotate(q, cos, sin), self.rotate(k, cos, sin)


@register_layer("self-attention")
class CausalSelfAttention(nn.Module):
    """
        Вход: [batch, sequence, d_model]
        Выход: [batch, sequence, d_model]
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            dropout: float = 0.0,
            rope_base: int = 10_000,
            attn_implementation: Literal["eager", "sdpa"] = "sdpa",
            pos_encoding: Literal["rope", "learned", "alibi"] = "rope"
    ):
        super().__init__()

        if attn_implementation != "sdpa":
            raise ValueError(f"Неизвестный attn_implementation: {attn_implementation}")

        self.p = dropout
        self.n_heads = n_heads
        self.pos_encoding = pos_encoding
        self.head_dim = d_model // n_heads
        self.attn_implementation = attn_implementation
        
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def get_slopes(self) -> torch.Tensor:
        n = 2 ** math.floor(math.log2(self.n_heads))
        m_0 = 2.0 ** (-8.0 / n)
        m = m_0 ** torch.arange(1, n + 1, dtype=torch.float32)

        if n < self.n_heads:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = m_hat_0 ** torch.arange(1, 1 + 2 * (self.n_heads - n), 2, dtype=torch.float32)
            m = torch.cat((m, m_hat))

        return m

    def get_alibi_biases(
            self,
            sequence_length: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        slopes = self.get_slopes().to(device=device, dtype=torch.float32)
        positions = torch.arange(sequence_length, device=device)
        distance = positions[:, None] - positions[None, :]
        bias = -slopes[:, None, None] * distance.clamp_min(0)[None, :, :]
        bias = bias.masked_fill_(distance[None, None, :, :] < 0, float("-inf")).to(dtype=dtype)

        return bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        dropout_p = self.p if self.training else 0.0

        q = self.q(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        k = self.k(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        v = self.v(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.pos_encoding == "rope":
            q, k = self.rope(q, k)

        if self.pos_encoding == "alibi":
            alibi_mask = self.get_alibi_biases(sequence_length, device=q.device, dtype=q.dtype)
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=alibi_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=True,
            )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)

        return self.output_projection(attn)


@register_layer("ffn")
class FeedForward(nn.Module):
    """
        Вход: [batch, sequence, d_model]
        Выход: [batch, sequence, d_model]
    """
    def __init__(
            self, 
            d_model: int, 
            d_ff: int,
            dropout: float = 0.0,
            activation: Literal["gelu", "swiglu", "relu"] = "swiglu"
    ):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.activation == "swiglu":
            self.gate_projection = nn.Linear(d_model, d_ff)
            self.value_projection = nn.Linear(d_model, d_ff)
            self.output_projection = nn.Linear(d_ff, d_model)
        else:
            self.input_projection = nn.Linear(d_model, d_ff)
            self.output_projection = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = F.silu(self.gate_projection(x)) * self.value_projection(x)
        elif self.activation == "gelu":
            hidden = F.gelu(self.input_projection(x))
        else:
            hidden = F.relu(self.input_projection(x))

        return self.output_projection(self.dropout(hidden))


@register_layer("transformer_block")
class TransformerBlock(nn.Module):
    """
        pre-norm decoder block: 
            CausalSelfAttention + FFN + residual connection

        Вход: [batch, sequence, d_model]
        Выход: [batch, sequence, d_model]
    """
    def __init__(
            self,
            d_model: int, 
            n_heads: int,
            d_ff: int | None = None, 
            dropout: float = 0.0,
            norm: Literal["layernorm", "rmsnorm"] = "rmsnorm",
            attn_implementation: Literal["eager", "sdpa"] = "sdpa",
            activation: Literal["gelu", "swiglu", "relu"] = "swiglu",
            pos_encoding: Literal["rope", "learned", "alibi"] = "rope"
    ):
        super().__init__()

        if d_ff is None:
            d_ff = round(8 * d_model / 3) if activation == "swiglu" else 4 * d_model

        normalization = RMSnorm if norm == "rmsnorm" else nn.LayerNorm

        self.attn_norm = normalization(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            pos_encoding=pos_encoding,
            attn_implementation=attn_implementation
        )
        self.ffn_norm = normalization(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))

        return x