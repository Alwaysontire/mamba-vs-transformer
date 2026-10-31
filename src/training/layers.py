"""Реестр кастомных слоев для сборки модели"""


from __future__ import annotations
from collections.abc import Callable
from typing import Literal

import torch 
import torch.nn as nn
import math
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
    def __init__(self, 
                config: MambaBlockConfig,
                d_model: int,
            ) -> None:
        super().__init__()

        self.mixer = MixerBlock(
            d_model=d_model,
            d_state=config.d_state, 
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            chunk_size=config.chunk_size,
            conv_bias=config.conv_bias,
            bias=config.bias,
        )
        self.norm = RMSnorm(d_model)

    def forward(self, x) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


@register_layer("mixer")
class MixerBlock(nn.Module):
    """Mixer блок для mamba"""
    def __init__(self, 
                d_model: int,
                d_state: int,
                d_conv: int,
                expand: int,
                headdim: int, 
                ngroups: int,
                dt_min: float = 0.001,
                dt_max: float = 0.1,
                chunk_size: int = 256,
                conv_bias: bool = True,
                bias: bool = False,
                ):
        super().__init__()
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.ngroups = ngroups
        self.headdim = headdim
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * ngroups * d_state

        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        self.in_proj = nn.Linear(d_model, self.d_inner + self.conv_dim + self.nheads, bias=bias)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=d_conv, groups=self.conv_dim, padding=d_conv - 1, bias=conv_bias)

        self.output_projection = nn.Linear(self.d_inner, d_model, bias=bias)
        self.norm = RMSnorm(self.d_inner)

        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, L, _ = u.shape
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.conv_dim, self.nheads], dim=-1)

        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        x = x.view(b, L, self.nheads, self.headdim)
        B = B.view(b, L, self.ngroups, self.d_state)
        C = C.view(b, L, self.ngroups, self.d_state)

        y = ssm(x, dt, A, B, C, self.D, self.chunk_size).reshape(b, L, self.d_inner)
        y = self.norm(y * F.silu(z))
                
        return self.output_projection(y)


def ssm(
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        chunk_size: int,
) -> torch.Tensor:
    """ Ядро mamba для обучения на CUDA"""
    if x.is_cuda:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        return mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D)

    heads_per_group = x.shape[2] // B.shape[2]
    B = B.repeat_interleave(heads_per_group, dim=2)
    C = C.repeat_interleave(heads_per_group, dim=2)
    return ssm_sequential(x, dt, A, B, C, D)


def ssm_sequential(
            x: torch.Tensor,
            dt: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            C: torch.Tensor,
            D: torch.Tensor,
    ) -> torch.Tensor:
        b, L, H, P = x.shape
        N = B.shape[-1]
        h = x.new_zeros(b, H, P, N)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t] * A)
            h = dA[:, :, None, None] * h + (dt[:, t, :, None] * x[:, t])[..., None] * B[:, t, :, None, :]
            y_t = torch.einsum("bhpn,bhn->bhp", h, C[:, t]) + D[:, None] * x[:, t]

            ys.append(y_t)

        return torch.stack(ys, dim=1)

        
        
        


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