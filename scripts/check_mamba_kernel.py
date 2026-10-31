"""Сервер: ядро mamba_chunk_scan_combined == эталон ssm_sequential, и скорость шага Mamba vs Transformer

    uv run python scripts/check_mamba_kernel.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.schemas import ModelConfig
from src.training.layers import ssm, ssm_sequential
from src.training.model import build_model


def check_ssm(device: torch.device) -> None:
    torch.manual_seed(0)
    b, L, H, P, G, N, chunk = 2, 96, 4, 16, 2, 8, 32
    x = torch.randn(b, L, H, P, device=device, requires_grad=True)
    dt = torch.nn.functional.softplus(torch.randn(b, L, H, device=device) - 2).requires_grad_()
    A = (-torch.rand(H, device=device) * 4 - 0.5).requires_grad_()
    B = torch.randn(b, L, G, N, device=device, requires_grad=True)
    C = torch.randn(b, L, G, N, device=device, requires_grad=True)
    D = torch.randn(H, device=device, requires_grad=True)
    inputs = (x, dt, A, B, C, D)

    y_kernel = ssm(*inputs, chunk)
    grads_kernel = torch.autograd.grad(y_kernel.square().sum(), inputs)

    rep = H // G
    y_ref = ssm_sequential(x, dt, A, B.repeat_interleave(rep, dim=2), C.repeat_interleave(rep, dim=2), D)
    grads_ref = torch.autograd.grad(y_ref.square().sum(), inputs)

    print(f"forward  max|Δ| = {(y_kernel - y_ref).abs().max().item():.2e}")
    for name, gk, gr in zip("x dt A B C D".split(), grads_kernel, grads_ref):
        rel = (gk - gr).norm() / gr.norm()
        print(f"grad {name:<2} rel = {rel.item():.2e}")


def step_time(env: str, device: torch.device, batch: int = 8, steps: int = 10) -> None:
    cfg = TypeAdapter(ModelConfig).validate_python(yaml.safe_load(open(f"conf/{env}/parameters.yml"))["model"])
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (batch, cfg.max_length + 1), device=device)

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    sec = (time.perf_counter() - start) / steps
    tokens = batch * cfg.max_length
    print(f"{env:<12} {sec * 1000:7.1f} ms/step  {tokens / sec:9.0f} tok/s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("Нужна CUDA")
    device = torch.device("cuda")
    print("== ssm: ядро vs эталон (fp32)")
    check_ssm(device)
    print("== шаг обучения, batch 8 x max_length, bf16")
    step_time("transformer", device)
    step_time("mamba", device)
