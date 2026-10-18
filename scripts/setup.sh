#!/usr/bin/env bash
# Per-project setup. Запускается ОДИН раз после git clone (Mac / Linux).
#
#   uv (если нет) -> Python 3.10 -> uv sync из uv.lock -> conf/local -> проверка kedro
#
# Дальше day-to-day: `uv run kedro run --pipeline=...`
set -euo pipefail

PROJECT_NAME="mamba-vs-transformer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[setup] Project: $PROJECT_NAME | Root: $PROJECT_ROOT | Platform: $(uname -s)"

# 1. uv: установить, если нет.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found, installing via astral.sh installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || { echo "[setup] ERROR: uv не в PATH; перезапустите shell." >&2; exit 1; }
fi
echo "[setup] uv: $(uv --version)"

# 2. Python 3.10 через uv (без sudo); версия зафиксирована в .python-version.
uv python find 3.10 >/dev/null 2>&1 || uv python install 3.10

# 3. Зависимости из uv.lock. --frozen: lock не переписывается, ставится ровно то, что закоммичено.
echo "[setup] uv sync..."
if [ -f uv.lock ]; then
    uv sync --frozen
else
    uv sync
fi

# 3b. Linux (сервер): CUDA-ядра Mamba ставятся готовыми wheel'ами (cu12 + torch2.6, x86_64).
#     Нужен только драйвер NVIDIA с поддержкой CUDA 12; nvcc/toolkit не нужен.
if [ "$(uname -s)" = "Linux" ]; then
    command -v nvidia-smi >/dev/null 2>&1 \
        || echo "[setup] WARN: nvidia-smi не найден — mamba-ssm без GPU работать не будет" >&2
    uv run python -c "
import torch, mamba_ssm, causal_conv1d
print(f'[setup] torch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.is_available()}')
print(f'[setup] mamba_ssm {mamba_ssm.__version__} | causal_conv1d {causal_conv1d.__version__}')
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # грузит CUDA-extension
" || echo "[setup] WARN: mamba_ssm/causal_conv1d не импортируются (несовпадение CUDA/glibc/архитектуры?)" >&2
fi

# 4. conf/local — локальные overrides/секреты kedro (в git не попадают).
mkdir -p conf/local

# 5. Diagnostic: Kedro видит проект и пайплайны.
echo "[setup] Kedro pipelines:"
uv run kedro registry list 2>/dev/null | grep -E '^- ' || echo "[setup] WARN: kedro registry list не отработал" >&2

# 6. Путь к WikiText: без него data_processing упадёт.
_wikitext_path="$(uv run python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('conf/base/source.yml').data.wikitext.path or '')
" 2>/dev/null || true)"
if [ -z "$_wikitext_path" ]; then
    echo "[setup] WARN: data.wikitext.path не задан в conf/base/source.yml" >&2
    echo "[setup]       укажите директорию с train*/validation*/test* parquet-файлами WikiText-103" >&2
elif [ ! -d "$_wikitext_path" ]; then
    echo "[setup] WARN: директория WikiText не найдена: $_wikitext_path" >&2
fi
unset _wikitext_path

# 7. Pre-commit hook (если конфиг есть).
if [ -f .pre-commit-config.yaml ] && uv run python -c "import pre_commit" >/dev/null 2>&1; then
    uv run pre-commit install && echo "[setup] pre-commit hook installed"
fi

echo
echo "[setup] Done. Day-to-day:"
echo "  uv run kedro run --pipeline=data_processing   # WikiText -> dataset.parquet, dataset_split.parquet"
echo "  uv run kedro run --env mamba                  # полный пайплайн для Mamba"
echo "  uv run kedro run --env transformer            # полный пайплайн для Transformer"
echo "  uv run kedro run --env mamba --pipeline=training   # только обучение"
