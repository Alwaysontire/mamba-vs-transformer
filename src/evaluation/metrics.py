"""Метрики оценки"""


from __future__ import annotations

import math

UNIFIED_METRICS: tuple[str, ...] = (
    "token_level_perplexity",
    "word_level_perplexity",
)


def compute_metrics(nll_sum: float, n_tokens: int, n_words: int | None = None) -> dict[str, float | None]:
    """
    Args:
        nll_sum (float): Суммарный NLL (в натах) по засчитанным токенам.
        n_tokens (int): Число засчитанных токенов.
        n_words (int | None): Число слов исходного текста сплита.

    """
    if n_tokens <= 0:
        raise ValueError(f"Нет засчитанных токенов: n_tokens={n_tokens}")

    loss = nll_sum / n_tokens
    return {
        "loss": loss,
        "token_level_perplexity": math.exp(loss),
        "word_level_perplexity": math.exp(nll_sum / n_words) if n_words else None,
    }
