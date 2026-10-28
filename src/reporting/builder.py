"""Сборка HTML-отчёта по одному или нескольким запускам (env mamba / transformer)

build_report(runs) -> str: одна функция и для отчёта по запуску, и для сравнения —
сравнение это просто отчёт по N запускам.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from html import escape
from typing import Any, Literal

from src.reporting.inputs import RunResult
from src.reporting.section import (
    Cell,
    Section,
    fmt_duration,
    fmt_num,
    fmt_params,
    line_chart,
    stat_tiles,
    table,
)

PlotlyMode = Literal["cdn", "inline"]

PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
MAX_SLOTS = 8   

_KNOWN_METRIC_KEYS = {
    "split", "token_level_perplexity", "word_level_perplexity", "loss", "n_tokens", "n_words",
    "context_length", "stride", "throughput", "loss_by_position",
}


def _best_flags(values: Sequence[float | None], better: Literal["min", "max"]) -> list[bool]:
    """Флаги лучшего значения; при одном запуске сравнивать не с чем"""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [False] * len(values)
    target = min(present) if better == "min" else max(present)
    return [v is not None and v == target for v in values]


def _metric_column(
    runs: Sequence[RunResult],
    get: Callable[[RunResult], float | int | None],
    fmt: Callable[[Any], str],
    better: Literal["min", "max"] | None,
) -> list[Cell]:
    values = [get(r) for r in runs]
    flags = _best_flags(values, better) if better else [False] * len(values)
    return [Cell(fmt(v), numeric=True, best=f) for v, f in zip(values, flags)]


def _summary(runs: Sequence[RunResult]) -> Section:
    def tp(r: RunResult, key: str) -> float | None:
        return getattr(r.metrics.throughput, key) if r.metrics.throughput else None

    def params(r: RunResult, key: str) -> int | None:
        return getattr(r.manifest.n_params, key) if r.manifest.n_params else None

    columns: list[tuple[str, list[Cell]]] = [
        ("Token PPL", _metric_column(runs, lambda r: r.metrics.token_level_perplexity, fmt_num, "min")),
        ("Word PPL", _metric_column(runs, lambda r: r.metrics.word_level_perplexity, fmt_num, "min")),
        ("Loss, nats", _metric_column(runs, lambda r: r.metrics.loss, lambda v: fmt_num(v, 3), "min")),
        ("Параметры без эмб.", _metric_column(runs, lambda r: params(r, "non_embedding"), fmt_params, None)),
        ("Параметры всего", _metric_column(runs, lambda r: params(r, "total"), fmt_params, None)),
        ("Время обучения", _metric_column(runs, lambda r: r.manifest.train_time_sec, fmt_duration, "min")),
        ("Токенов/с", _metric_column(runs, lambda r: tp(r, "tokens_per_sec"), lambda v: fmt_num(v, 0), "max")),
        ("Пик памяти, МБ", _metric_column(runs, lambda r: tp(r, "peak_memory_mb"), lambda v: fmt_num(v, 0), "min")),
    ]
    # Колонки, где нет ни одного значения, не показываем
    columns = [(h, cells) for h, cells in columns if any(c.text != "—" for c in cells)]

    header = ["Запуск", "Архитектура", *(h for h, _ in columns)]
    rows = [[r.name, r.arch, *(cells[i] for _, cells in columns)] for i, r in enumerate(runs)]

    body = table(header, rows)
    if len(runs) == 1:
        r = runs[0]
        tiles = [("Token PPL", fmt_num(r.metrics.token_level_perplexity), r.metrics.split)]
        if r.metrics.word_level_perplexity is not None:
            tiles.append(("Word PPL", fmt_num(r.metrics.word_level_perplexity), r.metrics.split))
        if r.manifest.n_params and r.manifest.n_params.non_embedding is not None:
            tiles.append(("Параметры без эмб.", fmt_params(r.manifest.n_params.non_embedding), None))
        if r.manifest.train_time_sec is not None:
            tiles.append(("Время обучения", fmt_duration(r.manifest.train_time_sec), None))
        body = stat_tiles(tiles) + body

    note = "▲ — лучшее значение среди запусков" if len(runs) > 1 else None
    return Section("Сводка", body, note=note)


def _data_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Таблица данных графика под спойлером (доступность + точные значения)"""
    cells = [[Cell(fmt_num(v, 4) if isinstance(v, float) else str(v), numeric=isinstance(v, (int, float))) for v in row] for row in rows]
    return f"<details><summary>Таблица значений</summary>{table(header, cells)}</details>"


def _history_chart(runs: Sequence[RunResult], key: str, title: str, chart_id: str) -> str | None:
    series, rows = [], []
    for slot, r in enumerate(runs, start=1):
        points = [(h.step, getattr(h, key)) for h in r.manifest.history if getattr(h, key) is not None]
        if not points:
            continue
        series.append({"name": r.name, "slot": slot, "x": [p[0] for p in points], "y": [p[1] for p in points]})
        rows.extend((r.name, s, v) for s, v in points)
    if not series:
        return None
    chart = line_chart(series, x_title="Шаг", y_title=f"{title} (log)", chart_id=chart_id, log_y=True)
    return f'<figure><figcaption>{escape(title)}</figcaption>{chart}{_data_table(["Запуск", "Шаг", title], rows)}</figure>'


def _learning_curves(runs: Sequence[RunResult]) -> Section | None:
    # Train и eval loss — отдельными графиками: на одном цвет кодирует запуск, а не вид loss
    charts = [
        _history_chart(runs, "train_loss", "Train loss", "chart-train-loss"),
        _history_chart(runs, "eval_loss", "Validation loss", "chart-eval-loss"),
    ]
    charts = [c for c in charts if c]
    if not charts:
        return None
    return Section("Кривые обучения", f'<div class="grid-2">{"".join(charts)}</div>')


def _loss_by_position(runs: Sequence[RunResult]) -> Section | None:
    series, rows = [], []
    for slot, r in enumerate(runs, start=1):
        points = r.metrics.loss_by_position or []
        if not points:
            continue
        series.append({"name": r.name, "slot": slot, "x": [p.position for p in points], "y": [p.loss for p in points]})
        rows.extend((r.name, p.position, p.loss) for p in points)
    if not series:
        return None
    chart = line_chart(series, x_title="Позиция токена в контексте", y_title="Loss, nats", chart_id="chart-loss-pos")
    note = "Как падает loss по мере роста доступного контекста: здесь видно, насколько модель использует дальний контекст."
    return Section(
        "Loss по позиции в контексте",
        f"<figure>{chart}{_data_table(['Запуск', 'Позиция', 'Loss'], rows)}</figure>",
        note=note,
    )


def _eval_details(runs: Sequence[RunResult]) -> Section:
    fields: list[tuple[str, Callable[[RunResult], Any]]] = [
        ("Сплит", lambda r: r.metrics.split),
        ("Токенов", lambda r: r.metrics.n_tokens),
        ("Слов", lambda r: r.metrics.n_words),
        ("Длина контекста", lambda r: r.metrics.context_length),
        ("Stride", lambda r: r.metrics.stride),
    ]
    extra_keys = sorted({k for r in runs for k in (r.metrics.model_extra or {}) if k not in _KNOWN_METRIC_KEYS})
    for k in extra_keys:
        fields.append((k, lambda r, k=k: (r.metrics.model_extra or {}).get(k)))

    rows, marks = [], []
    for label, get in fields:
        values = [get(r) for r in runs]
        if all(v is None for v in values):
            continue
        rows.append([label, *(fmt_num(v) if isinstance(v, (int, float)) else ("—" if v is None else str(v)) for v in values)])
        marks.append(len(runs) > 1 and len({repr(v) for v in values}) > 1)
    if not rows:
        return Section("Параметры оценки", '<p class="note">Нет данных</p>')
    return Section("Параметры оценки", table(["", *(r.name for r in runs)], rows, row_marks=marks))


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _config(runs: Sequence[RunResult]) -> Section | None:
    flat = [_flatten({"model": r.manifest.model, "training": r.manifest.training}) for r in runs]
    keys = sorted({k for f in flat for k in f})
    if not keys:
        return None
    rows, marks = [], []
    for k in keys:
        values = [f.get(k) for f in flat]
        rows.append([k, *("—" if v is None else str(v) for v in values)])
        marks.append(len(runs) > 1 and len({repr(v) for v in values}) > 1)
    note = "Подсвечены параметры, которые различаются между запусками" if len(runs) > 1 else None
    return Section("Конфигурация", table(["Параметр", *(r.name for r in runs)], rows, row_marks=marks), note=note)


def build_report(
    runs: Sequence[RunResult],
    *,
    title: str | None = None,
    plotly: PlotlyMode = "cdn",
) -> str:
    """Собрать самодостаточный HTML-отчёт

    Args:
        runs: Запуски; цвет закреплён за позицией запуска, поэтому передавайте
            их в стабильном порядке (например, отсортированными по имени).
        title: Заголовок; по умолчанию — по именам запусков.
        plotly: ``"cdn"`` — plotly.js с jsDelivr (лёгкий файл, нужен интернет при
            просмотре); ``"inline"`` — встроить (~4.5 МБ, открывается офлайн).
    """
    if not runs:
        raise ValueError("Нет запусков для отчёта")
    if len(runs) > MAX_SLOTS:
        raise ValueError(f"Не больше {MAX_SLOTS} запусков в одном отчёте, передано {len(runs)}")

    title = title or ("Сравнение: " + " vs ".join(r.name for r in runs) if len(runs) > 1 else f"Отчёт: {runs[0].name}")
    sections = [s for s in (_summary(runs), _learning_curves(runs), _loss_by_position(runs), _eval_details(runs), _config(runs)) if s]
    body = "".join(s.render() for s in sections)

    has_charts = 'class="chart"' in body
    if not has_charts:
        plotly_tag = ""
    elif plotly == "inline":
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _TEMPLATE.format(
        title=escape(title),
        generated=escape(generated),
        body=body,
        plotly_tag=plotly_tag,
        chart_js=_CHART_JS if has_charts else "",
    )

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --diff: rgba(42,120,214,0.08);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --diff: rgba(57,135,229,0.14);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  --diff: rgba(57,135,229,0.14);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }
header h1 { font-size: 26px; line-height: 1.25; margin: 0 0 4px; font-weight: 650; }
header p { margin: 0; color: var(--muted); font-size: 13px; }
section {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; margin-top: 20px;
}
h2 { font-size: 17px; margin: 0 0 12px; font-weight: 600; }
.note { color: var(--text-secondary); font-size: 13px; margin: -6px 0 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile-label { color: var(--text-secondary); font-size: 13px; }
.tile-value { font-size: 28px; font-weight: 650; line-height: 1.2; margin-top: 2px; }
.tile-sub { color: var(--muted); font-size: 12px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 500; font-size: 13px; white-space: normal; vertical-align: bottom; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { font-weight: 650; }
.best-mark { color: var(--text-secondary); font-size: 10px; vertical-align: 1px; }
tr.diff td { background: var(--diff); }
figure { margin: 0; min-width: 0; }
figcaption { color: var(--text-secondary); font-size: 13px; margin-bottom: 4px; }
.chart { width: 100%; height: 320px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
details { margin-top: 8px; font-size: 13px; }
summary { cursor: pointer; color: var(--text-secondary); }
details table { margin-top: 8px; }
"""

_CHART_JS = """
<script>
(function () {
  if (!window.Plotly) {
    document.querySelectorAll('.chart').forEach(function (el) {
      el.innerHTML = '<p class="note">plotly.js не загрузился (нет интернета?) — данные в таблице ниже</p>';
      el.style.height = 'auto';
    });
    return;
  }
  function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function render(el) {
    var spec = JSON.parse(el.dataset.spec);
    var traces = spec.series.map(function (s) {
      return {
        type: 'scatter', mode: s.x.length > 1 ? 'lines' : 'markers', name: s.name, x: s.x, y: s.y,
        line: { width: 2, color: token('--series-' + s.slot), shape: 'linear' },
        marker: { size: 8, color: token('--series-' + s.slot), line: { width: 2, color: token('--surface-1') } },
        hovertemplate: '%{y:.4f}<extra>' + s.name + '</extra>'
      };
    });
    var axis = function (title) {
      return {
        title: { text: title, font: { size: 12, color: token('--text-secondary') } },
        gridcolor: token('--grid'), linecolor: token('--baseline'), zeroline: false,
        tickfont: { size: 11, color: token('--muted') }, automargin: true
      };
    };
    var layout = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: token('--text-primary') },
      margin: { l: 8, r: 8, t: 8, b: 8 },
      xaxis: Object.assign(axis(spec.x_title), { showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: token('--baseline'), spikedash: 'solid' }),
      yaxis: Object.assign(axis(spec.y_title), spec.log_y ? { type: 'log' } : {}),
      hovermode: 'x unified',
      hoverlabel: { bgcolor: token('--surface-1'), bordercolor: token('--baseline'), font: { color: token('--text-primary') } },
      showlegend: traces.length > 1,
      legend: { orientation: 'h', x: 0, y: 1.08, font: { color: token('--text-secondary') } }
    };
    Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
  }
  function renderAll() { document.querySelectorAll('.chart').forEach(render); }
  renderAll();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
  new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
</script>
"""

_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>""" + _CSS.replace("{", "{{").replace("}", "}}") + """</style>
{plotly_tag}
</head>
<body>
<main>
<header><h1>{title}</h1><p>Сгенерирован {generated}</p></header>
{body}
</main>
{chart_js}
</body>
</html>
"""
