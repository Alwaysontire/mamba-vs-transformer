"""Строительные блоки HTML-отчёта: секция, таблица, стат-плитки, график plotly"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class Section:
    """Секция отчёта: заголовок + готовый HTML тела"""

    title: str
    body: str
    note: str | None = None

    def render(self) -> str:
        note = f'<p class="note">{escape(self.note)}</p>' if self.note else ""
        return f"<section><h2>{escape(self.title)}</h2>{note}{self.body}</section>"


@dataclass(frozen=True)
class Cell:
    """Ячейка таблицы: текст + флаги оформления"""

    text: str
    numeric: bool = False
    best: bool = False


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    """Число для таблицы: тысячи через тонкий пробел, ``—`` для пропуска"""
    if value is None:
        return "—"
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 1000:
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.{digits}f}".replace(",", " ")


def fmt_params(value: int | None) -> str:
    """Число параметров в миллионах"""
    return "—" if value is None else f"{value / 1e6:.2f}M"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}ч {m:02d}м" if h else f"{m}м {s:02d}с"


def table(header: Sequence[str], rows: Sequence[Sequence[Cell | str]], *, row_marks: Sequence[bool] | None = None) -> str:
    """HTML-таблица; ``best`` — жирный + «▲ лучше», ``row_marks`` — подсветка строки"""

    def td(c: Cell | str) -> str:
        c = c if isinstance(c, Cell) else Cell(str(c))
        cls = " ".join(x for x in ("num" if c.numeric else "", "best" if c.best else "") if x)
        mark = ' <span class="best-mark" title="лучшее значение">▲</span>' if c.best else ""
        return f'<td class="{cls}">{escape(c.text)}{mark}</td>'

    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="diff"' if row_marks and row_marks[i] else ""
        body.append(f"<tr{cls}>{''.join(td(c) for c in row)}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def stat_tiles(items: Sequence[tuple[str, str, str | None]]) -> str:
    """Ряд стат-плиток: (подпись, значение, подстрочник)"""
    tiles = []
    for label, value, sub in items:
        sub_html = f'<div class="tile-sub">{escape(sub)}</div>' if sub else ""
        tiles.append(
            f'<div class="tile"><div class="tile-label">{escape(label)}</div>'
            f'<div class="tile-value">{escape(value)}</div>{sub_html}</div>'
        )
    return f'<div class="tiles">{"".join(tiles)}</div>'


def line_chart(
    series: Sequence[dict[str, Any]],
    *,
    x_title: str,
    y_title: str,
    chart_id: str,
    log_y: bool = False,
) -> str:
    """Линейный график plotly; цвета/тема применяются на клиенте из CSS-токенов

    Args:
        series: ``[{"name": ..., "slot": 1, "x": [...], "y": [...]}]`` —
            ``slot`` — номер категориального цвета (закреплён за запуском).
        log_y: Логарифмическая ось Y (для loss: виден хвост, а не только первые шаги).
    """
    spec = {"series": list(series), "x_title": x_title, "y_title": y_title, "log_y": log_y}
    payload = escape(json.dumps(spec, ensure_ascii=False, allow_nan=False), quote=True)
    return f'<div class="chart" id="{escape(chart_id)}" data-spec="{payload}"></div>'
