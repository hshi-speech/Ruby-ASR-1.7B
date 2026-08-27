"""Parse ruby-bracket ASR output into (base, reading) pairs for display.

The fine-tuned model emits compact bracket furigana: the kanji surface sits
OUTSIDE the brackets and the kana reading INSIDE, e.g.

    発売[はつばい]された日[ひ]にか。

`brackets_to_parens` normalises this to the paren form 発売(はつばい)…; the
parser accepts BOTH bracket styles (ASCII and full-width):
``base[reading]`` / ``base(reading)`` / ``base（reading）`` / ``base［reading］``.

The base of each reading is the maximal kanji-containing run immediately
before the bracket (leading plain kana stays plain text; the surface starts at
the first kanji). Text without annotations passes through unchanged as plain
segments.
"""
from __future__ import annotations

import html
import re

# CJK ideograph ranges + 々 iteration mark + CJK compatibility ideographs.
_KANJI = r"一-鿿㐀-䶿々豈-﫿"

# One annotated group: (leading plain text)(base starting at a kanji)(reading).
# Bracket chars are excluded from base/reading so groups never nest or overlap.
_OPEN = r"\(\[（［"
_CLOSE = r"\)\]）］"
# A ruby base may start at a kanji OR a digit / Latin letter (half- and
# full-width). The model annotates numbers and acronyms too (`2020[にせんにじゅう]`,
# `5[ご]`, `AI[エーアイ]`); without digits/letters here those never matched a ruby
# span, so their reading leaked out as literal `数字(よみ)` parens.
_BASE_START = _KANJI + r"0-9０-９A-Za-zＡ-Ｚａ-ｚ"

_GROUP_RE = re.compile(
    r"([^" + _OPEN + _CLOSE + r"]*?)"          # 1: plain prefix (kana/latin/punct)
    r"([" + _BASE_START + r"][^" + _OPEN + _CLOSE + r"]*?)"  # 2: base (kanji/digit/latin start)
    r"[" + _OPEN + r"]([^" + _OPEN + _CLOSE + r"]*)[" + _CLOSE + r"]"  # 3: reading
)


def parse_ruby(text: str) -> list[dict]:
    """Split bracket-annotated text into segments.

    Returns a list of ``{"type": "ruby", "base": ..., "reading": ...}`` and
    ``{"type": "text", "text": ...}`` dicts, in order. Brackets that do not
    follow a kanji base (e.g. ordinary parentheses around plain kana) are kept
    verbatim as plain text.
    """
    segments: list[dict] = []

    def _text(s: str) -> None:
        if s:
            # merge with a preceding plain segment to keep output canonical
            if segments and segments[-1]["type"] == "text":
                segments[-1]["text"] += s
            else:
                segments.append({"type": "text", "text": s})

    pos = 0
    for m in _GROUP_RE.finditer(text):
        _text(text[pos:m.start()])
        _text(m.group(1))
        segments.append({"type": "ruby", "base": m.group(2), "reading": m.group(3)})
        pos = m.end()
    _text(text[pos:])
    return segments


def to_html(text: str) -> str:
    """Render bracket-annotated text as HTML ``<ruby>`` markup.

    ``漢字(かんじ)を読む`` -> ``<ruby><rb>漢字</rb><rt>かんじ</rt></ruby>を読む``
    All content is HTML-escaped.
    """
    out = []
    for seg in parse_ruby(text):
        if seg["type"] == "ruby":
            out.append(
                f"<ruby><rb>{html.escape(seg['base'])}</rb>"
                f"<rt>{html.escape(seg['reading'])}</rt></ruby>"
            )
        else:
            out.append(html.escape(seg["text"]))
    return "".join(out)


_SQ_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


def brackets_to_parens(text: str) -> str:
    """Model-native ``base[reading]`` -> the paren form ``base(reading)``.

    An EMPTY reading (``base[]`` — the model marks a word boundary but assigns no
    reading, common on numbers) is dropped entirely rather than left as an empty
    ``()``: ``10[]年`` -> ``10年``."""
    return _SQ_BRACKET_RE.sub(
        lambda m: f"({m.group(1)})" if m.group(1).strip() else "", text)


def to_surface(text: str) -> str:
    """Annotated text -> kanji surface (readings dropped)."""
    return "".join(
        seg["base"] if seg["type"] == "ruby" else seg["text"]
        for seg in parse_ruby(text)
    )


def to_reading(text: str) -> str:
    """Annotated text -> reading (each kanji word replaced by its kana; plain
    runs kept). Empty readings fall back to the surface."""
    return "".join(
        (seg["reading"] or seg["base"]) if seg["type"] == "ruby" else seg["text"]
        for seg in parse_ruby(text)
    )
