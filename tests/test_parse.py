"""Unit tests for the ruby-bracket parser (`pytest tests/test_parse.py`)."""
from ruby_asr.parse_ruby import brackets_to_parens, parse_ruby, to_html


def test_single_pair_plus_plain_text():
    segs = parse_ruby("漢字(かんじ)を読む")
    assert segs == [
        {"type": "ruby", "base": "漢字", "reading": "かんじ"},
        {"type": "text", "text": "を読む"},
    ]


def test_plain_text_passes_through():
    assert parse_ruby("こんにちは、世界。") == [
        {"type": "text", "text": "こんにちは、世界。"}
    ]


def test_empty_string():
    assert parse_ruby("") == []


def test_multiple_pairs_with_okurigana():
    # leading kana stays plain; base starts at the first kanji (発売, 日)
    segs = parse_ruby("しかもゲームが発売(はつばい)された日(ひ)にか。")
    assert segs == [
        {"type": "text", "text": "しかもゲームが"},
        {"type": "ruby", "base": "発売", "reading": "はつばい"},
        {"type": "text", "text": "された"},
        {"type": "ruby", "base": "日", "reading": "ひ"},
        {"type": "text", "text": "にか。"},
    ]


def test_square_bracket_model_native_format():
    segs = parse_ruby("発売[はつばい]された")
    assert segs == [
        {"type": "ruby", "base": "発売", "reading": "はつばい"},
        {"type": "text", "text": "された"},
    ]


def test_fullwidth_brackets():
    segs = parse_ruby("漢字（かんじ）")
    assert segs == [{"type": "ruby", "base": "漢字", "reading": "かんじ"}]


def test_parens_without_kanji_base_stay_plain():
    # ordinary parentheses around kana, no kanji base -> not a ruby pair
    assert parse_ruby("はい(あらすじ)ですか") == [
        {"type": "text", "text": "はい(あらすじ)ですか"}
    ]


def test_to_html():
    assert to_html("漢字(かんじ)を読む") == (
        "<ruby><rb>漢字</rb><rt>かんじ</rt></ruby>を読む"
    )


def test_to_html_escapes():
    assert to_html("<b>") == "&lt;b&gt;"


def test_brackets_to_parens():
    assert brackets_to_parens("発売[はつばい]された日[ひ]に") == \
        "発売(はつばい)された日(ひ)に"
