"""Unit tests for the mora CTC tokenizer (`pytest tests/test_ctc_tokenizer.py`)."""
from ruby_asr.ctc_tokenizer import BLANK, CTCTokenizer, kana_to_morae, unify_long_vowels


def test_hira_converted_and_segmented():
    assert kana_to_morae("かんじ") == ["カ", "ン", "ジ"]


def test_small_kana_merge_into_previous_mora():
    assert kana_to_morae("キョウト") == ["キョ", "ー", "ト"]  # ョ merges, オ段+ウ -> ー


def test_sokuon_and_chouon_stand_alone():
    assert kana_to_morae("ガッコウ") == ["ガ", "ッ", "コ", "ー"]


def test_long_vowel_unification():
    assert unify_long_vowels("トウキョウ") == "トーキョー"
    assert unify_long_vowels("ジュウ") == "ジュー"
    # A+ア / E+イ are deliberately left alone
    assert unify_long_vowels("ケイ") == "ケイ"


def test_non_kana_dropped():
    assert kana_to_morae("エーアイ、AIです。") == ["エ", "ー", "ア", "イ", "デ", "ス"]


def _tok():
    vocab = {BLANK: 0, "ナ": 1, "ネ": 2, "ン": 3, "カ": 4, "デ": 5}
    return CTCTokenizer(vocab)


def test_encode_decode_roundtrip():
    tok = _tok()
    ids = tok.encode("ななねんかんで")
    assert ids == [1, 1, 2, 3, 4, 3, 5]
    assert tok.decode(ids) == "ナナネンカンデ"


def test_greedy_ctc_collapse():
    tok = _tok()
    # blanks separate repeats; repeats collapse; blanks drop
    ids = [0, 1, 1, 0, 1, 3, 3, 0, 0, 5]
    assert tok.decode(ids, collapse_repeats=True, drop_blank=True) == "ナナンデ"


def test_oov_morae_silently_dropped():
    tok = _tok()
    assert tok.encode("なぞ") == [1]  # ゾ not in vocab
