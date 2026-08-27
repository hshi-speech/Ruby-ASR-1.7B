"""ruby-asr: Japanese ASR with inline furigana (ruby) output and an auxiliary
encoder-CTC mora-reading head, built on Qwen3-ASR.

Two inference modes:

    Mode 1 — encoder + CTC -> actual-pronunciation mora reading (kana):
        from ruby_asr import MoraCTCRecognizer
        rec = MoraCTCRecognizer.from_pretrained("<hf-repo>")
        rec.transcribe("audio.wav")

    Mode 2 — standard Qwen3-ASR decode -> transcript with 漢字[かんじ] furigana
        (works with the official qwen-asr package and stock vLLM; see examples/).
"""
from .ctc_tokenizer import BLANK, CTCTokenizer, kana_to_morae
from .parse_ruby import (brackets_to_parens, parse_ruby, to_html, to_reading,
                         to_surface)

__version__ = "0.1.0"

__all__ = [
    "MoraCTCRecognizer", "CTCBranch", "CTCTokenizer", "kana_to_morae", "BLANK",
    "parse_ruby", "brackets_to_parens", "to_surface", "to_reading", "to_html",
]

_LAZY = {"MoraCTCRecognizer", "CTCBranch"}


def __getattr__(name):
    # torch-dependent classes load on first use, so the pure-text helpers
    # (parse_ruby etc.) stay importable in client-only environments.
    if name in _LAZY:
        from . import ctc_model
        return getattr(ctc_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
