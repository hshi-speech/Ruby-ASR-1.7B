"""CTC tokenizer for kana text <-> mora-vocab ids.

Single source of truth for the CTC tokenisation pipeline:

    reading_kana  ->  hiragana->katakana  ->  unify_long_vowels  ->  mora-segment  ->  ids
                    (the first three steps together are kana_to_morae())

The vocab is a plain `{mora: id}` JSON: id 0 is `<blank>`, ids 1-26 are `a`-`z`
(for embedded English), then Japanese morae by descending corpus frequency.
The released model ships its vocab as `ctc/mora_vocab.json`.
"""

import json
import re

BLANK = "<blank>"


# ---- kana helpers -----------------------------------------------------------

def _hira_to_kata(ch):
    """Single hiragana char -> katakana; everything else passed through."""
    cp = ord(ch)
    if 0x3041 <= cp <= 0x3096:
        return chr(cp + 0x60)
    return ch


def _hira_to_kata_str(s):
    """All hiragana in s converted to katakana; everything else unchanged."""
    return "".join(_hira_to_kata(c) for c in s)


# Long-vowel unification (matches pyopenjtalk g2p(kana=True) convention so the
# morae line up with a vocab built from pyopenjtalk output):
#   O-vowel mora + ウ -> ー   (コウジ -> コージ,  トウキョウ -> トーキョー)
#   U-vowel mora + ウ -> ー   (ジュウ -> ジュー,  スウパー   -> スーパー)
# A palatalised /Cyo/ or /Cyu/ is an I-row base + small ョ or ュ. A+ア and E+イ
# are deliberately left alone (loanwords already write those with ー directly).
_KATA_O_BASE   = "オコゴソゾトドノホボポモヨロヲ"
_KATA_U_BASE   = "ウクグスズツヅヌフブプムユルヴ"
_KATA_PAL_BASE = "キギシジチヂニヒビピミリ"          # palatalises with ョ or ュ
_LONG_O_RE = re.compile(f"([{_KATA_O_BASE}]|[{_KATA_PAL_BASE}]ョ)ウ")
_LONG_U_RE = re.compile(f"([{_KATA_U_BASE}]|[{_KATA_PAL_BASE}]ュ)ウ")


def unify_long_vowels(katakana):
    """Normalise native-spelling long vowels to ー (operates on katakana).

    Hiragana input is passed through unchanged here -- convert via
    `_hira_to_kata_str` first if you want unification to apply.
    """
    katakana = _LONG_O_RE.sub(r"\1ー", katakana)
    katakana = _LONG_U_RE.sub(r"\1ー", katakana)
    return katakana


# Mora segmentation: small kana (ャュョ / ァィゥェォ / ヮ) merge into the
# previous mora; ッ, ン and ー are each their own mora. Non-katakana characters
# (punctuation, ASCII, ...) are dropped.
_MORA_SMALL   = set("ャュョァィゥェォヮ")
_MORA_KATA_RE = re.compile(r"[ァ-ヶー]")


def _segment_morae(katakana):
    morae = []
    for ch in katakana:
        if not _MORA_KATA_RE.match(ch):
            continue
        if ch in _MORA_SMALL and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def kana_to_morae(kana_text):
    """Reading-kana (mixed hira/kata) -> list of mora strings, ready for CTC.

    Pipeline: hira->kata, unify long vowels, mora-segment.

    'ヨコハマエキ、イマもコウジがオワラず…'
      -> ['ヨ','コ','ハ','マ','エ','キ','イ','マ','モ','コ','ー','ジ','ガ','オ','ワ','ラ','ズ', ...]
    """
    return _segment_morae(unify_long_vowels(_hira_to_kata_str(kana_text)))


# ---- tokenizer --------------------------------------------------------------

class CTCTokenizer:
    """Mora-level CTC tokenizer.

    Construct from a vocab JSON path or an existing `{mora: id}` mapping.

    Useful attributes:
        vocab       {mora: id}
        id_to_mora  inverse mapping
        blank_id    id of '<blank>' (typically 0)
        vocab_size  len(vocab)
    """

    def __init__(self, vocab):
        if isinstance(vocab, str):
            with open(vocab, encoding="utf-8") as f:
                self.vocab = json.load(f)
        elif isinstance(vocab, dict):
            self.vocab = dict(vocab)
        else:
            raise TypeError("vocab must be a path or a {mora: id} dict")
        if BLANK not in self.vocab:
            raise ValueError(f"vocab is missing the '{BLANK}' entry")
        self.id_to_mora = {i: m for m, i in self.vocab.items()}
        self.blank_id = self.vocab[BLANK]
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.vocab_size

    # ---- text -> tokens / ids ---------------------------------------------
    def tokenize(self, kana_text):
        """Reading-kana -> list of mora strings (no vocab lookup)."""
        return kana_to_morae(kana_text)

    def encode_morae(self, morae):
        """List of mora strings -> list of ids. OOV morae are silently dropped."""
        return [self.vocab[m] for m in morae if m in self.vocab]

    def encode(self, kana_text):
        """Reading-kana -> list of CTC mora ids. = tokenize + encode_morae."""
        return self.encode_morae(self.tokenize(kana_text))

    # ---- ids -> tokens / text ---------------------------------------------
    def decode(self, ids, *, collapse_repeats=False, drop_blank=False, join=""):
        """Ids -> string. For greedy CTC decoding, set
        `collapse_repeats=True` and `drop_blank=True`. Unknown ids render as '?'.
        """
        out = []
        prev = object()
        for i in ids:
            i = int(i)
            if collapse_repeats and i == prev:
                continue
            prev = i
            if drop_blank and i == self.blank_id:
                continue
            out.append(self.id_to_mora.get(i, "?"))
        return join.join(out)
