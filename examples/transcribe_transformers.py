#!/usr/bin/env python
"""Mode 2 — ruby transcription with the official `qwen-asr` package (transformers).

The released checkpoint is a standard Qwen3-ASR model (the auxiliary CTC branch
lives in a separate `ctc/` subfolder that this path never touches), so it loads
exactly like the original Qwen3-ASR-1.7B:

    pip install qwen-asr
    python examples/transcribe_transformers.py audio.wav --model <hf-repo-or-local-dir>

The model emits `漢字[かんじ]` bracket furigana inline; `ruby_asr.parse_ruby`
post-processes it into surface / reading / HTML views.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ruby_asr.parse_ruby import brackets_to_parens, to_html, to_reading, to_surface  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", help="audio file (wav/mp3/m4a/... or URL)")
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)

    # No language pin: the ruby furigana is the model's learned behaviour under
    # the default prompt. Result text is the bracket form 漢字[かんじ].
    result = model.transcribe(audio=args.audio)[0]
    native = result.text

    print("model output :", native)                      # 漢字[かんじ] bracket form
    print("paren form   :", brackets_to_parens(native))  # 漢字(かんじ)
    print("surface      :", to_surface(native))          # kanji text only
    print("reading      :", to_reading(native))          # kana only
    print("html         :", to_html(brackets_to_parens(native)))


if __name__ == "__main__":
    main()
