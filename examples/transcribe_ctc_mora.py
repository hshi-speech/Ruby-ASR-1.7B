#!/usr/bin/env python
"""Mode 1 — encoder + CTC mora-reading recognition (no LLM decoding).

    python examples/transcribe_ctc_mora.py audio.wav --model <hf-repo-or-local-dir>

Prints one actual-pronunciation kana string per input file, e.g.

    audio.wav	ナナネンカンデロスニモトモダチデキタシ

For a raw joint-CTC training checkpoint (CTC tensors still embedded in
model.safetensors) also pass --mora-vocab <mora_vocab.json>.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ruby_asr import MoraCTCRecognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", nargs="+", help="audio file(s) (wav/flac/mp3/m4a/...)")
    ap.add_argument("--model", required=True, help="HF repo id or local model dir")
    ap.add_argument("--subfolder", default=None,
                    help='model subfolder inside the repo, e.g. "subtitle" or "verbatim"')
    ap.add_argument("--mora-vocab", default=None,
                    help="mora vocab JSON (only for raw training checkpoints)")
    ap.add_argument("--device", default=None, help='e.g. "cuda:0"; default: auto')
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    rec = MoraCTCRecognizer.from_pretrained(
        args.model, device=args.device, mora_vocab=args.mora_vocab,
        subfolder=args.subfolder)
    for path, mora in zip(args.audio,
                          rec.transcribe(args.audio, batch_size=args.batch_size)):
        print(f"{path}\t{mora}")


if __name__ == "__main__":
    main()
