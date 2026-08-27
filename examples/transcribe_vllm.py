#!/usr/bin/env python
"""Mode 2 — ruby transcription through a vLLM OpenAI-compatible server.

Start the server first (the released checkpoint is stock-vLLM loadable, no
custom code needed):

    vllm serve <hf-repo-or-local-dir> \
        --served-model-name ruby-asr \
        --max-model-len 8192

Then:

    python examples/transcribe_vllm.py audio.wav --base-url http://127.0.0.1:8000/v1

IMPORTANT — use /v1/chat/completions, NOT /v1/audio/transcriptions. The ruby
furigana is the model's learned behaviour under the standard Qwen3-ASR chat
prompt (empty system + audio); there is no textual "add furigana" instruction.
vLLM's /v1/audio/transcriptions endpoint builds a different internal prompt and
returns plain text with no brackets.

Requires: pip install openai   (plus ffmpeg on PATH for non-wav input)
"""
import argparse
import base64
import os
import subprocess
import sys
import tempfile

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ruby_asr.parse_ruby import brackets_to_parens, to_html, to_reading, to_surface  # noqa: E402

ASR_TEXT_TAG = "<asr_text>"


def strip_asr_prefix(raw: str) -> str:
    """Qwen3-ASR raw output -> transcript, dropping the `language X<asr_text>`
    meta prefix. `language None<asr_text>` with no text means empty audio."""
    s = (raw or "").strip()
    if ASR_TEXT_TAG not in s:
        return s
    meta, text = s.split(ASR_TEXT_TAG, 1)
    if "language none" in meta.lower() and not text.strip():
        return ""
    return text.strip()


def to_wav16k(path: str, dst: str) -> str:
    """Convert any audio to 16 kHz mono wav via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1", "-ar", "16000",
         "-f", "wav", dst],
        check=True, capture_output=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", help="audio file (wav/mp3/m4a/...)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="ruby-asr",
                    help="the --served-model-name passed to vllm serve")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        wav = args.audio
        if not args.audio.lower().endswith(".wav"):
            wav = to_wav16k(args.audio, os.path.join(td, "audio_16k.wav"))
        with open(wav, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": [
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "wav"}}]},
        ],
        temperature=0.0, max_tokens=args.max_tokens)

    native = strip_asr_prefix(resp.choices[0].message.content)
    print("model output :", native)                      # 漢字[かんじ] bracket form
    print("paren form   :", brackets_to_parens(native))  # 漢字(かんじ)
    print("surface      :", to_surface(native))          # kanji text only
    print("reading      :", to_reading(native))          # kana only
    print("html         :", to_html(brackets_to_parens(native)))


if __name__ == "__main__":
    main()
