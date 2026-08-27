# Ruby-ASR

Japanese ASR built on [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
with **two complementary outputs** from one model:

| mode | path | output | example |
|---|---|---|---|
| **1. Mora reading** | audio encoder → CTC adapter → CTC head (greedy) | actual-pronunciation kana morae, no LLM decoding | `ナナネンカンデロスニモトモダチデキタシ` |
| **2. Ruby transcription** | standard Qwen3-ASR seq2seq decode | transcript with inline bracket furigana | `七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、` |

The released checkpoints keep the main `model.safetensors` **byte-compatible
with stock Qwen3-ASR** (the auxiliary CTC branch lives in a `ctc/` subfolder of
each model repo), so Mode 2 needs **no custom code at all**: unmodified vLLM and
the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) toolkit load it
directly. Mode 1 is provided by the small `ruby_asr` package in this repo.

## Checkpoints

Both variants live in one HF repo,
[`hshispeech/Ruby-ASR-1.7B`](https://huggingface.co/hshispeech/Ruby-ASR-1.7B):

| model | subfolder | style |
|---|---|---|
| **Ruby-ASR-sub** | `subtitle/` | subtitle-style transcription |
| **Ruby-ASR-ver** | `verbatim/` | verbatim-style transcription |

Use `subtitle` when prioritizing concise, readable orthographic transcription;
use `verbatim` when prioritizing script-aware and lexical-reading accuracy
(see [Results](#results)).

They are independent fine-tunes of the same base with the same architecture
and mora vocab, one layout:

```
hshispeech/Ruby-ASR-1.7B (HuggingFace)
├── subtitle/                                    ← and verbatim/, same layout
│   ├── model-0000x-of-00003.safetensors + index ← Mode 2 ── vllm / qwen-asr
│   ├── config.json, tokenizer*                     (no code from this repo needed)
│   └── ctc/                                     ← Mode 1 ── ruby_asr.MoraCTCRecognizer
│       ├── model.safetensors
│       ├── config.json
│       └── mora_vocab.json
└── verbatim/
```

## Install

```bash
pip install -e .              # Mode 1 + text utilities
pip install qwen-asr          # Mode 2 via transformers (optional)
pip install "qwen-asr[vllm]"  # Mode 2 via vLLM (optional)
```

The exact versions everything was verified with are pinned in
[requirements-verified.txt](requirements-verified.txt) (vLLM belongs in its own
virtualenv — see the notes in that file).

## Mode 1 — mora reading (encoder + CTC)

```python
from ruby_asr import MoraCTCRecognizer

rec = MoraCTCRecognizer.from_pretrained("hshispeech/Ruby-ASR-1.7B",
                                        subfolder="subtitle")   # or "verbatim"
print(rec.transcribe("audio.wav")[0])
# ナナネンカンデロスニモトモダチデキタシ
```

CLI: `python examples/transcribe_ctc_mora.py audio.wav --model hshispeech/Ruby-ASR-1.7B --subfolder subtitle`

The output alphabet is katakana morae (small-kana clusters like `キョ` are one
token; `ッ`, `ン`, `ー` stand alone) plus `a`–`z` for embedded English. Long
vowels are unified to `ー` (`トウキョウ` → `トーキョー`). Decoding is greedy
CTC: per-frame argmax, collapse repeats, drop blanks.

It also loads a *raw* joint-CTC training checkpoint (CTC tensors still inside
`model.safetensors`) — pass `mora_vocab=` since raw checkpoints carry no vocab.

## Mode 2 — ruby transcription (stock Qwen3-ASR path)

Download the variant you need, then serve with vLLM (nothing custom):

```bash
hf download hshispeech/Ruby-ASR-1.7B --include "subtitle/*" --local-dir Ruby-ASR-1.7B
vllm serve Ruby-ASR-1.7B/subtitle --served-model-name ruby-asr --max-model-len 8192
python examples/transcribe_vllm.py audio.wav        # OpenAI-client example
```

Or with transformers via the official toolkit:

```bash
python examples/transcribe_transformers.py audio.wav --model Ruby-ASR-1.7B/subtitle
```

> **Always query vLLM through `/v1/chat/completions`** (empty system message +
> audio), which reproduces the training prompt — the furigana brackets are the
> model's learned behaviour under that prompt. vLLM's
> `/v1/audio/transcriptions` builds a different prompt and returns plain text.

Post-processing helpers for the bracket output live in `ruby_asr.parse_ruby`:

```python
from ruby_asr import parse_ruby, to_surface, to_reading, to_html, brackets_to_parens
text = "発売[はつばい]された日[ひ]にか。"
to_surface(text)        # 発売された日にか。          (kanji surface)
to_reading(text)        # はつばいされたひにか。      (kana reading)
to_html(text)           # <ruby><rb>発売</rb><rt>はつばい</rt></ruby>…
brackets_to_parens(text)  # 発売(はつばい)された日(ひ)にか。
```

## Model architecture

Qwen3-ASR-1.7B (audio encoder `d_model 1024` → projected to 2048, Qwen3 LLM
decoder) fine-tuned on Japanese with a joint loss
`0.3 · CTC + 0.7 · CE`. The CTC branch consumes the projected encoder frames:

```
audio_tower (frozen arch, fine-tuned) ──► seq2seq decoder ──► 漢字[かんじ] ruby text
        │ (T, 2048)
        └► CTCAdapter: 2 × TransformerEncoderLayer(d=2048, nhead=8, ffn=8192, GELU, pre-norm)
           └► Linear(2048 → 278) ──► greedy CTC ──► mora reading
```

The seq2seq path never sees the adapter — ruby decoding is bit-identical to a
checkpoint without the branch, which is why stripping the CTC tensors yields a
stock-vLLM-servable model.

## Results

Character-count-weighted averages (W.Avg.) over five Japanese benchmarks —
B5K, CSJ, JSUT-Book, Common Voice 8, and TEDx — weighted by the number of
reference characters per test set. All values are percentages; lower is
better.

| Model           |  Raw CER |   SA-CER | Kana CER |
| --------------- | -------: | -------: | -------: |
| ReazonSpeech-k2 |    10.35 |     8.51 |     5.64 |
| Qwen3-ASR-1.7B  |    10.78 |     8.59 |     5.74 |
| Kana-Whisper    |      N/A |      N/A |     4.66 |
| Ruby-ASR-ver    |     9.10 | **6.59** | **3.75** |
| Ruby-ASR-sub    | **8.51** |     6.64 |     4.01 |

On this weighted aggregate, **Ruby-ASR-sub** achieves the best Raw CER
(8.51%, a 21.1% relative reduction over Qwen3-ASR), and **Ruby-ASR-ver** the
best SA-CER (6.59%) and Kana CER (3.75%) — relative reductions of 23.3% and
34.7% over Qwen3-ASR. Compared with Kana-Whisper, Ruby-ASR-ver reduces
weighted Kana CER from 4.66% to 3.75%, a 19.5% relative reduction.

Metrics:

- **Raw CER** — normalized orthographic transcription, with the original
  script preserved.
- **SA-CER** — script-aware: kana regions are scored by reading, kanji
  regions by orthography.
- **Kana CER** — lexical-reading transcription.
- Ruby-ASR and Kana-Whisper are scored on their direct reading outputs; the
  orthographic baselines are converted through the shared G2P pipeline.

## Building the HF upload (maintainers)

From a raw trainer checkpoint (never modified), one export per variant:

```bash
python scripts/prepare_hf_checkpoint.py <subtitle-checkpoint-dir> <staging>/subtitle \
    --mora-vocab <mora_vocab.json>
python scripts/prepare_hf_checkpoint.py <verbatim-checkpoint-dir> <staging>/verbatim \
    --mora-vocab <mora_vocab.json>
# shard each variant's model.safetensors (2GB shards + index), put the model
# card as <staging>/README.md and LICENSE next to it, then:
hf upload hshispeech/Ruby-ASR-1.7B <staging> .
```

The script strips the CTC tensors out of `model.safetensors`, writes them to
`ctc/model.safetensors` (with the `thinker.` prefix removed), records the
adapter hyper-parameters in `ctc/config.json` (the trainer never wrote them to
`config.json`), and copies the standard config/tokenizer files.

## Tests

```bash
pytest tests/
```

The tests cover the ruby-bracket parser and the mora CTC tokenizer; they are
pure-text and need no GPU, model download, or training data.

## Limitations

- Japanese-focused fine-tune; other Qwen3-ASR languages were not part of
  training and quality on them is untested.
- Mode 1 output is pronunciation kana only — no kanji, punctuation, or word
  boundaries.
- Furigana brackets are learned behaviour, not constrained decoding: rare
  formatting slips (e.g. an empty `[]` on numbers) are possible;
  `ruby_asr.parse_ruby` handles them.

## Citation

<!-- EDIT before release: fill in the author list. -->

```bibtex
@misc{rubyasr2026,
  title        = {Ruby-ASR: Japanese ASR with inline furigana and a CTC mora-reading head},
  author       = {AUTHOR_LIST},
  year         = {2026},
  howpublished = {\url{https://github.com/hshi-speech/Ruby-ASR-1.7B}}
}
```

## License

Apache-2.0 (same as the Qwen3-ASR base model). See [LICENSE](LICENSE).
