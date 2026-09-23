---
license: apache-2.0
language:
- ja
base_model: Qwen/Qwen3-ASR-1.7B
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- automatic-speech-recognition
- japanese
- furigana
- ruby
- ctc
- qwen3-asr
---

# Ruby-ASR-1.7B: Japanese ASR with inline furigana + a CTC mora-reading head

Ruby-ASR-1.7B is a pair of Japanese fine-tunes of
[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), shipped as
two variants in this repo:

| model | subfolder | style |
|---|---|---|
| **Ruby-ASR-sub** | `subtitle/` | subtitle-style transcription |
| **Ruby-ASR-ver** | `verbatim/` | verbatim-style transcription |

Use `subtitle` when prioritizing concise, readable orthographic transcription;
use `verbatim` when prioritizing script-aware and lexical-reading accuracy
(see [Evaluation](#evaluation)).

Both variants share the same architecture, mora vocab, and layout, and each
adds **two complementary outputs**:

1. **Ruby (furigana) transcription** — the standard Qwen3-ASR seq2seq decoder
   emits the transcript with inline bracket furigana:
   `七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、`
   Every annotated word carries its reading, so downstream text display
   (HTML `<ruby>`), reading-aware subtitling, and pronunciation-controlled TTS
   all get the reading for free.
2. **Mora reading (CTC)** — an auxiliary branch on the audio encoder
   (2-layer Transformer adapter + linear head, trained jointly with CTC loss)
   that outputs the **actual pronunciation** as kana morae directly from the
   encoder, without running the LLM decoder:
   `ナナネンカンデロスニモトモダチデキタシ`

Each variant's sharded main weights are a **stock Qwen3-ASR checkpoint** (the
CTC branch is stored separately under `ctc/`), so Mode 2 works with unmodified
vLLM and the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR)
toolkit. Mode 1 is provided by the
[`ruby_asr`](https://github.com/hshi-speech/Ruby-ASR-1.7B) package.

## Repo layout

```
subtitle/  verbatim/                    (same layout in both)
├── model-0000x-of-00003.safetensors    sharded Qwen3-ASR weights (CTC stripped) — Mode 2
├── model.safetensors.index.json
├── config.json, generation_config.json
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
│   added_tokens.json, special_tokens_map.json, chat_template.json,
│   preprocessor_config.json
└── ctc/
    ├── model.safetensors               the CTC branch (ctc_adapter.* + ctc_head.*) — Mode 1
    ├── config.json                     adapter hyper-params (transformer, 2 layers, nhead 8), blank id
    └── mora_vocab.json                 {mora: id}; id 0 = <blank>, ids 1-26 = a-z, then morae
```

## Mode 1 — mora reading via encoder + CTC

No LLM decoding: audio → encoder frames → CTC adapter → head → greedy collapse.
Fast, and reflects the *actual pronunciation* (e.g. numbers and kanji resolved
to what was said). Loads straight from this repo:

```python
# pip install git+https://github.com/hshi-speech/Ruby-ASR-1.7B.git
from ruby_asr import MoraCTCRecognizer

rec = MoraCTCRecognizer.from_pretrained("hshispeech/Ruby-ASR-1.7B",
                                        subfolder="subtitle")   # or "verbatim"
print(rec.transcribe("audio.wav")[0])
# ナナネンカンデロスニモトモダチデキタシ
```

Output alphabet: katakana morae (`キョ`, `ッ`, `ン`, `ー` are single tokens)
plus `a`–`z` for embedded English. Long vowels are unified to `ー`
(`トウキョウ` → `トーキョー`), matching the pyopenjtalk `g2p(kana=True)`
convention the training targets were built with.

## Mode 2 — ruby transcription (transformers)

Download the variant you need, then load it like stock Qwen3-ASR:

```python
# pip install qwen-asr
import torch
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel

root = snapshot_download("hshispeech/Ruby-ASR-1.7B", allow_patterns=["subtitle/*"])
model = Qwen3ASRModel.from_pretrained(f"{root}/subtitle",
                                      dtype=torch.bfloat16, device_map="cuda")
print(model.transcribe(audio="audio.wav")[0].text)
# 七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、
```

Post-process the bracket output (surface / reading / HTML `<ruby>`) with the
`ruby_asr` helpers:

```python
from ruby_asr import to_surface, to_reading, to_html, brackets_to_parens
text = "七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、"
to_surface(text)   # 七年間でロスにも友達できたし、
to_reading(text)   # ななねんかんでロスにもともだちできたし、
to_html(brackets_to_parens(text))  # <ruby><rb>七</rb><rt>なな</rt></ruby>…
```

## Mode 2 — ruby transcription (vLLM)

```bash
pip install "qwen-asr[vllm]"     # or your own vLLM >= 0.14 install
hf download hshispeech/Ruby-ASR-1.7B --include "subtitle/*" --local-dir Ruby-ASR-1.7B
vllm serve Ruby-ASR-1.7B/subtitle --served-model-name ruby-asr --max-model-len 8192
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
audio_b64 = base64.b64encode(open("audio.wav", "rb").read()).decode()
resp = client.chat.completions.create(
    model="ruby-asr",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "input_audio",
             "input_audio": {"data": audio_b64, "format": "wav"}}]},
    ],
    temperature=0.0, max_tokens=512)
print(resp.choices[0].message.content)
# language Japanese<asr_text>七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、
```

> **Use `/v1/chat/completions`, not `/v1/audio/transcriptions`.** The furigana
> is the model's learned behaviour under the standard Qwen3-ASR chat prompt
> (empty system + audio). The transcriptions endpoint builds a different
> internal prompt and returns plain text without brackets.

## Training objective

Fine-tuned from Qwen3-ASR-1.7B with a joint loss `0.3 · CTC + 0.7 · CE`: the
standard cross-entropy over the seq2seq ruby transcript plus a CTC loss over
mora targets from the auxiliary encoder branch. The seq2seq path never sees
the CTC adapter, so ruby decoding is bit-identical to a checkpoint without
the branch.

## Evaluation

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

## Limitations

- Japanese-focused fine-tune; other Qwen3-ASR languages were not part of
  training and quality on them is untested.
- Mode 1 output is pronunciation kana only — no kanji, punctuation, or word
  boundaries.
- Furigana brackets are learned behaviour, not constrained decoding: rare
  formatting slips (e.g. an empty `[]` on numbers) are possible; the parser in
  the code repo handles them.

## Citation

```bibtex
@misc{shi2026rubyasr,
  title        = {Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition},
  author       = {Hao Shi and Yun Liu and Xuehao Yang and Jun Liu and Chuanbo Hua and Xuanjun Chen and Lianbo Liu and Shiao Zhu and Zixiong Su},
  year         = {2026},
  howpublished = {\url{https://github.com/hshi-speech/Ruby-ASR-1.7B}}
}
```

## License

Apache-2.0, same as the Qwen3-ASR base model.
