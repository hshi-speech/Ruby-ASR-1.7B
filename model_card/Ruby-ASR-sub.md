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

# Ruby-ASR-sub: subtitle-style Japanese ASR with inline furigana + a CTC mora-reading head

<!-- EDIT before uploading: replace HF_NAMESPACE with your HF user/org name. -->

Ruby-ASR-sub is a fine-tune of [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
for **subtitle-style** Japanese transcription. A sibling checkpoint,
[`HF_NAMESPACE/Ruby-ASR-ver`](https://huggingface.co/HF_NAMESPACE/Ruby-ASR-ver),
targets verbatim-style transcription; both share the same architecture, mora
vocab, and repo layout, and differ only in the fine-tuning run.

The model adds **two complementary outputs**:

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

The main `model.safetensors` here is a **stock Qwen3-ASR checkpoint** (the CTC
branch is stored separately under `ctc/`), so Mode 2 works with unmodified
vLLM and the official [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) toolkit.

## Repo layout

```
model.safetensors        standard Qwen3-ASR weights (CTC stripped) — Mode 2
config.json, tokenizer*  standard Qwen3-ASR config/tokenizer files
ctc/
  model.safetensors      the CTC branch (ctc_adapter.* + ctc_head.*) — Mode 1
  config.json            adapter hyper-params (transformer, 2 layers, nhead 8), blank id
  mora_vocab.json        {mora: id}; id 0 = <blank>, ids 1-26 = a-z, then morae
```

## Mode 2 — ruby transcription (vLLM)

```bash
pip install "qwen-asr[vllm]"     # or your own vLLM >= 0.14 install
vllm serve HF_NAMESPACE/Ruby-ASR-sub --served-model-name ruby-asr --max-model-len 8192
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

## Mode 2 — ruby transcription (transformers)

```python
# pip install qwen-asr
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained("HF_NAMESPACE/Ruby-ASR-sub",
                                      dtype=torch.bfloat16, device_map="cuda")
print(model.transcribe(audio="audio.wav")[0].text)
# 七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、
```

Post-process the bracket output (surface / reading / HTML `<ruby>`) with the
`ruby_asr` helpers from the code repo (https://github.com/hshi-speech/Ruby-ASR):

```python
from ruby_asr import to_surface, to_reading, to_html, brackets_to_parens
text = "七[なな]年間[ねんかん]でロスにも友達[ともだち]できたし、"
to_surface(text)   # 七年間でロスにも友達できたし、
to_reading(text)   # ななねんかんでロスにもともだちできたし、
to_html(brackets_to_parens(text))  # <ruby><rb>七</rb><rt>なな</rt></ruby>…
```

## Mode 1 — mora reading via encoder + CTC

No LLM decoding: audio → encoder frames → CTC adapter → head → greedy collapse.
Fast, and reflects the *actual pronunciation* (e.g. numbers and kanji resolved
to what was said).

```python
# pip install git+https://github.com/hshi-speech/Ruby-ASR.git
from ruby_asr import MoraCTCRecognizer

rec = MoraCTCRecognizer.from_pretrained("HF_NAMESPACE/Ruby-ASR-sub")  # reads ctc/ subfolder
print(rec.transcribe("audio.wav")[0])
# ナナネンカンデロスニモトモダチデキタシ
```

Output alphabet: katakana morae (`キョ`, `ッ`, `ン`, `ー` are single tokens)
plus `a`–`z` for embedded English. Long vowels are unified to `ー`
(`トウキョウ` → `トーキョー`), matching the pyopenjtalk `g2p(kana=True)`
convention the training targets were built with.

## Training objective

Fine-tuned from Qwen3-ASR-1.7B with a joint loss `0.3 · CTC + 0.7 · CE`: the
standard cross-entropy over the seq2seq ruby transcript plus a CTC loss over
mora targets from the auxiliary encoder branch. The seq2seq path never sees
the CTC adapter, so ruby decoding is bit-identical to a checkpoint without
the branch.

## Evaluation

<!-- EDIT: fill in your numbers before uploading. -->

| test set | surface CER (Mode 2) | reading accuracy | mora error rate (Mode 1) |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Limitations

- Japanese-focused fine-tune; other Qwen3-ASR languages were not part of
  training and quality on them is untested.
- Subtitle-style output: the transcript follows subtitle conventions rather
  than strict verbatim content — for verbatim-style output use
  [`HF_NAMESPACE/Ruby-ASR-ver`](https://huggingface.co/HF_NAMESPACE/Ruby-ASR-ver).
- Mode 1 output is pronunciation kana only — no kanji, punctuation, or word
  boundaries.
- Furigana brackets are learned behaviour, not constrained decoding: rare
  formatting slips (e.g. an empty `[]` on numbers) are possible; the parser in
  the code repo handles them.

## Citation

<!-- EDIT before release: fill in the author list. -->

```bibtex
@misc{rubyasr2026,
  title        = {Ruby-ASR: Japanese ASR with inline furigana and a CTC mora-reading head},
  author       = {AUTHOR_LIST},
  year         = {2026},
  howpublished = {\url{https://github.com/hshi-speech/Ruby-ASR}}
}
```

## License

Apache-2.0, same as the Qwen3-ASR base model.
