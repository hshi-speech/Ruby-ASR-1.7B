"""Mode 1 — encoder + CTC mora-reading recognition.

The released checkpoint carries an auxiliary CTC branch trained jointly with the
seq2seq decoder:

    audio -> audio_tower frames -> ctc_adapter -> ctc_head
          -> per-frame argmax -> collapse repeats + drop blank -> mora/kana string

The branch lives in the model repo's `ctc/` subfolder (weights + config + mora
vocab) so the main `model.safetensors` stays a stock Qwen3-ASR checkpoint that
vLLM / transformers load unmodified. This module wires the branch back on:

    from ruby_asr import MoraCTCRecognizer
    rec = MoraCTCRecognizer.from_pretrained("<hf-repo-or-local-dir>")
    print(rec.transcribe("audio.wav")[0])      # -> 'ナナネンカンデロスニモトモダチデキタシ'

Decoding is greedy CTC (frame argmax, collapse repeats, drop blank). The output
is actual-pronunciation kana morae (katakana + a-z for embedded English), NOT
orthographic text — for full transcripts with furigana use Mode 2 (the standard
Qwen3-ASR decode path).

Also accepts a raw training checkpoint whose `model.safetensors` still contains
the `thinker.ctc_adapter.*` / `thinker.ctc_head.*` tensors — pass `mora_vocab=`
in that case (raw checkpoints record nothing about the CTC branch).
"""
import glob
import json
import os
from types import SimpleNamespace

import torch
from torch import nn

from .ctc_adapter import CTCAdapter
from .ctc_tokenizer import CTCTokenizer

CTC_SUBFOLDER = "ctc"

# Default branch hyper-parameters for raw training checkpoints (the released
# repo carries them in ctc/config.json instead).
DEFAULT_CTC_CONFIG = {
    "ctc_adapter_type": "transformer",
    "ctc_adapter_layers": 2,
    "ctc_adapter_nhead": 8,
    "ctc_adapter_dropout": 0.0,
    "ctc_blank_id": 0,
}


class CTCBranch(nn.Module):
    """The auxiliary branch: adapter + linear head over the mora vocab."""

    def __init__(self, d_model, vocab_size, cfg: dict):
        super().__init__()
        self.ctc_adapter = CTCAdapter(SimpleNamespace(d_model=d_model, **{
            k: cfg[k] for k in
            ("ctc_adapter_type", "ctc_adapter_layers",
             "ctc_adapter_nhead", "ctc_adapter_dropout")}))
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, frames):
        """(B, T, d_model) encoder frames -> (B, T, vocab) logits."""
        return self.ctc_head(self.ctc_adapter(frames))


def _resolve_model_dir(model_path, subfolder=None):
    """Local dir passed through; otherwise treat as an HF repo id and download.

    With `subfolder` (e.g. "subtitle"), only that subfolder of the repo is
    downloaded and the resolved path points inside it.
    """
    if os.path.isdir(model_path):
        return os.path.join(model_path, subfolder) if subfolder else model_path
    from huggingface_hub import snapshot_download
    if subfolder:
        root = snapshot_download(model_path, allow_patterns=[f"{subfolder}/*"])
        return os.path.join(root, subfolder)
    return snapshot_download(model_path)


def _load_ctc_state(model_dir):
    """Return (state_dict with ctc_adapter./ctc_head. keys, cfg dict, vocab or None).

    Prefers the released layout (`ctc/` subfolder); falls back to CTC tensors
    embedded in the top-level safetensors of a raw training checkpoint.
    """
    sub = os.path.join(model_dir, CTC_SUBFOLDER)
    st = os.path.join(sub, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        cfg = dict(DEFAULT_CTC_CONFIG)
        cfg_path = os.path.join(sub, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg.update(json.load(f))
        vocab = None
        vocab_path = os.path.join(sub, cfg.get("mora_vocab_file", "mora_vocab.json"))
        if os.path.isfile(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                vocab = json.load(f)
        return load_file(st), cfg, vocab

    # raw training checkpoint: filter thinker.ctc_* out of the full state dict
    sd = {}
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if shards:
        from safetensors.torch import load_file
        for f in shards:
            sd.update(load_file(f))
    else:
        bin_f = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.isfile(bin_f):
            raise FileNotFoundError(
                f"no {CTC_SUBFOLDER}/model.safetensors, *.safetensors or "
                f"pytorch_model.bin under {model_dir}")
        sd = torch.load(bin_f, map_location="cpu")
    prefix = "thinker."
    ctc_sd = {k[len(prefix):]: v for k, v in sd.items()
              if k.startswith(prefix + "ctc_adapter") or k.startswith(prefix + "ctc_head")}
    if not ctc_sd:
        raise RuntimeError(
            f"no CTC tensors found in {model_dir} -- is this a joint-CTC "
            "checkpoint (or a stripped vLLM copy, which has no CTC branch)?")
    return ctc_sd, dict(DEFAULT_CTC_CONFIG), None


class MoraCTCRecognizer:
    """Encoder + CTC mora-reading recognizer (Mode 1).

    Loads the standard Qwen3-ASR model for its audio encoder + feature
    processor, then attaches the trained CTC branch next to it. The seq2seq
    decoder is untouched (and unused) in this mode.
    """

    def __init__(self, model, processor, ctc_branch, ctc_tokenizer):
        self.model = model
        self.processor = processor
        self.ctc_branch = ctc_branch
        self.ctc_tokenizer = ctc_tokenizer

    @classmethod
    def from_pretrained(cls, model_path, device=None, dtype=None, mora_vocab=None,
                        subfolder=None):
        """Load from an HF repo id or local dir.

        Args:
            model_path: HF repo id or a local model dir (released layout, or a
                raw joint-CTC training checkpoint).
            device: e.g. "cuda:0"; defaults to cuda if available.
            dtype: defaults to bf16 on Ampere+, else fp16 on cuda, fp32 on cpu.
            mora_vocab: vocab JSON path or {mora: id} dict. Only needed for raw
                training checkpoints; the released repo ships ctc/mora_vocab.json.
            subfolder: model subfolder inside the repo/dir, e.g. "subtitle" or
                "verbatim" for the released Ruby-ASR-1.7B layout.
        """
        from qwen_asr import Qwen3ASRModel

        model_dir = _resolve_model_dir(str(model_path), subfolder)
        device = torch.device(device if device is not None
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            if device.type == "cuda":
                bf16 = torch.cuda.get_device_capability(device)[0] >= 8
                dtype = torch.bfloat16 if bf16 else torch.float16
            else:
                dtype = torch.float32

        ctc_sd, cfg, vocab = _load_ctc_state(model_dir)
        if mora_vocab is not None:
            vocab = (json.load(open(mora_vocab, encoding="utf-8"))
                     if isinstance(mora_vocab, str) else mora_vocab)
        if vocab is None:
            raise ValueError(
                "no mora vocab found: raw training checkpoints don't carry one, "
                "pass mora_vocab=<path to mora_vocab.json>")
        ctc_tok = CTCTokenizer(vocab)

        head_v = ctc_sd["ctc_head.weight"].shape[0]
        if head_v != ctc_tok.vocab_size:
            raise ValueError(
                f"mora vocab size {ctc_tok.vocab_size} != ctc_head rows {head_v} "
                "-- wrong vocab for this checkpoint")

        wrapper = Qwen3ASRModel.from_pretrained(model_dir, dtype=dtype, device_map=None)
        model = wrapper.model
        model.to(device)
        model.eval()

        d_model = model.thinker.audio_tower.proj2.out_features
        branch = CTCBranch(d_model, ctc_tok.vocab_size, cfg)
        branch.load_state_dict(ctc_sd, strict=True)
        branch.to(device, dtype)
        branch.eval()

        return cls(model, wrapper.processor, branch, ctc_tok)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _audio_prompt(self):
        """Audio-only chat prompt; the text is irrelevant to the audio features,
        it just makes the processor emit input_features for one audio per row."""
        msgs = [{"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

    @torch.inference_mode()
    def transcribe(self, audio, batch_size=8):
        """Audio -> greedy-CTC mora/kana strings.

        Args:
            audio: one or a list of: file path (wav/flac/mp3/m4a/...) or a
                16 kHz mono float numpy array.
            batch_size: encoder batch size.

        Returns:
            list[str]: one mora-reading string per input.
        """
        from .audio import load_audio

        items = audio if isinstance(audio, (list, tuple)) else [audio]
        wavs = [load_audio(a) if isinstance(a, (str, os.PathLike)) else a
                for a in items]

        out = []
        for i in range(0, len(wavs), batch_size):
            out.extend(self._transcribe_batch(wavs[i:i + batch_size]))
        return out

    def _transcribe_batch(self, wavs):
        prompt = self._audio_prompt()
        inputs = self.processor(text=[prompt] * len(wavs), audio=wavs,
                                return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        thinker = self.model.thinker
        feature_lens = inputs["feature_attention_mask"].sum(dim=1)
        out = []
        for feat, flen in zip(inputs["input_features"].to(self.model.dtype),
                              feature_lens):
            # (T_i, d_model) encoder frames for this sample only
            frames = thinker.audio_tower(
                feat[:, :flen], feature_lens=flen.unsqueeze(0)).last_hidden_state
            logits = self.ctc_branch(frames.unsqueeze(0))     # (1, T_i, V)
            ids = logits.float().argmax(dim=-1)[0].tolist()
            out.append(self.ctc_tokenizer.decode(
                ids, collapse_repeats=True, drop_blank=True, join=""))
        return out
