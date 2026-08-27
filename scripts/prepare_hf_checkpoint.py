#!/usr/bin/env python
"""Build the HuggingFace upload directory from a raw joint-CTC training checkpoint.

The released layout keeps ONE model repo usable by both modes:

    <dst>/
      config.json ... tokenizer files     copied as-is (standard Qwen3-ASR)
      model.safetensors                   CTC tensors stripped -> stock vLLM and
                                          transformers load it unmodified (Mode 2)
      ctc/
        model.safetensors                 the CTC branch (ctc_adapter.* / ctc_head.*)
        config.json                       adapter hyper-params + blank id
        mora_vocab.json                   {mora: id}, id 0 = <blank>
      README.md                           model card (from model_card/README.md)

Mode 1 (`ruby_asr.MoraCTCRecognizer`) reads the `ctc/` subfolder; Mode 2 never
touches it (top-level weight globs don't descend into subfolders).

    python scripts/prepare_hf_checkpoint.py <src_ckpt_dir> <dst_dir> \
        --mora-vocab <mora_vocab.json>

Then upload:

    hf upload <user>/<repo> <dst_dir> .   # or: huggingface-cli upload

Idempotent: re-running rewrites <dst>. The source checkpoint is never modified.
"""
import argparse
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

# files Mode 2 needs (config + tokenizer + preprocessor); trainer junk is skipped.
COPY_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "chat_template.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
)

CTC_PREFIX = "thinker."


def is_ctc_key(k):
    return "ctc_adapter" in k or "ctc_head" in k


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="raw joint-CTC checkpoint dir (the trainer's checkpoint-*)")
    ap.add_argument("dst", help="output dir to upload to HF")
    ap.add_argument("--mora-vocab", required=True,
                    help="mora vocab JSON the checkpoint was trained with "
                         "(head rows must equal vocab size)")
    ap.add_argument("--adapter-type", default="transformer",
                    choices=["placeholder", "lstm", "transformer"])
    ap.add_argument("--adapter-layers", type=int, default=2)
    ap.add_argument("--adapter-nhead", type=int, default=8)
    ap.add_argument("--adapter-dropout", type=float, default=0.0)
    ap.add_argument("--blank-id", type=int, default=0)
    ap.add_argument("--model-card", default="",
        help="model card to copy in as README.md ('' to skip; the released "
             "Ruby-ASR-1.7B repo keeps one card at the repo root instead)")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    st_in = os.path.join(src, "model.safetensors")
    if not os.path.isfile(st_in):
        sys.exit(f"error: {st_in} not found (sharded checkpoints not handled)")

    with open(args.mora_vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    if vocab.get("<blank>") != args.blank_id:
        sys.exit(f"error: vocab['<blank>'] = {vocab.get('<blank>')!r} != "
                 f"--blank-id {args.blank_id}")

    os.makedirs(os.path.join(dst, "ctc"), exist_ok=True)

    main_sd, ctc_sd = {}, {}
    with safe_open(st_in, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            if is_ctc_key(k):
                key = k[len(CTC_PREFIX):] if k.startswith(CTC_PREFIX) else k
                ctc_sd[key] = f.get_tensor(k)
            else:
                main_sd[k] = f.get_tensor(k)

    if not ctc_sd:
        sys.exit(f"error: no CTC tensors in {st_in} -- this is not a joint-CTC "
                 "checkpoint (or it was already stripped)")
    head_rows = ctc_sd["ctc_head.weight"].shape[0]
    if head_rows != len(vocab):
        sys.exit(f"error: ctc_head has {head_rows} rows but the vocab has "
                 f"{len(vocab)} entries -- wrong --mora-vocab for this checkpoint")
    d_model = ctc_sd["ctc_head.weight"].shape[1]

    fmt = {"format": meta.get("format", "pt")}
    save_file(main_sd, os.path.join(dst, "model.safetensors"), metadata=fmt)
    save_file(ctc_sd, os.path.join(dst, "ctc", "model.safetensors"), metadata=fmt)
    print(f"[prepare] model.safetensors: {len(main_sd)} tensors (CTC stripped)")
    print(f"[prepare] ctc/model.safetensors: {len(ctc_sd)} tensors "
          f"(head {head_rows} x {d_model})")

    ctc_cfg = {
        "ctc_adapter_type": args.adapter_type,
        "ctc_adapter_layers": args.adapter_layers,
        "ctc_adapter_nhead": args.adapter_nhead,
        "ctc_adapter_dropout": args.adapter_dropout,
        "ctc_blank_id": args.blank_id,
        "ctc_vocab_size": len(vocab),
        "d_model": d_model,
        "mora_vocab_file": "mora_vocab.json",
    }
    with open(os.path.join(dst, "ctc", "config.json"), "w", encoding="utf-8") as f:
        json.dump(ctc_cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(dst, "ctc", "mora_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    for name in COPY_FILES:
        srcf = os.path.join(src, name)
        if os.path.isfile(srcf):
            shutil.copyfile(srcf, os.path.join(dst, name))  # follows symlinks
    print(f"[prepare] copied config/tokenizer files -> {dst}")

    if args.model_card and os.path.isfile(args.model_card):
        shutil.copyfile(args.model_card, os.path.join(dst, "README.md"))
        print("[prepare] model card -> README.md "
              "(EDIT the repo-id placeholders before uploading)")

    print(f"[prepare] done. upload with:\n"
          f"    hf upload <user>/<repo> {dst} .")


if __name__ == "__main__":
    main()
