"""Adapter between the Qwen3-ASR audio encoder and the CTC head.

Only the CTC branch uses this; the seq2seq decoder consumes the raw encoder
output, so this module never affects ruby (furigana) decoding.

Config knobs (all read via getattr on a simple namespace/config object):
    d_model             encoder output dim (2048 for Qwen3-ASR-1.7B)
    ctc_adapter_type    'placeholder' | 'lstm' | 'transformer'   (default 'placeholder')
    ctc_adapter_layers  int                                       (default 2)
    ctc_adapter_nhead   int (transformer only)                    (default 8)
    ctc_adapter_dropout float                                     (default 0.0)

The released checkpoints use the `transformer` kind (2 layers, nhead 8).
Contract: forward `(B, T, d_model)` -> `(B, T, d_model)` so `ctc_head` fits.
"""

from torch import nn


class CTCAdapterPlaceholder(nn.Module):
    """No-op adapter: the encoder output feeds the CTC head directly."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


class CTCAdapter(nn.Module):
    """Dispatch wrapper: routes encoder hidden states through one of the
    built-in adapter kinds (`placeholder` | `lstm` | `transformer`)."""

    def __init__(self, config):
        super().__init__()
        self.d_model = d = config.d_model
        self.kind = str(getattr(config, "ctc_adapter_type", "placeholder")).lower()
        if self.kind == "identity":               # backward-compat alias
            self.kind = "placeholder"
        n_layers = int(getattr(config, "ctc_adapter_layers", 2))
        nhead = int(getattr(config, "ctc_adapter_nhead", 8))
        dropout = float(getattr(config, "ctc_adapter_dropout", 0.0))

        if self.kind == "placeholder":
            self.layers = CTCAdapterPlaceholder(config)
        elif self.kind == "lstm":
            h = d // 2                            # bidirectional -> 2*h = d_model
            self.layers = nn.LSTM(
                d, h, num_layers=n_layers, bidirectional=True, batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.out_proj = nn.Linear(2 * h, d) if 2 * h != d else nn.Identity()
        elif self.kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=4 * d, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            # enable_nested_tensor=False keeps every forward on the same autograd
            # path (the NestedTensor fast path is data-dependent); it also keeps
            # the module structure identical to how the checkpoints were trained.
            self.layers = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False)
        else:
            raise ValueError(
                f"unknown ctc_adapter_type {self.kind!r}; "
                "choose: placeholder | lstm | transformer")

    def forward(self, hidden_states, attention_mask=None):
        """`(B, T, d_model)` -> `(B, T, d_model)`."""
        if self.kind == "placeholder":
            return self.layers(hidden_states, attention_mask=attention_mask)
        if self.kind == "lstm":
            out, _ = self.layers(hidden_states)
            return self.out_proj(out)
        return self.layers(hidden_states)
