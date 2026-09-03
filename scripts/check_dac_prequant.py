"""Check that ``openamp.probes.extract`` reads DAC's **pre-quantization** latent.

``extract.py`` calls ``model.encoder(x)`` directly, not ``model.encode(x)``, on
the stated reasoning (its ``_encode`` docstring) that this hands back the
continuous encoder-trunk output *before* the residual vector quantizer. This
script verifies that empirically, on the shipped ``_encode`` function:

  1. ``_encode`` output must equal ``model.encoder(x)``      -- the pre-quant latent.
  2. ``_encode`` output must NOT equal
     ``model.encode(x).quantized_representation``            -- the post-quant one.

If check 2 failed (the two matched), every dac_frames shard would be quantized
codes, not the continuous trunk.

    python scripts/check_dac_prequant.py        # CPU, ~15 s after the model loads
"""
from __future__ import annotations

import sys

import torch

from openamp.probes.extract import _encode, load_encoder


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 difference ``||a - b|| / ||b||``."""
    return float((a - b).norm() / b.norm())


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    model, sr, hop = load_encoder(device)

    # 1 s of noise. Content is irrelevant: pre- vs post-quantization is a property
    # of which model method is called, not of the audio.
    x = torch.randn(1, 1, sr, device=device) * 0.1

    with torch.no_grad():
        enc = _encode(model, x)                              # what extract.py stores, [C, T]
        pre = model.encoder(x)[0]                            # continuous trunk, pre-quant
        post = model.encode(x, return_dict=True).quantized_representation[0]  # post-quant

    to_pre = rel_l2(enc, pre)
    to_post = rel_l2(enc, post)
    quant_gap = rel_l2(pre, post)                            # how much the quantizer moves it

    print(f"pre snippet: {pre.detach().cpu().numpy().flatten()[:10] }")
    print(f"enc snippet: {post.detach().cpu().numpy().flatten()[:10]}")
    
    print(f"[dac] {sr} Hz  hop {hop} smp/frame  latent {tuple(enc.shape)}")
    print()
    print(f"  _encode  vs  model.encoder(x)              rel L2 = {to_pre:.2e}   (want ~0)")
    print(f"  _encode  vs  quantized_representation      rel L2 = {to_post:.2e}   (want >> 0)")
    print(f"  quantizer displacement (pre vs post)       rel L2 = {quant_gap:.2e}")
    print()

    ok = to_pre < 1e-6 and to_post > 1e-2
    print("PASS: extract._encode returns the pre-quantization latent" if ok else
          "FAIL: extract._encode does NOT match the pre-quantization latent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
