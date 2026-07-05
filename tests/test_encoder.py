"""Encoder: shapes, embedding dim, parameter budget, and determinism.

torch-guarded (the encoder is pure torch); skips cleanly on a bare stack.
"""

import pytest

torch = pytest.importorskip("torch")

from models.encoder import (ConvEncoder, ProjectionHead, build_encoder,
                            build_projection_head, count_parameters)
from train.config import ModelConfig


def test_forward_shapes_and_embed_dim():
    enc = build_encoder(ModelConfig())
    x = torch.randn(4, 96_000)                 # [B, T] mono
    z = enc(x)
    assert z.shape == (4, 64)
    # [B, 1, T] input is accepted too.
    z2 = enc(x.unsqueeze(1))
    assert z2.shape == (4, 64)


def test_param_budget_near_113k():
    n = count_parameters(build_encoder(ModelConfig()))
    assert 90_000 <= n <= 135_000, n           # ~113K target per spec §2


def test_projection_head_shape():
    head = build_projection_head(ModelConfig())
    assert head(torch.randn(8, 64)).shape == (8, 64)


def test_deterministic_given_seed():
    torch.manual_seed(0); a = build_encoder(ModelConfig())
    torch.manual_seed(0); b = build_encoder(ModelConfig())
    x = torch.randn(2, 96_000)
    a.eval(); b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_custom_block_plan_changes_params():
    small = ConvEncoder(blocks=[[16, 4], [16, 4], [16, 4]], embed_dim=16)
    assert count_parameters(small) < count_parameters(build_encoder(ModelConfig()))
    assert small(torch.randn(2, 96_000)).shape == (2, 16)
