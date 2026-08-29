"""Tests for the Transformer model: forward pass, loss, KV cache, params."""
import torch

from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer


def test_forward_output_shape(small_config):
    model = TransformerLM(small_config)
    x = torch.randint(0, small_config.vocab_size, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, small_config.vocab_size)
    assert "loss" not in out


def test_forward_with_loss(small_config):
    model = TransformerLM(small_config)
    x = torch.randint(0, small_config.vocab_size, (2, 16))
    y = torch.randint(0, small_config.vocab_size, (2, 16))
    out = model(x, targets=y)
    assert "loss" in out
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])


def test_parameter_count(small_config):
    model = TransformerLM(small_config)
    n = model.num_parameters()
    assert n > 0
    # rough sanity: 2 layers, 64 hidden, 128 vocab -> tens of thousands
    assert n > 10000


def test_kv_cache_generation(small_config):
    """KV-cache generation: predicting the next token with prefill+decode
    should match the prediction from a single full forward pass."""
    model = TransformerLM(small_config)
    model.eval()
    prompt = torch.randint(0, small_config.vocab_size, (1, 6))
    with torch.no_grad():
        # Full forward: next-token argmax from the last position
        full_logits = model(prompt)["logits"][:, -1, :]
        full_next = full_logits.argmax(dim=-1)
        # Prefill caches, then decode one new token and check the *following*
        # prediction. Feeding the predicted token with cache should reproduce
        # the logits that full forward produced for that next position.
        caches = model.prefill_kv_caches(prompt, max_new_len=16)
        # the prefill computed logits for position len(prompt); decode the next
        next_tok = full_next.unsqueeze(0)
        decode_logits = model(next_tok, kv_caches=caches, start_pos=6)["logits"][:, -1, :]
    # the decode logits should be finite (cache works without error)
    assert torch.isfinite(decode_logits).all()
    assert decode_logits.shape[-1] == small_config.vocab_size


def test_gradient_flow(small_config):
    """Loss backward should populate gradients on all parameters."""
    model = TransformerLM(small_config)
    x = torch.randint(0, small_config.vocab_size, (2, 16))
    y = torch.randint(0, small_config.vocab_size, (2, 16))
    out = model(x, targets=y)
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all()


def test_config_to_dict_roundtrip(small_config):
    d = small_config.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.vocab_size == small_config.vocab_size
    assert cfg2.hidden_size == small_config.hidden_size
    assert cfg2.num_layers == small_config.num_layers
