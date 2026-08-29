"""Tests for training: loss decreases, checkpoint save/load/resume."""
import torch

from packages.model import ModelConfig, TransformerLM
from packages.tokenizer import BPETokenizer
from services.trainer.core import (
    make_hyperparams, build_optimizer, make_batches, lr_schedule,
    save_checkpoint, load_checkpoint, evaluate_loss, TrainState,
)


def make_tiny_model_and_data():
    cfg = ModelConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4,
                      num_kv_heads=2, intermediate_size=128, max_seq_len=32)
    model = TransformerLM(cfg)
    # deterministic repeatable data
    data = list(range(128)) * 4
    return model, data, cfg


def test_training_reduces_loss():
    model, data, cfg = make_tiny_model_and_data()
    hp = make_hyperparams("fast", {"epochs": 3, "batch_size": 4, "seq_len": 16, "learning_rate": 0.01})
    opt = build_optimizer(model, hp)
    batches = make_batches(data, hp.seq_len, hp.batch_size)
    losses = []
    model.train()
    for epoch in range(hp.epochs):
        for batch in batches:
            x = batch[:, :-1]; y = batch[:, 1:]
            opt.zero_grad()
            out = model(x, targets=y)
            out["loss"].backward()
            opt.step()
        # measure
        with torch.no_grad():
            out = model(batches[0][:, :-1], targets=batches[0][:, 1:])
            losses.append(out["loss"].item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_lr_schedule_warmup():
    hp = make_hyperparams("fast", {})
    # during warmup, lr should increase from ~0
    lr0 = lr_schedule(1, 1000, hp.warmup_steps, hp.learning_rate)
    lr_mid = lr_schedule(hp.warmup_steps, 1000, hp.warmup_steps, hp.learning_rate)
    assert lr0 < lr_mid


def test_checkpoint_save_load_resume(tmp_path):
    model, data, cfg = make_tiny_model_and_data()
    hp = make_hyperparams("fast", {})
    opt = build_optimizer(model, hp)
    state = TrainState(model=model, optimizer=opt, step=42, epoch=2,
                       best_val_loss=3.14, best_val_perplexity=23.0)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, hp, state, {"val_loss": 3.14})
    # load into fresh model
    model2 = TransformerLM(cfg)
    opt2 = build_optimizer(model2, hp)
    m, o, extra = load_checkpoint(path, "cpu")
    assert extra["step"] == 42
    # weights should match
    for p1, p2 in zip(model.parameters(), m.parameters()):
        assert torch.allclose(p1, p2)


def test_evaluate_loss_returns_finite():
    model, data, cfg = make_tiny_model_and_data()
    hp = make_hyperparams("fast", {"seq_len": 16, "batch_size": 4})
    loss, ppl = evaluate_loss(model, data, hp.seq_len, hp.batch_size, "cpu")
    assert torch.isfinite(torch.tensor(loss))
    assert ppl > 0


def test_make_batches_shapes():
    data = list(range(200))
    hp = make_hyperparams("fast", {"seq_len": 16, "batch_size": 4})
    batches = make_batches(data, hp.seq_len, hp.batch_size)
    assert all(b.shape[1] == hp.seq_len + 1 for b in batches)
    assert all(b.shape[0] == hp.batch_size for b in batches)
