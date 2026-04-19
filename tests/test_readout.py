"""Tests for PANIC readout network."""

import numpy as np
import torch
import pytest
import tempfile
import os
from panic.readout.model import ReadoutNetwork, ReadoutConfig


@pytest.fixture
def model():
    return ReadoutNetwork()


@pytest.fixture
def small_model():
    return ReadoutNetwork(ReadoutConfig(
        reservoir_dim=64,
        query_dim=32,
        candidate_dim=32,
        hidden_dims=(32, 16),
    ))


class TestArchitecture:
    def test_default_config(self, model):
        assert model.config.reservoir_dim == 4096
        assert model.config.query_dim == 384
        assert model.config.n_timescales == 3

    def test_param_count_reasonable(self, model):
        count = model.param_count()
        # Should be roughly ~2.5M params, definitely under 10M
        assert 1_000_000 < count < 10_000_000, f"Param count {count} seems off"

    def test_custom_config(self):
        config = ReadoutConfig(reservoir_dim=128, query_dim=64, hidden_dims=(64, 32))
        model = ReadoutNetwork(config)
        assert model.config.reservoir_dim == 128


class TestForward:
    def test_unbatched_forward(self, model):
        s = torch.randn(4096)
        q = torch.randn(384)
        c_emb = torch.randn(10, 384)  # 10 candidates
        c_meta = torch.randn(10, 4)

        scores, weights = model(s, q, c_emb, c_meta)

        assert scores.shape == (10,)
        assert weights.shape == (3,)

    def test_batched_forward(self, model):
        s = torch.randn(4, 4096)  # batch of 4
        q = torch.randn(4, 384)
        c_emb = torch.randn(4, 10, 384)
        c_meta = torch.randn(4, 10, 4)

        scores, weights = model(s, q, c_emb, c_meta)

        assert scores.shape == (4, 10)
        assert weights.shape == (4, 3)

    def test_scores_bounded(self, model):
        s = torch.randn(4096)
        q = torch.randn(384)
        c_emb = torch.randn(20, 384)
        c_meta = torch.randn(20, 4)

        scores, weights = model(s, q, c_emb, c_meta)

        assert torch.all(scores >= 0.0)
        assert torch.all(scores <= 1.0)

    def test_timescale_weights_sum_to_one(self, model):
        s = torch.randn(4096)
        q = torch.randn(384)
        c_emb = torch.randn(5, 384)
        c_meta = torch.randn(5, 4)

        _, weights = model(s, q, c_emb, c_meta)

        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_different_queries_give_different_scores(self, model):
        s = torch.randn(4096)
        q1 = torch.randn(384)
        q2 = torch.randn(384) * -1  # very different query
        c_emb = torch.randn(5, 384)
        c_meta = torch.randn(5, 4)

        scores1, _ = model(s, q1, c_emb, c_meta)
        scores2, _ = model(s, q2, c_emb, c_meta)

        assert not torch.allclose(scores1, scores2), "Different queries should give different scores"

    def test_single_candidate(self, model):
        s = torch.randn(4096)
        q = torch.randn(384)
        c_emb = torch.randn(1, 384)
        c_meta = torch.randn(1, 4)

        scores, weights = model(s, q, c_emb, c_meta)
        assert scores.shape == (1,)


class TestNumpy:
    def test_numpy_interface(self, model):
        s = np.random.randn(4096).astype(np.float32)
        q = np.random.randn(384).astype(np.float32)
        c_emb = np.random.randn(10, 384).astype(np.float32)
        c_meta = np.random.randn(10, 4).astype(np.float32)

        scores, weights = model.score_candidates(s, q, c_emb, c_meta)

        assert isinstance(scores, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert scores.shape == (10,)
        assert weights.shape == (3,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestPersistence:
    def test_save_and_load(self, small_model):
        small_model.eval()

        s = torch.randn(64)
        q = torch.randn(32)
        c_emb = torch.randn(3, 32)
        c_meta = torch.randn(3, 4)

        with torch.no_grad():
            scores_before, weights_before = small_model(s, q, c_emb, c_meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "readout.pt")
            small_model.save(path)
            loaded = ReadoutNetwork.load(path)

            with torch.no_grad():
                scores_after, weights_after = loaded(s, q, c_emb, c_meta)

            assert torch.allclose(scores_before, scores_after)
            assert torch.allclose(weights_before, weights_after)

    def test_load_preserves_config(self, small_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "readout.pt")
            small_model.save(path)
            loaded = ReadoutNetwork.load(path)

            assert loaded.config.reservoir_dim == small_model.config.reservoir_dim
            assert loaded.config.query_dim == small_model.config.query_dim


class TestGradients:
    def test_gradients_flow(self, small_model):
        """Verify the model is trainable — gradients reach all parameters."""
        s = torch.randn(4, 64)
        q = torch.randn(4, 32)
        c_emb = torch.randn(4, 5, 32)
        c_meta = torch.randn(4, 5, 4)

        scores, weights = small_model(s, q, c_emb, c_meta)
        loss = scores.mean() + weights.mean()
        loss.backward()

        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
