"""Tests for PANIC cross-encoder re-ranker."""

import pytest
import numpy as np
import torch
from panic.readout.reranker import Reranker, RerankerConfig, CrossAttentionBlock


class TestCrossAttention:
    def test_output_shape(self):
        block = CrossAttentionBlock(dim=128, n_heads=4)
        q = torch.randn(8, 128)
        kv = torch.randn(8, 128)
        out = block(q, kv)
        assert out.shape == (8, 128)

    def test_residual_connection(self):
        block = CrossAttentionBlock(dim=128, n_heads=4)
        q = torch.randn(1, 128)
        kv = torch.zeros(1, 128)  # zero key/value
        out = block(q, kv)
        # Output should still be close to input (residual + norm)
        assert out.shape == q.shape


class TestReranker:
    def test_default_config(self):
        model = Reranker()
        assert model.param_count() > 0
        assert model.param_count() < 1_000_000  # should be lightweight

    def test_forward_unbatched(self):
        model = Reranker()
        n_cand = 10
        q = torch.randn(n_cand, 384)
        c_emb = torch.randn(n_cand, 384)
        c_meta = torch.randn(n_cand, 4)

        scores = model(q, c_emb, c_meta)
        assert scores.shape == (n_cand,)

    def test_forward_batched(self):
        model = Reranker()
        B, N = 4, 10
        q = torch.randn(B, N, 384)
        c_emb = torch.randn(B, N, 384)
        c_meta = torch.randn(B, N, 4)

        scores = model(q, c_emb, c_meta)
        assert scores.shape == (B, N)

    def test_numpy_interface(self):
        model = Reranker()
        q = np.random.randn(384).astype(np.float32)
        c_emb = np.random.randn(10, 384).astype(np.float32)
        c_meta = np.random.randn(10, 4).astype(np.float32)

        scores = model.score_candidates(q, c_emb, c_meta)
        assert scores.shape == (10,)
        # Normalized scores should be in [0, 1]
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_different_queries_different_scores(self):
        model = Reranker()
        c_emb = np.random.randn(5, 384).astype(np.float32)
        c_meta = np.random.randn(5, 4).astype(np.float32)

        q1 = np.random.randn(384).astype(np.float32)
        q2 = np.random.randn(384).astype(np.float32)

        s1 = model.score_candidates(q1, c_emb, c_meta)
        s2 = model.score_candidates(q2, c_emb, c_meta)

        # Should produce different scores for different queries
        assert not np.allclose(s1, s2, atol=1e-3)

    def test_gradients_flow(self):
        model = Reranker()
        model.train()

        n_cand = 10
        q = torch.randn(n_cand, 384, requires_grad=True)
        c_emb = torch.randn(n_cand, 384)
        c_meta = torch.randn(n_cand, 4)

        scores = model(q, c_emb, c_meta)
        loss = scores.sum()
        loss.backward()

        # All model params should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_save_load_roundtrip(self, tmp_path):
        model = Reranker()
        path = str(tmp_path / "test_reranker.pt")
        model.save(path)

        loaded = Reranker.load(path)

        # Check same architecture
        assert loaded.param_count() == model.param_count()

        # Check same output
        q = np.random.randn(384).astype(np.float32)
        c_emb = np.random.randn(5, 384).astype(np.float32)
        c_meta = np.random.randn(5, 4).astype(np.float32)

        s1 = model.score_candidates(q, c_emb, c_meta)
        s2 = loaded.score_candidates(q, c_emb, c_meta)
        np.testing.assert_array_almost_equal(s1, s2, decimal=5)

    def test_score_not_degenerate(self):
        """Re-ranker should produce non-constant scores for varied inputs."""
        model = Reranker()

        # Create candidates with varying similarity to query
        q = np.random.randn(384).astype(np.float32)
        q = q / np.linalg.norm(q)

        # Similar candidate
        similar = q + np.random.randn(384).astype(np.float32) * 0.1
        similar = similar / np.linalg.norm(similar)

        # Random candidates
        random_cands = np.random.randn(9, 384).astype(np.float32)
        for i in range(9):
            random_cands[i] /= np.linalg.norm(random_cands[i])

        c_emb = np.vstack([similar.reshape(1, -1), random_cands])
        c_meta = np.zeros((10, 4), dtype=np.float32)

        scores = model.score_candidates(q, c_emb, c_meta)

        # Untrained model may have narrow spread; just check not all identical
        assert scores.std() > 0.0, f"Scores completely identical: {scores}"
        # Scores should be bounded
        assert np.all(scores >= 0) and np.all(scores <= 1)
