"""Tests for PFC Working Memory + Basal Ganglia Selection."""

import numpy as np
import pytest
from panic.reservoir.working_memory import WorkingMemory, WorkingMemoryConfig
from panic.reservoir.reservoir import Reservoir, ReservoirConfig


class TestWorkingMemory:
    """Tests for Experiment #6: PFC Working Memory."""

    def _make_states(self, dim=64):
        """Helper to create pre/post states with known delta."""
        pre = np.random.randn(dim).astype(np.float32) * 0.1
        post = pre + np.random.randn(dim).astype(np.float32) * 0.5
        return pre, post

    def test_stores_high_surprise_turns(self):
        wm = WorkingMemory(WorkingMemoryConfig(capacity=10, surprise_threshold=0.3))

        # First few turns establish baseline
        for t in range(5):
            emb = np.random.randn(384).astype(np.float32)
            pre = np.random.randn(64).astype(np.float32) * 0.1
            post = pre + np.random.randn(64).astype(np.float32) * 0.01  # small change
            wm.observe(t, emb, pre, post, {"fast": pre[:32]}, {"fast": post[:32]})

        # Now a big surprise
        emb = np.random.randn(384).astype(np.float32)
        pre = np.random.randn(64).astype(np.float32) * 0.1
        post = pre + np.random.randn(64).astype(np.float32) * 5.0  # huge change
        wm.observe(5, emb, pre, post, {"fast": pre[:32]}, {"fast": post[:32]})

        assert wm.size > 0, "Should store at least the high-surprise turn"
        assert 5 in wm.stored_turns, "Turn 5 (high surprise) should be stored"

    def test_capacity_limit(self):
        wm = WorkingMemory(WorkingMemoryConfig(capacity=5, surprise_threshold=0.0))

        for t in range(20):
            emb = np.random.randn(384).astype(np.float32)
            pre = np.random.randn(64).astype(np.float32)
            post = pre + np.random.randn(64).astype(np.float32) * 0.5
            wm.observe(t, emb, pre, post, {}, {})

        assert wm.size <= 5, f"Should not exceed capacity, got {wm.size}"

    def test_evicts_lowest_relevance(self):
        wm = WorkingMemory(WorkingMemoryConfig(
            capacity=3, surprise_threshold=0.0, decay_rate=0.5
        ))

        # Store 3 entries
        for t in range(3):
            emb = np.random.randn(384).astype(np.float32)
            pre = np.random.randn(64).astype(np.float32)
            post = pre + np.random.randn(64).astype(np.float32) * 0.5
            wm.observe(t, emb, pre, post, {}, {})

        assert wm.size == 3

        # Add one more — should evict the oldest (most decayed)
        emb = np.random.randn(384).astype(np.float32)
        pre = np.random.randn(64).astype(np.float32)
        post = pre + np.random.randn(64).astype(np.float32) * 0.5
        wm.observe(3, emb, pre, post, {}, {})

        assert wm.size == 3
        assert 3 in wm.stored_turns, "Newest entry should be stored"

    def test_score_candidates_shape(self):
        wm = WorkingMemory(WorkingMemoryConfig(surprise_threshold=0.0))

        for t in range(10):
            emb = np.random.randn(384).astype(np.float32)
            pre, post = np.random.randn(64).astype(np.float32), np.random.randn(64).astype(np.float32)
            wm.observe(t, emb, pre, post, {}, {})

        query = np.random.randn(384).astype(np.float32)
        scores = wm.score_candidates(query, [0, 3, 5, 7, 9])
        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_score_similar_query_higher(self):
        """Query similar to a stored WM entry should score higher."""
        wm = WorkingMemory(WorkingMemoryConfig(surprise_threshold=0.0))

        # Store a turn with a specific embedding
        target_emb = np.ones(384, dtype=np.float32)
        target_emb /= np.linalg.norm(target_emb)
        pre = np.zeros(64, dtype=np.float32)
        post = np.ones(64, dtype=np.float32)  # big change = high surprise
        wm.observe(5, target_emb, pre, post, {}, {})

        # Query similar to target
        similar_query = target_emb + np.random.randn(384).astype(np.float32) * 0.1
        # Query different from target
        diff_query = -target_emb

        scores_sim = wm.score_candidates(similar_query, [5])
        scores_diff = wm.score_candidates(diff_query, [5])

        assert scores_sim[0] > scores_diff[0], (
            f"Similar query should score higher: {scores_sim[0]:.4f} vs {scores_diff[0]:.4f}"
        )

    def test_disabled_returns_zeros(self):
        wm = WorkingMemory(WorkingMemoryConfig(enabled=False))
        emb = np.random.randn(384).astype(np.float32)
        pre, post = np.random.randn(64).astype(np.float32), np.random.randn(64).astype(np.float32)
        wm.observe(0, emb, pre, post, {}, {})
        assert wm.size == 0

        scores = wm.score_candidates(emb, [0, 1, 2])
        assert np.allclose(scores, 0.0)

    def test_reset(self):
        wm = WorkingMemory(WorkingMemoryConfig(surprise_threshold=0.0))
        for t in range(5):
            emb = np.random.randn(384).astype(np.float32)
            pre, post = np.random.randn(64).astype(np.float32), np.random.randn(64).astype(np.float32)
            wm.observe(t, emb, pre, post, {}, {})

        assert wm.size > 0
        wm.reset()
        assert wm.size == 0
        assert len(wm.stored_turns) == 0

    def test_get_entries_diagnostic(self):
        wm = WorkingMemory(WorkingMemoryConfig(surprise_threshold=0.0))
        emb = np.random.randn(384).astype(np.float32)
        pre, post = np.zeros(64, dtype=np.float32), np.ones(64, dtype=np.float32)
        wm.observe(0, emb, pre, post, {"fast": pre[:32], "slow": pre[32:]},
                   {"fast": post[:32], "slow": post[32:]})

        entries = wm.get_entries()
        assert len(entries) == 1
        assert entries[0]["turn"] == 0
        assert "surprise" in entries[0]
        assert "sub_deltas" in entries[0]

    def test_integration_with_reservoir(self):
        """Full integration: reservoir + working memory."""
        reservoir = Reservoir()
        wm = WorkingMemory()

        for t in range(20):
            emb = np.random.randn(384).astype(np.float32)
            pre_state = reservoir.state.copy()
            pre_sub = {k: v.copy() for k, v in reservoir.sub_states.items()}
            post_state = reservoir.update(emb)
            post_sub = reservoir.sub_states
            wm.observe(t, emb, pre_state, post_state, pre_sub, post_sub)

        assert wm.size > 0, "WM should have entries after 20 turns"
        assert wm.size <= wm.config.capacity

        # Score candidates
        query = np.random.randn(384).astype(np.float32)
        scores = wm.score_candidates(query, list(range(20)))
        assert scores.shape == (20,)
        assert np.any(scores > 0), "At least some candidates should have non-zero WM scores"

    def test_nearby_turn_matching(self):
        """WM should match candidates within ±2 turns of stored entries."""
        wm = WorkingMemory(WorkingMemoryConfig(surprise_threshold=0.0))

        emb = np.ones(384, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        pre = np.zeros(64, dtype=np.float32)
        post = np.ones(64, dtype=np.float32)
        wm.observe(10, emb, pre, post, {}, {})

        query = emb.copy()
        # Turn 10 should score (exact match)
        # Turns 8-12 should also score (nearby matching)
        scores = wm.score_candidates(query, [5, 8, 10, 12, 15])
        assert scores[2] > 0, "Exact turn should score"
        assert scores[1] > 0, "Nearby turn (10-2=8) should score"
        assert scores[3] > 0, "Nearby turn (10+2=12) should score"
        assert scores[0] == 0, "Turn 5 (far away) should not score"
