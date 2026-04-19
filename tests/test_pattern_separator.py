"""Tests for hippocampal pattern separation layer."""

import numpy as np
import pytest
from panic.reservoir.pattern_separator import PatternSeparator, PatternSeparatorConfig


class TestPatternSeparator:
    """Tests for the pattern separation layer."""

    def test_output_shape(self):
        ps = PatternSeparator()
        x = np.random.randn(384).astype(np.float32)
        result = ps.separate(x)
        assert result.shape == (384,), f"Expected (384,), got {result.shape}"

    def test_output_not_zero(self):
        ps = PatternSeparator()
        x = np.random.randn(384).astype(np.float32)
        result = ps.separate(x)
        assert not np.allclose(result, 0.0), "Output should not be all zeros"

    def test_different_inputs_different_outputs(self):
        ps = PatternSeparator()
        x1 = np.random.randn(384).astype(np.float32)
        x2 = np.random.randn(384).astype(np.float32)
        s1 = ps.separate(x1)
        s2 = ps.separate(x2)
        assert not np.allclose(s1, s2), "Different inputs should produce different outputs"

    def test_separation_reduces_similarity(self):
        """Similar inputs should become less similar after separation."""
        ps = PatternSeparator(PatternSeparatorConfig(
            expansion_dim=4096, expansion_sparsity=0.03, top_k=100
        ))
        rng = np.random.RandomState(42)
        base = rng.randn(384).astype(np.float32)
        x1 = base + rng.randn(384).astype(np.float32) * 0.3
        x2 = base + rng.randn(384).astype(np.float32) * 0.3

        pre_sim = float(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))

        s1 = ps.separate(x1)
        s2 = ps.separate(x2)
        post_sim = float(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

        assert post_sim < pre_sim, (
            f"Separation should reduce similarity: pre={pre_sim:.4f}, post={post_sim:.4f}"
        )

    def test_disabled_passthrough(self):
        ps = PatternSeparator(PatternSeparatorConfig(enabled=False))
        x = np.random.randn(384).astype(np.float32)
        result = ps.separate(x)
        assert np.array_equal(result, x), "Disabled separator should pass through unchanged"

    def test_wrong_shape_raises(self):
        ps = PatternSeparator()
        with pytest.raises(AssertionError):
            ps.separate(np.random.randn(100).astype(np.float32))

    def test_deterministic(self):
        """Same input, same separator → same output."""
        ps = PatternSeparator()
        x = np.random.randn(384).astype(np.float32)
        s1 = ps.separate(x)
        s2 = ps.separate(x)
        assert np.array_equal(s1, s2), "Separator should be deterministic"

    def test_measure_separation(self):
        ps = PatternSeparator()
        x1 = np.random.randn(384).astype(np.float32)
        x2 = np.random.randn(384).astype(np.float32)
        result = ps.measure_separation(x1, x2)
        assert "pre_cosine_sim" in result
        assert "post_cosine_sim" in result
        assert "separation_gain" in result
        assert "expanded_active_dims" in result

    def test_reservoir_integration(self):
        """Pattern separator should work when plugged into reservoir."""
        from panic.reservoir.reservoir import Reservoir, ReservoirConfig
        ps = PatternSeparator()
        r = Reservoir(pattern_separator=ps)

        x = np.random.randn(384).astype(np.float32)
        state = r.update(x)

        assert state.shape == (4096,)
        assert not np.allclose(state, 0.0)
        assert r.stats()["pattern_separator"] is True

    def test_reservoir_without_separator(self):
        """Reservoir without separator should still work."""
        from panic.reservoir.reservoir import Reservoir
        r = Reservoir()
        x = np.random.randn(384).astype(np.float32)
        state = r.update(x)
        assert state.shape == (4096,)
        assert r.stats()["pattern_separator"] is False

    def test_separator_changes_reservoir_dynamics(self):
        """Reservoir with separator should evolve differently than without."""
        from panic.reservoir.reservoir import Reservoir, ReservoirConfig
        ps = PatternSeparator()
        r_with = Reservoir(pattern_separator=ps)
        r_without = Reservoir()

        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(384).astype(np.float32)
            r_with.update(x)
            r_without.update(x)

        assert not np.allclose(r_with.state, r_without.state), \
            "Separator should change reservoir dynamics"
