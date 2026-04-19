"""Tests for cerebellar expansion layer (Experiment #4)."""

import numpy as np
import pytest
from panic.reservoir.cerebellar_expander import CerebellarExpander, CerebellarExpanderConfig


class TestCerebellarExpander:
    """Tests for the cerebellar expansion layer."""

    def test_output_shape(self):
        ce = CerebellarExpander()
        x = np.random.randn(384).astype(np.float32)
        result = ce.separate(x)
        assert result.shape == (384,), f"Expected (384,), got {result.shape}"

    def test_output_not_zero(self):
        ce = CerebellarExpander()
        x = np.random.randn(384).astype(np.float32)
        result = ce.separate(x)
        assert not np.allclose(result, 0.0), "Output should not be all zeros"

    def test_different_inputs_different_outputs(self):
        ce = CerebellarExpander()
        x1 = np.random.randn(384).astype(np.float32)
        x2 = np.random.randn(384).astype(np.float32)
        s1 = ce.separate(x1)
        s2 = ce.separate(x2)
        assert not np.allclose(s1, s2), "Different inputs should produce different outputs"

    def test_separation_reduces_similarity(self):
        """Similar inputs should become less similar after separation."""
        ce = CerebellarExpander()
        rng = np.random.RandomState(42)
        base = rng.randn(384).astype(np.float32)
        x1 = base + rng.randn(384).astype(np.float32) * 0.3
        x2 = base + rng.randn(384).astype(np.float32) * 0.3

        pre_sim = float(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))

        s1 = ce.separate(x1)
        s2 = ce.separate(x2)
        post_sim = float(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

        assert post_sim < pre_sim, (
            f"Separation should reduce similarity: pre={pre_sim:.4f}, post={post_sim:.4f}"
        )

    def test_stronger_separation_than_hippocampal(self):
        """Cerebellar expander (12288 dim) should separate more than hippocampal (4096 dim)."""
        from panic.reservoir.pattern_separator import PatternSeparator, PatternSeparatorConfig

        rng = np.random.RandomState(42)
        base = rng.randn(384).astype(np.float32)
        # Make two fairly similar vectors
        x1 = base + rng.randn(384).astype(np.float32) * 0.2
        x2 = base + rng.randn(384).astype(np.float32) * 0.2

        # Hippocampal (Experiment #2 config)
        hippo = PatternSeparator(PatternSeparatorConfig(
            expansion_dim=4096, expansion_sparsity=0.03, top_k=100
        ))
        hippo_result = hippo.measure_separation(x1, x2)

        # Cerebellar (Experiment #4 config)
        cereb = CerebellarExpander()
        cereb_result = cereb.measure_separation(x1, x2)

        assert cereb_result["separation_gain"] > hippo_result["separation_gain"], (
            f"Cerebellar should separate more: cereb={cereb_result['separation_gain']:.4f}, "
            f"hippo={hippo_result['separation_gain']:.4f}"
        )

    def test_disabled_passthrough(self):
        ce = CerebellarExpander(CerebellarExpanderConfig(enabled=False))
        x = np.random.randn(384).astype(np.float32)
        result = ce.separate(x)
        assert np.array_equal(result, x), "Disabled expander should pass through unchanged"

    def test_wrong_shape_raises(self):
        ce = CerebellarExpander()
        with pytest.raises(AssertionError):
            ce.separate(np.random.randn(100).astype(np.float32))

    def test_deterministic(self):
        """Same input, same expander → same output."""
        ce = CerebellarExpander()
        x = np.random.randn(384).astype(np.float32)
        s1 = ce.separate(x)
        s2 = ce.separate(x)
        assert np.array_equal(s1, s2), "Expander should be deterministic"

    def test_measure_separation(self):
        ce = CerebellarExpander()
        x1 = np.random.randn(384).astype(np.float32)
        x2 = np.random.randn(384).astype(np.float32)
        result = ce.measure_separation(x1, x2)
        assert "pre_cosine_sim" in result
        assert "post_cosine_sim" in result
        assert "separation_gain" in result
        assert "expanded_active_dims" in result

    def test_expanded_dims_larger_than_hippocampal(self):
        """Cerebellar should have more active expanded dims."""
        ce = CerebellarExpander()
        x = np.random.randn(384).astype(np.float32)
        expanded = np.maximum(ce._w_expand @ x, 0)
        active = int(np.count_nonzero(expanded))
        # With 12288 dims and 3% connectivity, expect many active dims
        assert active > 1000, f"Expected >1000 active dims, got {active}"

    def test_reservoir_integration(self):
        """Cerebellar expander should work when plugged into reservoir."""
        from panic.reservoir.reservoir import Reservoir, ReservoirConfig
        ce = CerebellarExpander()
        r = Reservoir(pattern_separator=ce)

        x = np.random.randn(384).astype(np.float32)
        state = r.update(x)

        assert state.shape == (4096,)
        assert not np.allclose(state, 0.0)
        assert r.stats()["pattern_separator"] is True

    def test_reservoir_dynamics_differ_from_hippocampal(self):
        """Reservoir with cerebellar should evolve differently than with hippocampal."""
        from panic.reservoir.reservoir import Reservoir, ReservoirConfig
        from panic.reservoir.pattern_separator import PatternSeparator, PatternSeparatorConfig

        hippo = PatternSeparator(PatternSeparatorConfig(
            expansion_dim=4096, expansion_sparsity=0.03, top_k=100
        ))
        cereb = CerebellarExpander()

        r_hippo = Reservoir(pattern_separator=hippo)
        r_cereb = Reservoir(pattern_separator=cereb)

        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(384).astype(np.float32)
            r_hippo.update(x)
            r_cereb.update(x)

        assert not np.allclose(r_hippo.state, r_cereb.state), \
            "Cerebellar and hippocampal should produce different reservoir dynamics"

    def test_memory_footprint(self):
        """Verify the expansion matrix is appropriately large."""
        ce = CerebellarExpander()
        expand_size = ce._w_expand.nbytes / (1024 * 1024)  # MB
        compress_size = ce._w_compress.nbytes / (1024 * 1024)  # MB
        total_mb = expand_size + compress_size
        # 12288×384×4 bytes ≈ 18 MB for expansion, 384×12288×4 ≈ 18 MB for compression
        assert 30 < total_mb < 50, f"Expected ~36 MB total, got {total_mb:.1f} MB"

    def test_sparsity_of_expansion(self):
        """Verify the expansion matrix has approximately the target sparsity."""
        ce = CerebellarExpander()
        nonzero_frac = np.count_nonzero(ce._w_expand) / ce._w_expand.size
        # With normalized rows, zero rows become non-zero after normalization,
        # but the connectivity should still be sparse relative to dense
        assert nonzero_frac < 0.10, f"Expected <10% nonzero, got {nonzero_frac:.4f}"
