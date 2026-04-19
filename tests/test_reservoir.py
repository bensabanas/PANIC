"""Tests for the PANIC reservoir."""

import numpy as np
import pytest
from panic.reservoir.reservoir import SubReservoir, SubReservoirConfig, Reservoir, ReservoirConfig


class TestSubReservoir:
    """Tests for individual sub-reservoirs."""

    def make_config(self, **kwargs) -> SubReservoirConfig:
        defaults = dict(name="test", dimension=64, spectral_radius=0.9, seed=42)
        defaults.update(kwargs)
        return SubReservoirConfig(**defaults)

    def test_initial_state_is_zero(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)
        assert np.allclose(sr.state, 0.0)
        assert sr.turn == 0

    def test_update_changes_state(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)
        x = np.random.randn(32).astype(np.float32)

        state = sr.update(x)

        assert not np.allclose(state, 0.0), "State should change after input"
        assert state.shape == (64,)
        assert sr.turn == 1

    def test_state_bounded_by_tanh(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)

        # Feed many large inputs
        for _ in range(100):
            x = np.random.randn(32).astype(np.float32) * 10
            state = sr.update(x)

        # tanh output is always in [-1, 1]
        assert np.all(np.abs(state) <= 1.0), "State must be bounded by tanh"

    def test_different_inputs_produce_different_states(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)

        x1 = np.ones(32, dtype=np.float32)
        x2 = -np.ones(32, dtype=np.float32)

        state1 = sr.update(x1)
        sr.reset()
        state2 = sr.update(x2)

        assert not np.allclose(state1, state2), "Different inputs should produce different states"

    def test_fading_memory(self):
        """Old inputs should have diminishing influence on current state."""
        config = self.make_config(spectral_radius=0.8, noise_level=0.0)
        sr = SubReservoir(config, encoder_dim=32)

        # Feed a distinctive input
        signal = np.ones(32, dtype=np.float32) * 5.0
        sr.update(signal)
        state_after_signal = sr.state.copy()

        # Feed many zero inputs to let the signal decay
        zero = np.zeros(32, dtype=np.float32)
        for _ in range(50):
            sr.update(zero)

        state_after_decay = sr.state

        # The signal should have faded significantly
        signal_norm = np.linalg.norm(state_after_signal)
        decayed_norm = np.linalg.norm(state_after_decay)

        assert decayed_norm < signal_norm * 0.1, (
            f"State should decay: initial norm {signal_norm:.4f}, "
            f"after 50 zero turns {decayed_norm:.4f}"
        )

    def test_spectral_radius_affects_memory_length(self):
        """Higher spectral radius = longer memory."""
        encoder_dim = 32
        signal = np.ones(encoder_dim, dtype=np.float32) * 3.0
        zero = np.zeros(encoder_dim, dtype=np.float32)

        # Short memory reservoir
        sr_short = SubReservoir(
            self.make_config(spectral_radius=0.7, noise_level=0.0, seed=42),
            encoder_dim=encoder_dim,
        )
        sr_short.update(signal)
        for _ in range(30):
            sr_short.update(zero)
        short_norm = np.linalg.norm(sr_short.state)

        # Long memory reservoir
        sr_long = SubReservoir(
            self.make_config(spectral_radius=0.95, noise_level=0.0, seed=42),
            encoder_dim=encoder_dim,
        )
        sr_long.update(signal)
        for _ in range(30):
            sr_long.update(zero)
        long_norm = np.linalg.norm(sr_long.state)

        assert long_norm > short_norm, (
            f"Higher ρ should retain more: ρ=0.95 norm={long_norm:.4f}, "
            f"ρ=0.7 norm={short_norm:.4f}"
        )

    def test_sparsity(self):
        """W_res should be approximately 10% non-zero."""
        config = self.make_config(dimension=256, sparsity=0.1, seed=42)
        sr = SubReservoir(config, encoder_dim=32)

        nonzero_ratio = np.count_nonzero(sr._w_res) / sr._w_res.size
        # Allow some tolerance — sparsity is stochastic
        assert 0.05 < nonzero_ratio < 0.15, (
            f"Expected ~10% non-zero, got {nonzero_ratio:.2%}"
        )

    def test_snapshot_roundtrip(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)

        # Run a few updates
        for _ in range(5):
            sr.update(np.random.randn(32).astype(np.float32))

        snapshot = sr.get_snapshot()
        original_state = sr.state.copy()
        original_turn = sr.turn

        # Reset and restore
        sr.reset()
        assert np.allclose(sr.state, 0.0)

        sr.load_snapshot(snapshot)
        assert np.allclose(sr.state, original_state)
        assert sr.turn == original_turn

    def test_wrong_input_shape_raises(self):
        config = self.make_config()
        sr = SubReservoir(config, encoder_dim=32)

        with pytest.raises(AssertionError):
            sr.update(np.random.randn(64).astype(np.float32))  # wrong dim


class TestReservoir:
    """Tests for the full multi-timescale reservoir."""

    def test_default_config(self):
        r = Reservoir()
        assert r.total_dimension == 4096
        assert r.turn == 0
        assert r.state.shape == (4096,)

    def test_update(self):
        r = Reservoir()
        x = np.random.randn(384).astype(np.float32)

        state = r.update(x)

        assert state.shape == (4096,)
        assert not np.allclose(state, 0.0)
        assert r.turn == 1

    def test_sub_states(self):
        r = Reservoir()
        x = np.random.randn(384).astype(np.float32)
        r.update(x)

        sub_states = r.sub_states
        assert "fast" in sub_states
        assert "medium" in sub_states
        assert "slow" in sub_states
        assert sub_states["fast"].shape == (1024,)
        assert sub_states["medium"].shape == (1024,)
        assert sub_states["slow"].shape == (2048,)

    def test_snapshots_taken_at_interval(self):
        r = Reservoir(ReservoirConfig(snapshot_interval=5))
        x = np.random.randn(384).astype(np.float32)

        for _ in range(12):
            r.update(x)

        # Should have snapshots at turn 5 and 10
        assert len(r._snapshots) == 2
        assert r._snapshots[0]["turn"] == 5
        assert r._snapshots[1]["turn"] == 10

    def test_mode_affects_spectral_radii(self):
        r_conv = Reservoir(ReservoirConfig(mode="long_conversation"))
        r_doc = Reservoir(ReservoirConfig(mode="document_analysis"))

        # Document mode should have lower spectral radii
        for sr_conv, sr_doc in zip(r_conv._sub_reservoirs, r_doc._sub_reservoirs):
            assert sr_doc.config.spectral_radius <= sr_conv.config.spectral_radius

    def test_serialize_roundtrip(self):
        r = Reservoir()

        # Run some updates
        for _ in range(15):
            r.update(np.random.randn(384).astype(np.float32))

        original_state = r.state.copy()
        original_turn = r.turn

        # Serialize and restore
        data = r.serialize()
        r2 = Reservoir.deserialize(data)

        assert np.allclose(r2.state, original_state)
        assert r2.turn == original_turn

    def test_stats(self):
        r = Reservoir()
        r.update(np.random.randn(384).astype(np.float32))

        stats = r.stats()
        assert stats["turn"] == 1
        assert stats["total_dimension"] == 4096
        assert len(stats["sub_reservoirs"]) == 3

    def test_reset(self):
        r = Reservoir()
        for _ in range(15):
            r.update(np.random.randn(384).astype(np.float32))

        assert r.turn == 15
        assert len(r._snapshots) > 0

        r.reset()

        assert r.turn == 0
        assert len(r._snapshots) == 0
        assert np.allclose(r.state, 0.0)
