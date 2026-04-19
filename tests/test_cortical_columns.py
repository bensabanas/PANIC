"""Tests for cortical column structure in reservoir."""

import numpy as np
import pytest
from panic.reservoir.reservoir import (
    SubReservoir, SubReservoirConfig, Reservoir, ReservoirConfig, ColumnConfig,
)


class TestCorticalColumns:
    """Tests for Experiment #1: Cortical Columns."""

    def test_column_structure_creates_blocks(self):
        """W_res should have block-diagonal structure."""
        config = SubReservoirConfig(
            name="test", dimension=128, spectral_radius=0.9, seed=42,
            columns=ColumnConfig(enabled=True, column_size=32, intra_density=0.8, inter_density=0.0),
        )
        sr = SubReservoir(config, encoder_dim=32)

        w = sr._w_res
        # With inter_density=0, only block-diagonal should be non-zero
        # 4 blocks of 32x32
        for c in range(4):
            start = c * 32
            end = start + 32
            block = w[start:end, start:end]
            block_density = np.count_nonzero(block) / block.size
            assert block_density > 0.3, (
                f"Block {c} should be dense, got {block_density:.4f}"
            )

        # Off-diagonal blocks should be empty (inter=0)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                si, ei = i * 32, (i + 1) * 32
                sj, ej = j * 32, (j + 1) * 32
                off_block = w[si:ei, sj:ej]
                assert np.count_nonzero(off_block) == 0, (
                    f"Off-diagonal block ({i},{j}) should be empty with inter=0"
                )

    def test_inter_column_connections(self):
        """With inter_density > 0, off-diagonal blocks should have connections."""
        config = SubReservoirConfig(
            name="test", dimension=128, spectral_radius=0.9, seed=42,
            columns=ColumnConfig(enabled=True, column_size=32, intra_density=0.8, inter_density=0.1),
        )
        sr = SubReservoir(config, encoder_dim=32)

        w = sr._w_res
        # Count off-diagonal connections
        off_diag_count = 0
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                si, ei = i * 32, (i + 1) * 32
                sj, ej = j * 32, (j + 1) * 32
                off_diag_count += np.count_nonzero(w[si:ei, sj:ej])

        assert off_diag_count > 0, "Inter-column connections should exist"

    def test_intra_denser_than_inter(self):
        """Intra-column density should be higher than inter-column."""
        config = SubReservoirConfig(
            name="test", dimension=256, spectral_radius=0.9, seed=42,
            columns=ColumnConfig(enabled=True, column_size=32, intra_density=0.8, inter_density=0.05),
        )
        sr = SubReservoir(config, encoder_dim=32)

        w = sr._w_res
        col_size = 32
        n_cols = 256 // col_size

        intra_total, intra_nonzero = 0, 0
        inter_total, inter_nonzero = 0, 0

        for i in range(n_cols):
            for j in range(n_cols):
                si, ei = i * col_size, (i + 1) * col_size
                sj, ej = j * col_size, (j + 1) * col_size
                block = w[si:ei, sj:ej]
                count = np.count_nonzero(block)
                size = block.size
                if i == j:
                    intra_total += size
                    intra_nonzero += count
                else:
                    inter_total += size
                    inter_nonzero += count

        intra_density = intra_nonzero / intra_total
        inter_density = inter_nonzero / inter_total

        assert intra_density > inter_density * 3, (
            f"Intra ({intra_density:.4f}) should be much denser than inter ({inter_density:.4f})"
        )

    def test_columns_produce_different_dynamics(self):
        """Column reservoir should evolve differently than uniform."""
        col_config = SubReservoirConfig(
            name="col", dimension=64, spectral_radius=0.9, seed=42,
            columns=ColumnConfig(enabled=True, column_size=16, intra_density=0.8, inter_density=0.05),
        )
        flat_config = SubReservoirConfig(
            name="flat", dimension=64, spectral_radius=0.9, seed=42,
        )

        sr_col = SubReservoir(col_config, encoder_dim=32)
        sr_flat = SubReservoir(flat_config, encoder_dim=32)

        np.random.seed(123)
        for _ in range(20):
            x = np.random.randn(32).astype(np.float32)
            sr_col.update(x)
            sr_flat.update(x)

        assert not np.allclose(sr_col.state, sr_flat.state), \
            "Column reservoir should differ from uniform"

    def test_n_columns_tracked(self):
        """SubReservoir should track number of columns."""
        config = SubReservoirConfig(
            name="test", dimension=128, spectral_radius=0.9, seed=42,
            columns=ColumnConfig(enabled=True, column_size=32),
        )
        sr = SubReservoir(config, encoder_dim=32)
        assert sr._n_columns == 4
        assert sr._column_size == 32

    def test_full_reservoir_with_columns(self):
        """Full reservoir with columns should work correctly."""
        cfg = ReservoirConfig(
            columns=ColumnConfig(enabled=True, column_size=32, intra_density=0.8, inter_density=0.05),
        )
        r = Reservoir(cfg)

        x = np.random.randn(384).astype(np.float32)
        state = r.update(x)

        assert state.shape == (4096,)
        assert not np.allclose(state, 0.0)

        stats = r.stats()
        for sr in stats["sub_reservoirs"]:
            assert sr["columns"] > 0
            assert sr["column_size"] == 32

    def test_stats_show_column_info(self):
        """Stats should include column and density info."""
        cfg = ReservoirConfig(
            columns=ColumnConfig(enabled=True, column_size=32),
        )
        r = Reservoir(cfg)
        r.update(np.random.randn(384).astype(np.float32))

        stats = r.stats()
        for sr in stats["sub_reservoirs"]:
            assert "columns" in sr
            assert "column_size" in sr
            assert "connection_density" in sr
