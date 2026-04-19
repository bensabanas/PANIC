"""Tests for PANIC attention injector."""

import pytest
import numpy as np
from panic.translation.attention_injector import (
    AttentionInjector, InjectionConfig, AttentionInjection,
)
from panic.translation.translator import ContextItem
from panic.reservoir.reservoir import Reservoir, ReservoirConfig


@pytest.fixture
def reservoir():
    r = Reservoir(ReservoirConfig(mode="long_conversation"))
    # Feed some data to build up state
    for i in range(25):
        emb = np.random.randn(384).astype(np.float32)
        r.update(emb)
    return r


def make_items(n: int, max_turn: int = 100) -> list[ContextItem]:
    turns = sorted(np.random.choice(range(1, max_turn), size=min(n, max_turn - 1), replace=False))
    return [
        ContextItem(id=f"item_{i}", text=f"Context item {i} about topic {i % 5}",
                     source_turn=int(turns[i]), relevance_score=0.5 + np.random.random() * 0.5)
        for i in range(n)
    ]


class TestAttentionInjector:
    def test_empty_items(self, reservoir):
        inj = AttentionInjector()
        result = inj.generate_injection(reservoir, np.random.randn(384).astype(np.float32), [])
        assert isinstance(result, AttentionInjection)
        # No items = no injection text
        assert result.text == "" or "Memory context" in result.text

    def test_generates_injection_text(self, reservoir):
        items = make_items(10, max_turn=25)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        query = np.random.randn(384).astype(np.float32)

        inj = AttentionInjector()
        result = inj.generate_injection(reservoir, query, items, embeddings)
        assert isinstance(result, AttentionInjection)
        # Should produce some text
        assert len(result.text) > 0

    def test_drift_detection(self):
        """Feed very different data to create drift."""
        r = Reservoir(ReservoirConfig(mode="long_conversation"))

        # Phase 1: consistent topic
        for i in range(15):
            emb = np.ones(384, dtype=np.float32) * 0.5
            emb += np.random.randn(384).astype(np.float32) * 0.01
            r.update(emb)

        # Phase 2: sudden topic shift
        for i in range(15):
            emb = -np.ones(384, dtype=np.float32) * 0.5
            emb += np.random.randn(384).astype(np.float32) * 0.01
            r.update(emb)

        inj = AttentionInjector(InjectionConfig(detect_drift=True, drift_threshold=0.05))
        result = inj.generate_injection(r, np.random.randn(384).astype(np.float32), [])

        # Should detect drift after topic shift
        assert result.drift_detected is True
        assert result.drift_magnitude > 0

    def test_no_drift_when_stable(self):
        """Consistent conversation should not trigger drift."""
        r = Reservoir(ReservoirConfig(mode="long_conversation"))

        # Same general direction
        base = np.random.randn(384).astype(np.float32)
        for i in range(25):
            emb = base + np.random.randn(384).astype(np.float32) * 0.05
            r.update(emb.astype(np.float32))

        inj = AttentionInjector(InjectionConfig(detect_drift=True, drift_threshold=0.3))
        result = inj.generate_injection(r, np.random.randn(384).astype(np.float32), [])

        assert result.drift_detected is False

    def test_timescale_focus(self, reservoir):
        items = make_items(15, max_turn=25)
        embeddings = np.random.randn(15, 384).astype(np.float32)
        query = np.random.randn(384).astype(np.float32)

        inj = AttentionInjector(InjectionConfig(items_per_timescale=3))
        result = inj.generate_injection(reservoir, query, items, embeddings)

        # Should have timescale focus data
        if result.timescale_focus:
            for key in result.timescale_focus:
                assert key in ("fast", "medium", "slow")

    def test_attention_hints(self, reservoir):
        items = make_items(5, max_turn=25)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        query = np.random.randn(384).astype(np.float32)

        inj = AttentionInjector(InjectionConfig(include_hints=True))
        result = inj.generate_injection(reservoir, query, items, embeddings)

        assert isinstance(result.text, str)

    def test_config_disables_features(self, reservoir):
        items = make_items(5, max_turn=25)
        embeddings = np.random.randn(5, 384).astype(np.float32)
        query = np.random.randn(384).astype(np.float32)

        inj = AttentionInjector(InjectionConfig(
            detect_drift=False, include_hints=False
        ))
        result = inj.generate_injection(reservoir, query, items, embeddings)

        # Should still work, just less content
        assert isinstance(result, AttentionInjection)
        assert result.drift_detected is False
