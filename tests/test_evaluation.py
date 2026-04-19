"""Tests for PANIC evaluation harness."""

import pytest
from evaluation.generator import ConversationGenerator, SyntheticConversation, TEST_CONFIGS
from evaluation.scorer import (
    Scorer, EvalMetrics, TIER_1,
    baseline_naive_truncation, baseline_full_context, baseline_perfect_retrieval,
)


@pytest.fixture
def smoke_conversation() -> SyntheticConversation:
    gen = ConversationGenerator(seed=42)
    return gen.generate(**TEST_CONFIGS["smoke"])


@pytest.fixture
def standard_conversation() -> SyntheticConversation:
    gen = ConversationGenerator(seed=42)
    return gen.generate(**TEST_CONFIGS["standard"])


class TestConversationGenerator:
    def test_generates_correct_turn_count(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=50, n_needles=10, n_multi_hop=5, n_contradictions=2)

        # Each turn produces a user + assistant turn
        assert len(conv.turns) == 50 * 2

    def test_plants_needles(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=100, n_needles=20, n_multi_hop=10, n_contradictions=3)

        fact_items = [p for p in conv.planted_items if p.category == "fact"]
        assert len(fact_items) >= 15, f"Expected ~20 needles, got {len(fact_items)}"

    def test_plants_contradictions(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=100, n_needles=10, n_multi_hop=5, n_contradictions=5)

        contradictions = [p for p in conv.planted_items if p.category == "contradiction"]
        assert len(contradictions) >= 1, "Should plant at least some contradictions"

    def test_generates_probes(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=100, n_needles=20, n_multi_hop=10, n_contradictions=3)

        probe_types = set(p.probe_type for p in conv.probes)
        assert "single_hop" in probe_types
        assert "multi_hop" in probe_types or len(conv.planted_items) < 4  # need enough items for multi-hop

    def test_probes_have_expected_items(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=100, n_needles=20, n_multi_hop=10, n_contradictions=2)

        for probe in conv.probes:
            assert len(probe.expected_item_ids) >= 1, f"Probe {probe.id} has no expected items"
            assert probe.expected_answer, f"Probe {probe.id} has no expected answer"

    def test_planted_items_have_valid_turns(self):
        gen = ConversationGenerator(seed=42)
        conv = gen.generate(n_turns=100, n_needles=20, n_multi_hop=10, n_contradictions=2)

        for item in conv.planted_items:
            assert 0 <= item.turn < 100, f"Item {item.id} has invalid turn {item.turn}"

    def test_smoke_config(self, smoke_conversation):
        assert len(smoke_conversation.turns) == 100  # 50 * 2
        assert len(smoke_conversation.probes) >= 5

    def test_standard_config(self, standard_conversation):
        assert len(standard_conversation.turns) == 400  # 200 * 2
        assert len(standard_conversation.probes) >= 20

    def test_deterministic_with_seed(self):
        gen1 = ConversationGenerator(seed=123)
        conv1 = gen1.generate(n_turns=50, n_needles=10, n_multi_hop=5, n_contradictions=2)

        gen2 = ConversationGenerator(seed=123)
        conv2 = gen2.generate(n_turns=50, n_needles=10, n_multi_hop=5, n_contradictions=2)

        assert len(conv1.turns) == len(conv2.turns)
        assert len(conv1.planted_items) == len(conv2.planted_items)
        assert len(conv1.probes) == len(conv2.probes)

    def test_all_configs_generate(self):
        gen = ConversationGenerator(seed=42)
        for name, config in TEST_CONFIGS.items():
            conv = gen.generate(**config)
            assert len(conv.turns) > 0, f"Config '{name}' produced empty conversation"
            assert len(conv.probes) > 0, f"Config '{name}' produced no probes"


class TestScorer:
    def test_perfect_retrieval_scores_100(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_perfect_retrieval(smoke_conversation)

        metrics = scorer.evaluate_retrieval(smoke_conversation, retrieve)

        assert metrics.fact_retrieval_accuracy == 1.0
        assert metrics.context_inclusion_rate == 1.0
        assert metrics.total_correct == metrics.total_probes

    def test_full_context_baseline(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_full_context(smoke_conversation)

        metrics = scorer.evaluate_retrieval(smoke_conversation, retrieve)

        # Full context includes everything, so all expected items are present
        assert metrics.context_inclusion_rate == 1.0

    def test_naive_truncation_loses_old_items(self, standard_conversation):
        scorer = Scorer()
        retrieve = baseline_naive_truncation(standard_conversation, window_size=10)

        metrics = scorer.evaluate_retrieval(standard_conversation, retrieve)

        # With only last 10 turns visible, most planted items are invisible
        assert metrics.fact_retrieval_accuracy < 1.0, (
            "Naive truncation with small window should miss old facts"
        )

    def test_empty_retrieval(self, smoke_conversation):
        scorer = Scorer()

        def retrieve_nothing(query, turn):
            return []

        metrics = scorer.evaluate_retrieval(smoke_conversation, retrieve_nothing)

        assert metrics.fact_retrieval_accuracy == 0.0
        assert metrics.context_inclusion_rate == 0.0
        assert metrics.total_correct == 0

    def test_metrics_summary(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_perfect_retrieval(smoke_conversation)
        metrics = scorer.evaluate_retrieval(smoke_conversation, retrieve)

        summary = metrics.summary()
        assert "fact_retrieval_accuracy" in summary
        assert "multi_hop_accuracy" in summary
        assert "context_inclusion_rate" in summary
        assert "avg_latency_ms" in summary

    def test_compression_ratio(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_full_context(smoke_conversation)

        metrics = scorer.evaluate_retrieval(
            smoke_conversation, retrieve,
            total_tokens=100000, tokens_sent=10000,
        )

        assert metrics.compression_ratio == 10.0

    def test_latency_measured(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_perfect_retrieval(smoke_conversation)
        metrics = scorer.evaluate_retrieval(smoke_conversation, retrieve)

        assert metrics.avg_latency_ms >= 0


class TestTierTargets:
    def test_tier1_check_passing(self):
        metrics = EvalMetrics(
            fact_retrieval_accuracy=0.90,
            multi_hop_accuracy=0.70,
            contradiction_accuracy=0.96,
            context_inclusion_rate=0.85,
        )
        results = TIER_1.check(metrics)
        assert all(results.values()), f"Should pass Tier 1: {results}"

    def test_tier1_check_failing(self):
        metrics = EvalMetrics(
            fact_retrieval_accuracy=0.50,  # below 0.85 threshold
            multi_hop_accuracy=0.30,
            contradiction_accuracy=0.80,
            context_inclusion_rate=0.60,
        )
        results = TIER_1.check(metrics)
        assert not all(results.values()), "Should fail Tier 1"


class TestRegressionGate:
    def test_no_regression(self, smoke_conversation):
        scorer = Scorer()
        retrieve = baseline_perfect_retrieval(smoke_conversation)

        baseline = scorer.evaluate_retrieval(smoke_conversation, retrieve)
        current = scorer.evaluate_retrieval(smoke_conversation, retrieve)

        checks = scorer.regression_check(current, baseline)
        assert all(checks.values()), "Identical runs should not regress"

    def test_detects_regression(self):
        scorer = Scorer()

        baseline = EvalMetrics(fact_retrieval_accuracy=0.90, multi_hop_accuracy=0.70)
        regressed = EvalMetrics(fact_retrieval_accuracy=0.80, multi_hop_accuracy=0.65)

        checks = scorer.regression_check(regressed, baseline, max_regression=0.02)
        assert not checks["fact_retrieval_accuracy"], "Should detect 10% regression"
