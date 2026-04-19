"""Tests for PANIC translation layer."""

import pytest
from panic.translation.translator import (
    Translator, PromptConfig, ContextItem, ConstructedPrompt,
)


@pytest.fixture
def translator():
    return Translator(PromptConfig(max_context_tokens=1000, response_headroom=100))


def make_item(id: str, text: str, turn: int, score: float = 0.5, tier: str = "cold") -> ContextItem:
    return ContextItem(id=id, text=text, source_turn=turn, relevance_score=score, tier=tier)


class TestTokenCounting:
    def test_counts_tokens(self, translator):
        count = translator.count_tokens("Hello, how are you?")
        assert count > 0
        assert isinstance(count, int)

    def test_empty_string(self, translator):
        assert translator.count_tokens("") == 0

    def test_longer_text_more_tokens(self, translator):
        short = translator.count_tokens("Hi")
        long = translator.count_tokens("This is a much longer sentence with many more words in it")
        assert long > short


class TestPromptConstruction:
    def test_basic_construction(self, translator):
        result = translator.construct_prompt(
            query="What is the deadline?",
            immediate_buffer=[],
            retrieved_items=[make_item("r1", "The deadline is March 15th.", turn=5, score=0.9)],
            working_summaries=[],
        )

        assert "deadline" in result.full_prompt.lower()
        assert "March 15th" in result.full_prompt
        assert result.token_usage["total"] > 0
        assert "r1" in result.included_items

    def test_ordering_context_before_immediate(self, translator):
        result = translator.construct_prompt(
            query="Tell me about the project.",
            immediate_buffer=[make_item("im1", "User: What's next?\nAssistant: Let me check.", turn=10, tier="immediate")],
            retrieved_items=[make_item("r1", "The project uses PostgreSQL.", turn=2, score=0.8)],
            working_summaries=[],
        )

        # Retrieved context should appear before immediate buffer
        prompt = result.full_prompt
        retrieved_pos = prompt.find("PostgreSQL")
        immediate_pos = prompt.find("What's next")
        assert retrieved_pos < immediate_pos, "Retrieved context should come before immediate buffer"

    def test_query_at_end(self, translator):
        result = translator.construct_prompt(
            query="What is the plan?",
            immediate_buffer=[],
            retrieved_items=[make_item("r1", "Some context here.", turn=1)],
            working_summaries=[],
        )
        # Query should be at the very end
        assert result.full_prompt.rstrip().endswith("What is the plan?")

    def test_structural_cues_present(self, translator):
        result = translator.construct_prompt(
            query="Summarize.",
            immediate_buffer=[make_item("im1", "Recent chat.", turn=10, tier="immediate")],
            retrieved_items=[make_item("r1", "Old context.", turn=1)],
            working_summaries=[make_item("s1", "A summary.", turn=5, tier="working")],
        )

        assert "[Historical context" in result.full_prompt
        assert "[Recent conversation]" in result.full_prompt
        assert "[Current message]" in result.full_prompt

    def test_token_budget_respected(self):
        # Very small budget
        translator = Translator(PromptConfig(max_context_tokens=200, response_headroom=50))

        items = [make_item(f"r{i}", f"This is context item number {i} with some text." * 5, turn=i) for i in range(20)]

        result = translator.construct_prompt(
            query="Question?",
            immediate_buffer=[],
            retrieved_items=items,
            working_summaries=[],
        )

        assert result.token_usage["total"] <= 200
        assert len(result.dropped_items) > 0, "Should have dropped some items"

    def test_immediate_buffer_priority(self, translator):
        """Immediate buffer should always get some budget."""
        immediate = [make_item("im1", "Very important recent turn.", turn=10, tier="immediate")]
        retrieved = [make_item(f"r{i}", "Some old context. " * 20, turn=i, score=0.9) for i in range(10)]

        result = translator.construct_prompt(
            query="What happened?",
            immediate_buffer=immediate,
            retrieved_items=retrieved,
            working_summaries=[],
        )

        assert "im1" in result.included_items, "Immediate buffer item should be included"

    def test_empty_inputs(self, translator):
        result = translator.construct_prompt(
            query="Hello",
            immediate_buffer=[],
            retrieved_items=[],
            working_summaries=[],
        )

        assert "Hello" in result.full_prompt
        assert result.token_usage["retrieved"] == 0
        assert result.token_usage["immediate"] == 0

    def test_usage_stats(self, translator):
        result = translator.construct_prompt(
            query="Test query",
            immediate_buffer=[make_item("im1", "Recent.", turn=10, tier="immediate")],
            retrieved_items=[make_item("r1", "Old.", turn=1)],
            working_summaries=[],
        )

        usage = result.token_usage
        assert "system" in usage
        assert "query" in usage
        assert "retrieved" in usage
        assert "immediate" in usage
        assert "budget" in usage
        assert "utilization" in usage
        assert 0 <= usage["utilization"] <= 1


class TestResponseAnalysis:
    def test_detects_missing_info(self, translator):
        signals = translator.analyze_response(
            "I don't have information about the database configuration."
        )
        assert len(signals) >= 1
        assert signals[0].type == "missing_info"

    def test_detects_clarification_request(self, translator):
        signals = translator.analyze_response(
            "Did you mean the production database or the staging one?"
        )
        assert len(signals) >= 1
        assert signals[0].type == "clarification_needed"

    def test_no_signals_for_normal_response(self, translator):
        signals = translator.analyze_response(
            "The database is configured with PostgreSQL 15 and runs on port 5432."
        )
        assert len(signals) == 0

    def test_case_insensitive(self, translator):
        signals = translator.analyze_response(
            "I DON'T HAVE INFORMATION about that topic."
        )
        assert len(signals) >= 1
