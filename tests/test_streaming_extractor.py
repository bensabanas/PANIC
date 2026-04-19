"""Tests for PANIC streaming graph extractor.

Note: These tests don't call the LLM — they test the parsing,
application, and fallback logic using mocked responses.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from panic.graph.streaming_extractor import StreamingExtractor, StreamingExtractorConfig
from panic.graph.storage import GraphStorage, NodeType
from panic.graph.extractors import ExtractorPipeline


@pytest.fixture
def graph():
    g = GraphStorage(":memory:")
    yield g
    g.close()


@pytest.fixture
def rule_fallback(graph):
    return ExtractorPipeline(graph)


def make_llm_response(data: dict):
    """Create a mock litellm response."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(data)
    return mock_resp


class TestStreamingExtractor:
    def test_skips_short_turns(self, graph, rule_fallback):
        ext = StreamingExtractor(graph, rule_fallback=rule_fallback)
        result = ext.extract_turn("Hi", turn=1)
        # Should still work (falls back to rules or returns empty)
        assert result is not None

    def test_skips_already_extracted(self, graph, rule_fallback):
        ext = StreamingExtractor(graph, rule_fallback=rule_fallback)
        ext._extracted_turns.add(5)
        result = ext.extract_turn("Some long text that would be extracted", turn=5)
        assert len(result.nodes) == 0  # skipped

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_extracts_facts(self, mock_completion, graph):
        mock_completion.return_value = make_llm_response({
            "facts": [{"statement": "deadline is March 15", "subject": "deadline", "value": "March 15"}],
            "entities": [],
            "decisions": [],
            "relations": [],
        })

        ext = StreamingExtractor(graph)
        result = ext.extract_turn("The project deadline is March 15th.", turn=10)

        assert len(result.nodes) >= 1
        # Should be in graph
        stats = graph.stats()
        assert stats["nodes_active"] >= 1

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_extracts_entities(self, mock_completion, graph):
        mock_completion.return_value = make_llm_response({
            "facts": [],
            "entities": [{"name": "John Smith", "category": "person"}],
            "decisions": [],
            "relations": [],
        })

        ext = StreamingExtractor(graph)
        ext.extract_turn("John Smith will lead the project.", turn=5)

        stats = graph.stats()
        assert stats["entities"] >= 1

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_extracts_relations(self, mock_completion, graph):
        mock_completion.return_value = make_llm_response({
            "facts": [],
            "entities": [
                {"name": "Alice", "category": "person"},
                {"name": "Project X", "category": "project"},
            ],
            "decisions": [],
            "relations": [{"source": "Alice", "target": "Project X", "type": "owns"}],
        })

        ext = StreamingExtractor(graph)
        ext.extract_turn("Alice owns Project X and manages the team.", turn=10)

        stats = graph.stats()
        assert stats["edges"] >= 1

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_fallback_on_failure(self, mock_completion, graph, rule_fallback):
        mock_completion.side_effect = Exception("API error")

        ext = StreamingExtractor(graph, rule_fallback=rule_fallback)
        result = ext.extract_turn("John Smith works at Google on PANIC.", turn=1)

        # Should have fallen back to rule-based
        assert 1 in ext._extracted_turns

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_caches_extracted_turns(self, mock_completion, graph):
        mock_completion.return_value = make_llm_response({
            "facts": [{"statement": "test fact", "subject": "test", "value": "fact"}],
            "entities": [],
            "decisions": [],
            "relations": [],
        })

        ext = StreamingExtractor(graph)
        ext.extract_turn("First extraction of this turn.", turn=10)
        assert 10 in ext._extracted_turns

        # Second call should skip
        mock_completion.reset_mock()
        ext.extract_turn("Should not be extracted again.", turn=10)
        mock_completion.assert_not_called()

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_handles_empty_response(self, mock_completion, graph):
        mock_completion.return_value = make_llm_response({
            "facts": [],
            "entities": [],
            "decisions": [],
            "relations": [],
        })

        ext = StreamingExtractor(graph)
        result = ext.extract_turn("Just some casual conversation.", turn=1)
        assert len(result.nodes) == 0

    @patch("panic.graph.streaming_extractor.litellm.completion")
    def test_handles_markdown_fenced_json(self, mock_completion, graph):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '```json\n{"facts": [{"statement": "test", "subject": "x", "value": "y"}], "entities": [], "decisions": [], "relations": []}\n```'
        mock_completion.return_value = mock_resp

        ext = StreamingExtractor(graph)
        result = ext.extract_turn("The test value is important.", turn=5)
        assert len(result.nodes) >= 1
