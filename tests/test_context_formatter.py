"""Tests for PANIC context formatter — conversation-aware prompt construction."""

import pytest
from panic.translation.translator import ContextItem
from panic.translation.context_formatter import ContextFormatter, FormattedContext
from panic.graph.storage import (
    GraphStorage, GraphNode, GraphEdge, VectorRef,
    NodeType, NodeStatus, EdgeType,
)


@pytest.fixture
def rule_graph():
    g = GraphStorage(":memory:")
    yield g
    g.close()


def make_item(id: str, text: str, turn: int, score: float = 0.5) -> ContextItem:
    return ContextItem(id=id, text=text, source_turn=turn, relevance_score=score, tier="cold")


def make_history(turns: int) -> list[dict]:
    history = []
    for t in range(1, turns + 1):
        history.append({"role": "user", "content": f"User message at turn {t}", "turn": t})
        history.append({"role": "assistant", "content": f"Assistant reply at turn {t}", "turn": t})
    return history


class TestContextFormatter:
    def test_empty_items(self):
        fmt = ContextFormatter(total_turns=100)
        result = fmt.format_items([])
        assert result == []

    def test_adds_turn_labels(self):
        items = [make_item("r1", "The deadline is March.", turn=5)]
        fmt = ContextFormatter(total_turns=100)
        result = fmt.format_items(items)

        assert len(result) == 1
        assert "[Turn 5" in result[0].text

    def test_adds_temporal_markers(self):
        fmt = ContextFormatter(total_turns=100)

        early = fmt.format_items([make_item("r1", "Early fact.", turn=5)])
        assert "early" in early[0].text.lower()

        recent = fmt.format_items([make_item("r2", "Recent fact.", turn=95)])
        assert "recent" in recent[0].text.lower()

        mid = fmt.format_items([make_item("r3", "Mid fact.", turn=50)])
        assert "mid" in mid[0].text.lower()

    def test_adds_speaker_labels(self):
        history = [
            {"role": "user", "content": "Question", "turn": 10},
            {"role": "assistant", "content": "Answer", "turn": 11},
        ]
        fmt = ContextFormatter(chat_history=history, total_turns=100)

        result_user = fmt.format_items([make_item("r1", "Some info.", turn=10)])
        assert "user" in result_user[0].text.lower()

        result_asst = fmt.format_items([make_item("r2", "Some reply.", turn=11)])
        assert "assistant" in result_asst[0].text.lower()

    def test_sorts_by_turn(self):
        items = [
            make_item("r3", "Third.", turn=30),
            make_item("r1", "First.", turn=5),
            make_item("r2", "Second.", turn=15),
        ]
        fmt = ContextFormatter(total_turns=100)
        result = fmt.format_items(items)

        assert result[0].source_turn == 5
        assert result[1].source_turn == 15
        assert result[2].source_turn == 30

    def test_preserves_metadata(self):
        items = [make_item("r1", "Original text.", turn=10, score=0.9)]
        fmt = ContextFormatter(total_turns=100)
        result = fmt.format_items(items)

        assert result[0].id == "r1"
        assert result[0].source_turn == 10
        assert result[0].relevance_score == 0.9

    def test_contradiction_annotation(self, rule_graph):
        # Create a superseded node
        old_node = GraphNode(
            id="fact_old", type=NodeType.FACT,
            data={"statement": "deadline is March"},
            first_seen=10, last_seen=10,
            status=NodeStatus.SUPERSEDED,
        )
        new_node = GraphNode(
            id="fact_new", type=NodeType.FACT,
            data={"statement": "deadline is April"},
            first_seen=50, last_seen=50,
            status=NodeStatus.ACTIVE,
        )
        rule_graph.upsert_node(old_node)
        rule_graph.upsert_node(new_node)

        # Edge: new supersedes old
        rule_graph.upsert_edge(GraphEdge(
            source_id="fact_new", target_id="fact_old",
            type=EdgeType.SUPERSEDES, turn=50,
        ))

        # Vector refs
        rule_graph.add_vector_ref(VectorRef(node_id="fact_old", vector_index=10, turn=10))
        rule_graph.add_vector_ref(VectorRef(node_id="fact_new", vector_index=50, turn=50))

        history = make_history(60)
        fmt = ContextFormatter(rule_graph=rule_graph, chat_history=history, total_turns=60)

        items = [make_item("r1", "deadline is March", turn=10)]
        result = fmt.format_items(items)

        # Should have a contradiction note
        assert "Updated" in result[0].text or "updated" in result[0].text

    def test_related_turns_annotation(self, rule_graph):
        # Create two related nodes
        node_a = GraphNode(id="entity_a", type=NodeType.ENTITY,
                           data={"name": "Alice"}, first_seen=10, last_seen=10)
        node_b = GraphNode(id="entity_b", type=NodeType.ENTITY,
                           data={"name": "Bob"}, first_seen=20, last_seen=20)
        rule_graph.upsert_node(node_a)
        rule_graph.upsert_node(node_b)

        rule_graph.upsert_edge(GraphEdge(
            source_id="entity_a", target_id="entity_b",
            type=EdgeType.RELATED_TO, turn=20,
        ))

        rule_graph.add_vector_ref(VectorRef(node_id="entity_a", vector_index=10, turn=10))
        rule_graph.add_vector_ref(VectorRef(node_id="entity_b", vector_index=20, turn=20))

        fmt = ContextFormatter(rule_graph=rule_graph, total_turns=50)

        items = [make_item("r1", "Alice info", turn=10)]
        result = fmt.format_items(items)

        assert "Related" in result[0].text
        assert "20" in result[0].text

    def test_no_graph_still_works(self):
        """Formatter should work without a graph (just turn labels + temporal)."""
        items = [make_item("r1", "Some fact.", turn=25)]
        fmt = ContextFormatter(total_turns=100)
        result = fmt.format_items(items)

        assert len(result) == 1
        assert "[Turn 25" in result[0].text
        assert "earlier" in result[0].text.lower()

    def test_large_batch(self):
        """Format many items without crashing."""
        items = [make_item(f"r{i}", f"Fact number {i}.", turn=i) for i in range(1, 101)]
        fmt = ContextFormatter(total_turns=200)
        result = fmt.format_items(items)

        assert len(result) == 100
        # Should be sorted by turn
        for i in range(len(result) - 1):
            assert result[i].source_turn <= result[i + 1].source_turn
