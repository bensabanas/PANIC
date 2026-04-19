"""Tests for PANIC graph storage."""

import pytest
from panic.graph.storage import (
    GraphStorage, GraphNode, GraphEdge, VectorRef,
    NodeType, NodeStatus, EdgeType,
)


@pytest.fixture
def store():
    s = GraphStorage(":memory:")
    yield s
    s.close()


class TestNodes:
    def test_upsert_and_get(self, store):
        node = GraphNode(
            id="entity_python",
            type=NodeType.ENTITY,
            data={"name": "Python", "category": "language"},
            first_seen=1,
            last_seen=1,
        )
        store.upsert_node(node)

        result = store.get_node("entity_python")
        assert result is not None
        assert result.id == "entity_python"
        assert result.data["name"] == "Python"
        assert result.type == NodeType.ENTITY

    def test_upsert_increments_mention_count(self, store):
        node = GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1)
        store.upsert_node(node)
        assert store.get_node("e1").mention_count == 1

        node.last_seen = 5
        store.upsert_node(node)
        assert store.get_node("e1").mention_count == 2
        assert store.get_node("e1").last_seen == 5

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_node("nope") is None

    def test_get_nodes_by_type(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=2, last_seen=2))
        store.upsert_node(GraphNode(id="f1", type=NodeType.FACT, first_seen=3, last_seen=3))

        entities = store.get_nodes_by_type(NodeType.ENTITY)
        assert len(entities) == 2
        facts = store.get_nodes_by_type(NodeType.FACT)
        assert len(facts) == 1

    def test_update_status(self, store):
        store.upsert_node(GraphNode(id="f1", type=NodeType.FACT, first_seen=1, last_seen=1))
        store.update_node_status("f1", NodeStatus.SUPERSEDED)

        node = store.get_node("f1")
        assert node.status == NodeStatus.SUPERSEDED

    def test_search_nodes(self, store):
        store.upsert_node(GraphNode(
            id="f1", type=NodeType.FACT,
            data={"statement": "The deadline is March 15th"},
            first_seen=1, last_seen=1,
        ))
        store.upsert_node(GraphNode(
            id="f2", type=NodeType.FACT,
            data={"statement": "Use PostgreSQL for the database"},
            first_seen=2, last_seen=2,
        ))

        results = store.search_nodes("deadline")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_node_count(self, store):
        assert store.node_count() == 0
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        assert store.node_count() == 2

    def test_active_only_count(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.update_node_status("e2", NodeStatus.SUPERSEDED)

        assert store.node_count(active_only=True) == 1
        assert store.node_count(active_only=False) == 2


class TestEdges:
    def test_upsert_and_get(self, store):
        store.upsert_node(GraphNode(id="a", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="b", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_edge(GraphEdge(source_id="a", target_id="b", type=EdgeType.RELATED_TO, turn=1))

        edges = store.get_edges_from("a")
        assert len(edges) == 1
        assert edges[0].target_id == "b"
        assert edges[0].type == EdgeType.RELATED_TO

    def test_upsert_increases_weight(self, store):
        store.upsert_node(GraphNode(id="a", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="b", type=NodeType.ENTITY, first_seen=1, last_seen=1))

        store.upsert_edge(GraphEdge(source_id="a", target_id="b", type=EdgeType.RELATED_TO, turn=1))
        store.upsert_edge(GraphEdge(source_id="a", target_id="b", type=EdgeType.RELATED_TO, turn=5))

        edges = store.get_edges_from("a")
        assert len(edges) == 1
        assert edges[0].weight == 1.5  # 1.0 + 0.5

    def test_connected_nodes_single_hop(self, store):
        store.upsert_node(GraphNode(id="a", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="b", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="c", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_edge(GraphEdge(source_id="a", target_id="b", type=EdgeType.RELATED_TO, turn=1))
        store.upsert_edge(GraphEdge(source_id="b", target_id="c", type=EdgeType.RELATED_TO, turn=1))

        connected = store.get_connected_nodes("a", max_hops=1)
        assert "b" in connected
        assert "c" not in connected  # 2 hops away

    def test_connected_nodes_multi_hop(self, store):
        store.upsert_node(GraphNode(id="a", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="b", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="c", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_edge(GraphEdge(source_id="a", target_id="b", type=EdgeType.RELATED_TO, turn=1))
        store.upsert_edge(GraphEdge(source_id="b", target_id="c", type=EdgeType.RELATED_TO, turn=1))

        connected = store.get_connected_nodes("a", max_hops=2)
        assert "b" in connected
        assert "c" in connected


class TestVectorRefs:
    def test_add_and_get(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.add_vector_ref(VectorRef(node_id="e1", vector_index=5, turn=1))
        store.add_vector_ref(VectorRef(node_id="e1", vector_index=12, turn=3))

        refs = store.get_vector_refs("e1")
        assert len(refs) == 2
        indices = {r.vector_index for r in refs}
        assert indices == {5, 12}

    def test_get_nodes_for_vector(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.add_vector_ref(VectorRef(node_id="e1", vector_index=5, turn=1))
        store.add_vector_ref(VectorRef(node_id="e2", vector_index=5, turn=1))

        nodes = store.get_nodes_for_vector(5)
        assert set(nodes) == {"e1", "e2"}

    def test_penalize_vector_refs(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.add_vector_ref(VectorRef(node_id="e1", vector_index=5, turn=1))

        store.penalize_vector_refs("e1")
        refs = store.get_vector_refs("e1")
        assert refs[0].penalized is True

    def test_orphaned_nodes(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        # Only e2 gets a vector ref
        store.add_vector_ref(VectorRef(node_id="e2", vector_index=5, turn=1))

        orphaned = store.get_orphaned_nodes()
        assert len(orphaned) == 1
        assert orphaned[0].id == "e1"


class TestBulk:
    def test_clear(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="e2", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_edge(GraphEdge(source_id="e1", target_id="e2", type=EdgeType.RELATED_TO, turn=1))
        store.add_vector_ref(VectorRef(node_id="e1", vector_index=1, turn=1))

        store.clear()
        assert store.node_count() == 0
        assert store.edge_count() == 0

    def test_stats(self, store):
        store.upsert_node(GraphNode(id="e1", type=NodeType.ENTITY, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="f1", type=NodeType.FACT, first_seen=1, last_seen=1))
        store.upsert_node(GraphNode(id="d1", type=NodeType.DECISION, first_seen=1, last_seen=1))
        store.upsert_edge(GraphEdge(source_id="e1", target_id="f1", type=EdgeType.RELATED_TO, turn=1))

        stats = store.stats()
        assert stats["nodes_active"] == 3
        assert stats["entities"] == 1
        assert stats["facts"] == 1
        assert stats["decisions"] == 1
        assert stats["edges"] == 1
