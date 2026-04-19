"""
PANIC Graph Storage — SQLite Backend

Stores the 20% structured component:
- Nodes: entities, facts, decisions
- Edges: relationships between nodes
- Vector refs: links between graph nodes and turn embeddings

Schema is intentionally simple. SQLite handles our scale
(thousands of nodes, not millions).
"""

import sqlite3
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class NodeType(str, Enum):
    ENTITY = "entity"
    FACT = "fact"
    DECISION = "decision"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    ORPHANED = "orphaned"


class EdgeType(str, Enum):
    RELATED_TO = "related_to"
    OWNS = "owns"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    DECIDED_ON = "decided_on"
    SUPERSEDES = "supersedes"


@dataclass
class GraphNode:
    id: str
    type: NodeType
    data: dict = field(default_factory=dict)
    first_seen: int = 0  # turn number
    last_seen: int = 0
    mention_count: int = 1
    status: NodeStatus = NodeStatus.ACTIVE
    source: str = "user"  # "user" or "llm"


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    turn: int = 0


@dataclass
class VectorRef:
    node_id: str
    vector_index: int  # turn index in embedding store
    turn: int = 0
    penalized: bool = False


class GraphStorage:
    """SQLite-backed graph storage for PANIC's structured component."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                first_seen INTEGER NOT NULL DEFAULT 0,
                last_seen INTEGER NOT NULL DEFAULT 0,
                mention_count INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                source TEXT NOT NULL DEFAULT 'user'
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                turn INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE TABLE IF NOT EXISTS vector_refs (
                node_id TEXT NOT NULL,
                vector_index INTEGER NOT NULL,
                turn INTEGER NOT NULL DEFAULT 0,
                penalized INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (node_id, vector_index),
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_vector_refs_node ON vector_refs(node_id);
            CREATE INDEX IF NOT EXISTS idx_vector_refs_vector ON vector_refs(vector_index);
        """)

    # --- Node operations ---

    def upsert_node(self, node: GraphNode):
        """Insert or update a node. On conflict, update last_seen and increment mention_count."""
        self._conn.execute("""
            INSERT INTO nodes (id, type, data, first_seen, last_seen, mention_count, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_seen = excluded.last_seen,
                mention_count = mention_count + 1,
                data = CASE
                    WHEN excluded.data != '{}' THEN excluded.data
                    ELSE data
                END
        """, (
            node.id, node.type.value, json.dumps(node.data),
            node.first_seen, node.last_seen, node.mention_count,
            node.status.value, node.source,
        ))
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(self, node_type: NodeType, status: Optional[NodeStatus] = None) -> list[GraphNode]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE type = ? AND status = ?",
                (node_type.value, status.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE type = ?", (node_type.value,)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def update_node_status(self, node_id: str, status: NodeStatus):
        self._conn.execute(
            "UPDATE nodes SET status = ? WHERE id = ?", (status.value, node_id)
        )
        self._conn.commit()

    def search_nodes(self, query: str, node_type: Optional[NodeType] = None) -> list[GraphNode]:
        """Simple text search in node data. Not semantic — use vector search for that."""
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE type = ? AND data LIKE ? AND status = 'active'",
                (node_type.value, f"%{query}%"),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE data LIKE ? AND status = 'active'",
                (f"%{query}%",),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all_nodes(self, active_only: bool = True) -> list[GraphNode]:
        if active_only:
            rows = self._conn.execute("SELECT * FROM nodes WHERE status = 'active'").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    def node_count(self, active_only: bool = True) -> int:
        if active_only:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes WHERE status = 'active'").fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0]

    # --- Edge operations ---

    def upsert_edge(self, edge: GraphEdge):
        """Insert or update an edge. On conflict, increase weight."""
        self._conn.execute("""
            INSERT INTO edges (source_id, target_id, type, weight, turn)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, type) DO UPDATE SET
                weight = weight + 0.5,
                turn = excluded.turn
        """, (edge.source_id, edge.target_id, edge.type.value, edge.weight, edge.turn))
        self._conn.commit()

    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_connected_nodes(self, node_id: str, max_hops: int = 1) -> list[str]:
        """Get all node IDs connected within max_hops edges (BFS)."""
        visited = {node_id}
        frontier = {node_id}

        for _ in range(max_hops):
            next_frontier = set()
            for nid in frontier:
                # Outgoing
                rows = self._conn.execute(
                    "SELECT target_id FROM edges WHERE source_id = ?", (nid,)
                ).fetchall()
                for r in rows:
                    if r[0] not in visited:
                        next_frontier.add(r[0])

                # Incoming
                rows = self._conn.execute(
                    "SELECT source_id FROM edges WHERE target_id = ?", (nid,)
                ).fetchall()
                for r in rows:
                    if r[0] not in visited:
                        next_frontier.add(r[0])

            visited.update(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        visited.discard(node_id)
        return list(visited)

    def edge_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # --- Vector ref operations ---

    def add_vector_ref(self, ref: VectorRef):
        self._conn.execute("""
            INSERT OR IGNORE INTO vector_refs (node_id, vector_index, turn, penalized)
            VALUES (?, ?, ?, ?)
        """, (ref.node_id, ref.vector_index, ref.turn, int(ref.penalized)))
        self._conn.commit()

    def get_vector_refs(self, node_id: str) -> list[VectorRef]:
        rows = self._conn.execute(
            "SELECT * FROM vector_refs WHERE node_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_vector_ref(r) for r in rows]

    def get_nodes_for_vector(self, vector_index: int) -> list[str]:
        """Get all node IDs linked to a specific vector index."""
        rows = self._conn.execute(
            "SELECT node_id FROM vector_refs WHERE vector_index = ?", (vector_index,)
        ).fetchall()
        return [r[0] for r in rows]

    def penalize_vector_refs(self, node_id: str):
        """Mark all vector refs for a node as penalized."""
        self._conn.execute(
            "UPDATE vector_refs SET penalized = 1 WHERE node_id = ?", (node_id,)
        )
        self._conn.commit()

    def get_orphaned_nodes(self) -> list[GraphNode]:
        """Get nodes with no vector refs (orphaned after compression)."""
        rows = self._conn.execute("""
            SELECT n.* FROM nodes n
            LEFT JOIN vector_refs v ON n.id = v.node_id
            WHERE v.node_id IS NULL AND n.status = 'active'
        """).fetchall()
        return [self._row_to_node(r) for r in rows]

    # --- Bulk operations ---

    def clear(self):
        """Delete everything."""
        self._conn.executescript("""
            DELETE FROM vector_refs;
            DELETE FROM edges;
            DELETE FROM nodes;
        """)

    def stats(self) -> dict:
        return {
            "nodes_active": self.node_count(active_only=True),
            "nodes_total": self.node_count(active_only=False),
            "edges": self.edge_count(),
            "entities": len(self.get_nodes_by_type(NodeType.ENTITY)),
            "facts": len(self.get_nodes_by_type(NodeType.FACT)),
            "decisions": len(self.get_nodes_by_type(NodeType.DECISION)),
            "orphaned": len(self.get_orphaned_nodes()),
        }

    def close(self):
        self._conn.close()

    # --- Row converters ---

    def _row_to_node(self, row) -> GraphNode:
        return GraphNode(
            id=row["id"],
            type=NodeType(row["type"]),
            data=json.loads(row["data"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            mention_count=row["mention_count"],
            status=NodeStatus(row["status"]),
            source=row["source"],
        )

    def _row_to_edge(self, row) -> GraphEdge:
        return GraphEdge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=EdgeType(row["type"]),
            weight=row["weight"],
            turn=row["turn"],
        )

    def _row_to_vector_ref(self, row) -> VectorRef:
        return VectorRef(
            node_id=row["node_id"],
            vector_index=row["vector_index"],
            turn=row["turn"],
            penalized=bool(row["penalized"]),
        )
