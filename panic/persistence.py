"""
PANIC Session Persistence — SQLite-backed durable sessions (v5 simplified)

Stores everything needed to resume a session after restart:
  - Both graph stores (rule-based + LLM)
  - Turn history with embeddings
  - Chat history
  - Engine configuration and metadata

Sessions are identified by name. Multiple sessions can coexist.
The schema is append-friendly: turns are inserted, never updated.

Note: reservoir_state table is kept in the schema for backward compatibility
with existing databases (so old DBs don't break on open) but is no longer
written to or read from. Safe to drop in a future migration.
"""

import sqlite3
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionMeta:
    """Metadata about a persisted session."""
    name: str
    created_at: float
    updated_at: float
    turn_count: int
    mode: str
    provider: str = ""
    model: str = ""
    extraction_model: str = ""
    engine_version: str = "simplified_v5"


class SessionStore:
    """
    SQLite-backed persistence for PANIC sessions.

    One database file per session directory. Each session gets its own
    set of rows keyed by session name, so multiple sessions can share
    a single database file if desired.

    Usage:
        store = SessionStore("/path/to/sessions")
        store.save_session("my-chat", engine)
        store.load_session("my-chat", engine)
        store.list_sessions()
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path to the SQLite database file.
                     Use ":memory:" for testing.
        """
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                name TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                turn_count INTEGER NOT NULL DEFAULT 0,
                mode TEXT NOT NULL DEFAULT 'long_conversation',
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                extraction_model TEXT NOT NULL DEFAULT '',
                engine_version TEXT NOT NULL DEFAULT 'simplified_v5'
            );

            -- Kept for backward compat with existing DBs; no longer used
            CREATE TABLE IF NOT EXISTS reservoir_state (
                session_name TEXT PRIMARY KEY,
                state_blob BLOB NOT NULL,
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );

            CREATE TABLE IF NOT EXISTS turn_embeddings (
                session_name TEXT NOT NULL,
                turn INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (session_name, turn),
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                session_name TEXT NOT NULL,
                seq INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (session_name, seq),
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );

            CREATE TABLE IF NOT EXISTS immediate_buffer (
                session_name TEXT NOT NULL,
                seq INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                PRIMARY KEY (session_name, seq),
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );

            -- Graph nodes and edges stored as JSON blobs per session
            CREATE TABLE IF NOT EXISTS graph_snapshot (
                session_name TEXT NOT NULL,
                graph_type TEXT NOT NULL,  -- 'rule' or 'llm'
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                vector_refs_json TEXT NOT NULL,
                PRIMARY KEY (session_name, graph_type),
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );

            CREATE TABLE IF NOT EXISTS engine_config (
                session_name TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                FOREIGN KEY (session_name) REFERENCES sessions(name)
            );
        """)

    def save_session(self, name: str, engine) -> SessionMeta:
        """
        Persist the full engine state to the database.

        Args:
            name: Session name (unique identifier).
            engine: PanicEngine instance to persist.

        Returns:
            SessionMeta with saved session info.
        """
        now = time.time()

        # Upsert session metadata
        self._conn.execute("""
            INSERT INTO sessions (name, created_at, updated_at, turn_count, mode,
                                  provider, model, extraction_model, engine_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                updated_at = excluded.updated_at,
                turn_count = excluded.turn_count,
                mode = excluded.mode,
                provider = excluded.provider,
                model = excluded.model,
                extraction_model = excluded.extraction_model,
                engine_version = excluded.engine_version
        """, (
            name, now, now, engine.turn, engine.mode,
            engine.provider, engine.model, engine.extraction_model,
            "simplified_v5",
        ))

        # Save turn embeddings
        self._conn.execute(
            "DELETE FROM turn_embeddings WHERE session_name = ?", (name,)
        )
        for turn, emb in engine.turn_embeddings.items():
            self._conn.execute(
                "INSERT INTO turn_embeddings (session_name, turn, embedding) VALUES (?, ?, ?)",
                (name, turn, emb.tobytes()),
            )

        # Save chat history
        self._conn.execute(
            "DELETE FROM chat_history WHERE session_name = ?", (name,)
        )
        for i, entry in enumerate(engine.chat_history):
            self._conn.execute(
                "INSERT INTO chat_history (session_name, seq, turn, role, content, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, i, entry.get("turn", 0), entry["role"], entry["content"],
                 entry.get("timestamp", now)),
            )

        # Save immediate buffer
        self._conn.execute(
            "DELETE FROM immediate_buffer WHERE session_name = ?", (name,)
        )
        for i, entry in enumerate(engine.immediate_buffer):
            self._conn.execute(
                "INSERT INTO immediate_buffer (session_name, seq, turn, user_text, assistant_text) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, i, entry["turn"], entry["user"], entry["assistant"]),
            )

        # Save graph snapshots (both rule and LLM)
        self._save_graph(name, "rule", engine.rule_graph)
        self._save_graph(name, "llm", engine.llm_graph)

        # Save engine config
        config = {
            "buffer_size": engine.buffer_size,
            "llm_flush_interval": engine.llm_flush_interval,
            "turns_since_flush": engine._turns_since_flush,
            "item_blend": engine.item_blend,
            "w_cosine": engine.w_cosine,
            "w_graph": engine.w_graph,
        }
        self._conn.execute("""
            INSERT INTO engine_config (session_name, config_json)
            VALUES (?, ?)
            ON CONFLICT(session_name) DO UPDATE SET config_json = excluded.config_json
        """, (name, json.dumps(config)))

        self._conn.commit()

        return SessionMeta(
            name=name,
            created_at=now,
            updated_at=now,
            turn_count=engine.turn,
            mode=engine.mode,
            provider=engine.provider,
            model=engine.model,
            extraction_model=engine.extraction_model,
        )

    def load_session(self, name: str, engine) -> bool:
        """
        Restore engine state from a persisted session.

        Args:
            name: Session name to load.
            engine: PanicEngine instance to restore into.

        Returns:
            True if session was found and loaded, False if not found.
        """
        # Check session exists
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return False

        # Restore basic metadata
        engine.mode = row["mode"]
        engine.provider = row["provider"]
        engine.model = row["model"]
        engine.extraction_model = row["extraction_model"]
        engine.connected = bool(row["provider"])
        engine.turn = row["turn_count"]

        # Restore turn embeddings
        engine.turn_embeddings.clear()
        emb_rows = self._conn.execute(
            "SELECT turn, embedding FROM turn_embeddings WHERE session_name = ? ORDER BY turn",
            (name,),
        ).fetchall()
        for erow in emb_rows:
            engine.turn_embeddings[erow["turn"]] = np.frombuffer(
                erow["embedding"], dtype=np.float32
            ).copy()

        # Restore chat history
        engine.chat_history.clear()
        chat_rows = self._conn.execute(
            "SELECT turn, role, content, timestamp FROM chat_history "
            "WHERE session_name = ? ORDER BY seq",
            (name,),
        ).fetchall()
        for crow in chat_rows:
            engine.chat_history.append({
                "role": crow["role"],
                "content": crow["content"],
                "turn": crow["turn"],
                "timestamp": crow["timestamp"],
            })

        # Restore immediate buffer
        engine.immediate_buffer.clear()
        buf_rows = self._conn.execute(
            "SELECT turn, user_text, assistant_text FROM immediate_buffer "
            "WHERE session_name = ? ORDER BY seq",
            (name,),
        ).fetchall()
        for brow in buf_rows:
            engine.immediate_buffer.append({
                "turn": brow["turn"],
                "user": brow["user_text"],
                "assistant": brow["assistant_text"],
            })

        # Restore graphs
        self._load_graph(name, "rule", engine.rule_graph)
        self._load_graph(name, "llm", engine.llm_graph)

        # Restore engine config
        cfg_row = self._conn.execute(
            "SELECT config_json FROM engine_config WHERE session_name = ?", (name,)
        ).fetchone()
        if cfg_row:
            config = json.loads(cfg_row["config_json"])
            engine.buffer_size = config.get("buffer_size", 10)
            engine.llm_flush_interval = config.get("llm_flush_interval", 5)
            engine._turns_since_flush = config.get("turns_since_flush", 0)
            engine.item_blend = config.get("item_blend", 0.7)
            engine.w_cosine = config.get("w_cosine", 0.75)
            engine.w_graph = config.get("w_graph", 0.20)

        # Reinitialize LLM extractor with restored config
        engine._reinit_llm_extractor()

        return True

    def list_sessions(self) -> list[SessionMeta]:
        """List all persisted sessions."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            SessionMeta(
                name=r["name"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                turn_count=r["turn_count"],
                mode=r["mode"],
                provider=r["provider"],
                model=r["model"],
                extraction_model=r["extraction_model"],
                engine_version=r["engine_version"],
            )
            for r in rows
        ]

    def delete_session(self, name: str) -> bool:
        """Delete a persisted session."""
        row = self._conn.execute(
            "SELECT name FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return False

        for table in [
            "engine_config", "graph_snapshot", "immediate_buffer",
            "chat_history", "turn_embeddings", "reservoir_state", "sessions",
        ]:
            self._conn.execute(
                f"DELETE FROM {table} WHERE {'session_name' if table != 'sessions' else 'name'} = ?",
                (name,),
            )
        self._conn.commit()
        return True

    def session_exists(self, name: str) -> bool:
        """Check if a session exists."""
        row = self._conn.execute(
            "SELECT name FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def _save_graph(self, session_name: str, graph_type: str, graph):
        """Serialize a GraphStorage to JSON and save."""
        # Export all nodes
        all_nodes = graph.get_all_nodes(active_only=False)
        nodes_data = []
        for node in all_nodes:
            nodes_data.append({
                "id": node.id,
                "type": node.type.value,
                "data": node.data,
                "first_seen": node.first_seen,
                "last_seen": node.last_seen,
                "mention_count": node.mention_count,
                "status": node.status.value,
                "source": node.source,
            })

        # Export all edges
        edges_data = []
        seen_edges = set()
        for node in all_nodes:
            for edge in graph.get_edges_from(node.id):
                edge_key = (edge.source_id, edge.target_id, edge.type.value)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_data.append({
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "type": edge.type.value,
                        "weight": edge.weight,
                        "turn": edge.turn,
                    })

        # Export vector refs
        refs_data = []
        for node in all_nodes:
            for ref in graph.get_vector_refs(node.id):
                refs_data.append({
                    "node_id": ref.node_id,
                    "vector_index": ref.vector_index,
                    "turn": ref.turn,
                    "penalized": ref.penalized,
                })

        self._conn.execute("""
            INSERT INTO graph_snapshot (session_name, graph_type, nodes_json, edges_json, vector_refs_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_name, graph_type) DO UPDATE SET
                nodes_json = excluded.nodes_json,
                edges_json = excluded.edges_json,
                vector_refs_json = excluded.vector_refs_json
        """, (
            session_name, graph_type,
            json.dumps(nodes_data),
            json.dumps(edges_data),
            json.dumps(refs_data),
        ))

    def _load_graph(self, session_name: str, graph_type: str, graph):
        """Restore a GraphStorage from saved JSON."""
        from panic.graph.storage import (
            GraphNode, GraphEdge, VectorRef,
            NodeType, NodeStatus, EdgeType,
        )

        row = self._conn.execute(
            "SELECT * FROM graph_snapshot WHERE session_name = ? AND graph_type = ?",
            (session_name, graph_type),
        ).fetchone()
        if not row:
            return

        graph.clear()

        # Restore nodes
        nodes_data = json.loads(row["nodes_json"])
        for nd in nodes_data:
            node = GraphNode(
                id=nd["id"],
                type=NodeType(nd["type"]),
                data=nd.get("data", {}),
                first_seen=nd.get("first_seen", 0),
                last_seen=nd.get("last_seen", 0),
                mention_count=nd.get("mention_count", 1),
                status=NodeStatus(nd.get("status", "active")),
                source=nd.get("source", "user"),
            )
            graph.upsert_node(node)

            if nd.get("mention_count", 1) > 1:
                graph._conn.execute(
                    "UPDATE nodes SET mention_count = ? WHERE id = ?",
                    (nd["mention_count"], nd["id"]),
                )

        # Restore edges
        edges_data = json.loads(row["edges_json"])
        for ed in edges_data:
            edge = GraphEdge(
                source_id=ed["source_id"],
                target_id=ed["target_id"],
                type=EdgeType(ed["type"]),
                weight=ed.get("weight", 1.0),
                turn=ed.get("turn", 0),
            )
            graph.upsert_edge(edge)
            if ed.get("weight", 1.0) != 1.5:
                graph._conn.execute(
                    "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ? AND type = ?",
                    (ed["weight"], ed["source_id"], ed["target_id"], ed["type"]),
                )

        # Restore vector refs
        refs_data = json.loads(row["vector_refs_json"])
        for rd in refs_data:
            ref = VectorRef(
                node_id=rd["node_id"],
                vector_index=rd["vector_index"],
                turn=rd.get("turn", 0),
                penalized=rd.get("penalized", False),
            )
            graph.add_vector_ref(ref)

        graph._conn.commit()

    def close(self):
        """Close the database connection."""
        self._conn.close()
