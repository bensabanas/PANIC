"""Tests for PANIC session persistence."""

import pytest
import numpy as np
from panic.persistence import SessionStore, SessionMeta
from panic.api import PanicEngine


@pytest.fixture
def store():
    s = SessionStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def engine():
    e = PanicEngine()
    return e


class TestSessionStore:
    def test_list_empty(self, store):
        sessions = store.list_sessions()
        assert sessions == []

    def test_save_and_list(self, store, engine):
        meta = store.save_session("test-1", engine)
        assert meta.name == "test-1"
        assert meta.turn_count == 0

        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].name == "test-1"

    def test_session_exists(self, store, engine):
        assert not store.session_exists("nope")
        store.save_session("test-1", engine)
        assert store.session_exists("test-1")

    def test_delete_session(self, store, engine):
        store.save_session("test-1", engine)
        assert store.session_exists("test-1")

        deleted = store.delete_session("test-1")
        assert deleted is True
        assert not store.session_exists("test-1")

    def test_delete_nonexistent(self, store):
        assert store.delete_session("nope") is False

    def test_load_nonexistent(self, store, engine):
        found = store.load_session("nope", engine)
        assert found is False

    def test_save_load_roundtrip(self, store, engine):
        # Process some turns
        engine.process_turn("The deadline is March 15th.")
        engine.process_turn("John works on the PANIC project.")
        engine.process_turn("We decided to use PostgreSQL.")

        assert engine.turn == 3
        assert len(engine.chat_history) == 6  # 3 user + 3 assistant
        original_turn_count = engine.turn
        original_history_len = len(engine.chat_history)
        original_buffer_len = len(engine.immediate_buffer)

        # Save
        meta = store.save_session("roundtrip", engine)
        assert meta.turn_count == 3

        # Clear engine
        engine.clear()
        assert engine.turn == 0
        assert len(engine.chat_history) == 0

        # Load
        found = store.load_session("roundtrip", engine)
        assert found is True
        assert engine.turn == original_turn_count
        assert len(engine.chat_history) == original_history_len
        assert len(engine.immediate_buffer) == original_buffer_len

    def test_turn_embeddings_preserved(self, store, engine):
        engine.process_turn("First message.")
        engine.process_turn("Second message.")

        # Capture embeddings
        original_embs = {t: e.copy() for t, e in engine.turn_embeddings.items()}
        assert len(original_embs) >= 2

        store.save_session("emb-test", engine)
        engine.clear()
        engine.turn_embeddings.clear()

        store.load_session("emb-test", engine)

        for turn, orig_emb in original_embs.items():
            assert turn in engine.turn_embeddings
            np.testing.assert_array_almost_equal(
                engine.turn_embeddings[turn], orig_emb, decimal=5
            )

    def test_reservoir_state_preserved(self, store, engine):
        engine.process_turn("Building up reservoir state.")
        engine.process_turn("More state updates.")

        original_state = engine.reservoir.state.copy()
        original_turn = engine.reservoir.turn

        store.save_session("res-test", engine)
        engine.clear()

        store.load_session("res-test", engine)

        np.testing.assert_array_almost_equal(
            engine.reservoir.state, original_state, decimal=5
        )
        assert engine.reservoir.turn == original_turn

    def test_graph_preserved(self, store, engine):
        engine.process_turn("John Smith works at Google on the PANIC project.")
        engine.process_turn("The deadline for PANIC is March 2024.")

        rule_stats_before = engine.rule_graph.stats()
        assert rule_stats_before["nodes_active"] >= 1

        store.save_session("graph-test", engine)

        # Clear and reload
        engine.clear()
        assert engine.rule_graph.stats()["nodes_active"] == 0

        store.load_session("graph-test", engine)
        rule_stats_after = engine.rule_graph.stats()
        assert rule_stats_after["nodes_active"] == rule_stats_before["nodes_active"]

    def test_overwrite_session(self, store, engine):
        engine.process_turn("First save.")
        store.save_session("overwrite", engine)

        engine.process_turn("Second message.")
        engine.process_turn("Third message.")
        store.save_session("overwrite", engine)

        # Should have updated, not duplicated
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].turn_count == 3

    def test_multiple_sessions(self, store, engine):
        engine.process_turn("Session A message.")
        store.save_session("session-a", engine)

        engine.clear()
        engine.process_turn("Session B message 1.")
        engine.process_turn("Session B message 2.")
        store.save_session("session-b", engine)

        sessions = store.list_sessions()
        assert len(sessions) == 2

        # Load session A
        engine.clear()
        store.load_session("session-a", engine)
        assert engine.turn == 1

        # Load session B
        engine.clear()
        store.load_session("session-b", engine)
        assert engine.turn == 2

    def test_mode_preserved(self, store, engine):
        engine.set_mode("multi_session")
        engine.process_turn("Test.")
        store.save_session("mode-test", engine)

        engine.clear()
        engine.set_mode("long_conversation")

        store.load_session("mode-test", engine)
        assert engine.mode == "multi_session"

    def test_config_preserved(self, store, engine):
        engine.buffer_size = 20
        engine.llm_flush_interval = 10
        engine.process_turn("Test.")
        store.save_session("config-test", engine)

        engine.clear()
        engine.buffer_size = 10
        engine.llm_flush_interval = 5

        store.load_session("config-test", engine)
        assert engine.buffer_size == 20
        assert engine.llm_flush_interval == 10


class TestPersistenceAPI:
    """Test the FastAPI endpoints for session persistence."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from panic.api import app, engine as api_engine
        api_engine.clear()
        api_engine.connected = False
        api_engine.provider = ""
        api_engine.model = ""
        return TestClient(app)

    def test_save_session(self, client):
        client.post("/api/chat", json={"message": "Hello"})
        res = client.post("/api/session/save", json={"name": "api-test"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "saved"
        assert data["turn_count"] == 1

    def test_list_sessions(self, client):
        client.post("/api/chat", json={"message": "Hello"})
        client.post("/api/session/save", json={"name": "list-test"})

        res = client.get("/api/sessions")
        assert res.status_code == 200
        data = res.json()
        assert any(s["name"] == "list-test" for s in data)

    def test_load_session(self, client):
        client.post("/api/chat", json={"message": "Hello"})
        client.post("/api/session/save", json={"name": "load-test"})

        # Clear
        client.post("/api/session/clear")
        status = client.get("/api/status").json()
        assert status["turn"] == 0

        # Load
        res = client.post("/api/session/load", json={"name": "load-test"})
        assert res.status_code == 200
        assert res.json()["turn_count"] == 1

        status = client.get("/api/status").json()
        assert status["turn"] == 1

    def test_load_nonexistent(self, client):
        res = client.post("/api/session/load", json={"name": "nonexistent"})
        assert res.status_code == 404

    def test_delete_session(self, client):
        client.post("/api/chat", json={"message": "Hello"})
        client.post("/api/session/save", json={"name": "delete-test"})

        res = client.delete("/api/session/delete-test")
        assert res.status_code == 200

        # Should be gone
        res = client.post("/api/session/load", json={"name": "delete-test"})
        assert res.status_code == 404
