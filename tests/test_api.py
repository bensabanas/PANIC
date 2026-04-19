"""Tests for PANIC FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from panic.api import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset engine state before each test."""
    engine.clear()
    engine.connected = False
    engine.provider = ""
    engine.model = ""
    engine.set_mode("long_conversation")
    yield


@pytest.fixture
def client():
    return TestClient(app)


class TestRoot:
    def test_root_returns_something(self, client):
        res = client.get("/")
        assert res.status_code == 200


class TestConnect:
    def test_connect_llm(self, client):
        res = client.post("/api/connect", json={
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "test-key",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "connected"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"

    def test_connect_updates_status(self, client):
        client.post("/api/connect", json={"provider": "anthropic", "model": "claude-3"})
        res = client.get("/api/status")
        data = res.json()
        assert data["connected"] is True
        assert data["provider"] == "anthropic"


class TestChat:
    def test_chat_without_llm(self, client):
        """Chat without connected LLM should still work (returns placeholder)."""
        res = client.post("/api/chat", json={"message": "Hello, how are you?"})
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert data["turn"] == 1
        assert "panic_overhead_ms" in data

    def test_chat_increments_turn(self, client):
        client.post("/api/chat", json={"message": "First message"})
        res = client.post("/api/chat", json={"message": "Second message"})
        data = res.json()
        assert data["turn"] == 2

    def test_chat_empty_message_rejected(self, client):
        res = client.post("/api/chat", json={"message": ""})
        assert res.status_code == 400

    def test_chat_updates_graph(self, client):
        client.post("/api/chat", json={"message": "John Smith works at Google on the PANIC project."})
        res = client.get("/api/status")
        data = res.json()
        # Should have extracted some entities
        assert data["graph"]["nodes_active"] >= 1

    def test_chat_updates_buffer(self, client):
        client.post("/api/chat", json={"message": "Test message"})
        res = client.get("/api/status")
        data = res.json()
        assert data["buffer_size"] == 1

    def test_overhead_is_reasonable(self, client):
        res = client.post("/api/chat", json={"message": "What is the meaning of life?"})
        data = res.json()
        # PANIC overhead should be under 2 seconds (generous for first turn with model loading)
        assert data["panic_overhead_ms"] < 2000


class TestMode:
    def test_set_mode(self, client):
        res = client.post("/api/mode", json={"mode": "multi_session"})
        assert res.status_code == 200

        status = client.get("/api/status").json()
        assert status["mode"] == "multi_session"

    def test_invalid_mode(self, client):
        res = client.post("/api/mode", json={"mode": "invalid_mode"})
        assert res.status_code == 400

    def test_all_modes(self, client):
        for mode in ["long_conversation", "multi_session", "document_analysis"]:
            res = client.post("/api/mode", json={"mode": mode})
            assert res.status_code == 200


class TestStatus:
    def test_initial_status(self, client):
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()

        assert data["turn"] == 0
        assert data["connected"] is False
        assert data["mode"] == "long_conversation"
        assert "reservoir" in data
        assert "graph" in data

    def test_status_after_chat(self, client):
        client.post("/api/chat", json={"message": "Hello there."})
        res = client.get("/api/status")
        data = res.json()
        assert data["turn"] == 1
        assert data["buffer_size"] >= 1


class TestTransparency:
    def test_transparency_empty_initially(self, client):
        res = client.get("/api/transparency")
        assert res.status_code == 200
        data = res.json()
        assert data == {} or data == []  # empty before any chat

    def test_transparency_after_chat(self, client):
        client.post("/api/chat", json={"message": "Tell me about Python programming."})
        res = client.get("/api/transparency")
        assert res.status_code == 200
        data = res.json()

        assert "latency" in data
        assert "tokens_used" in data
        assert "graph_stats" in data


class TestSession:
    def test_clear_session(self, client):
        client.post("/api/chat", json={"message": "Test"})
        res = client.post("/api/session/clear")
        assert res.status_code == 200

        status = client.get("/api/status").json()
        assert status["turn"] == 0
        assert status["buffer_size"] == 0

    def test_export_empty(self, client):
        res = client.post("/api/session/export")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_export_after_chat(self, client):
        client.post("/api/chat", json={"message": "Hello"})
        res = client.post("/api/session/export")
        data = res.json()
        assert len(data) == 2  # user + assistant
        assert data[0]["role"] == "user"
        assert data[1]["role"] == "assistant"


class TestHistory:
    def test_history_empty(self, client):
        res = client.get("/api/history")
        assert res.status_code == 200
        assert res.json() == []

    def test_history_after_messages(self, client):
        client.post("/api/chat", json={"message": "First"})
        client.post("/api/chat", json={"message": "Second"})
        res = client.get("/api/history")
        data = res.json()
        assert len(data) == 4  # 2 user + 2 assistant
