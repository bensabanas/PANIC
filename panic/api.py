"""
PANIC FastAPI Backend — v5 Simplified (No Reservoir)

REST API serving the PANIC pipeline:
  - POST /chat — send a message, get PANIC-enhanced LLM response
  - POST /connect — connect an LLM provider
  - POST /mode — switch operating mode
  - GET /status — PANIC state health
  - GET /transparency — what PANIC fed to the LLM last turn
  - POST /session/end — end session (triggers extraction pipeline)
  - POST /session/clear — clear session
  - Profile management (create, switch, list, delete, clone)

Retrieval engine: embeddings + dual graph (rule-based + LLM extraction)
  - Cosine similarity with 0.7 item / 0.3 turn blended embeddings
  - Rule-based graph: sparse, precise contradiction detection
  - LLM graph: dense, high-recall multi-hop connectivity
  - Graph boost blended by query type classifier
  - No reservoir, no readout, no reranker, no pattern separator
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from panic.encoder.encoder import PanicEncoder
from panic.graph.storage import GraphStorage, NodeType, NodeStatus
from panic.graph.extractors import ExtractorPipeline
from panic.graph.llm_extractors import LLMExtractorPipeline, LLMExtractorConfig
from panic.translation.translator import Translator, PromptConfig, ContextItem
from panic.translation.context_formatter import ContextFormatter
from panic.profiles import ProfileManager, ProfileConfig

logger = logging.getLogger("panic.api")


# --- Multi-hop query classifier ---

# Keywords that indicate the user is asking about relationships between facts,
# not just a single fact lookup. Used to blend rule vs LLM graph boost weights.
MULTI_HOP_SIGNALS = {
    "together", "both", "combined", "imply", "implications", "relationship",
    "connect", "connected", "related", "between", "affect", "impact",
    "considering", "along with", "in light of", "given that", "taken together",
    "how does", "what does", "mean for", "suggest about",
}


def is_multi_hop_query(query: str) -> float:
    """
    Score how likely a query is multi-hop (0.0 = single fact, 1.0 = clearly multi-hop).
    Returns a blend weight: higher = lean more on LLM graph.
    """
    q_lower = query.lower()
    signal_count = sum(1 for s in MULTI_HOP_SIGNALS if s in q_lower)

    words = query.split()
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0

    if signal_count >= 2:
        return 0.9
    elif signal_count >= 1 and capitalized >= 2:
        return 0.7
    elif signal_count >= 1:
        return 0.5
    elif capitalized >= 2:
        return 0.3
    else:
        return 0.0


# --- Request/Response models ---

class ConnectRequest(BaseModel):
    provider: str  # "openai", "anthropic", "local"
    model: str
    api_key: str = ""
    base_url: str = ""
    extraction_model: str = ""  # e.g. "claude-haiku-4-5-20251001"
    extraction_api_key: str = ""


class ChatRequest(BaseModel):
    message: str


class ModeRequest(BaseModel):
    mode: str  # "long_conversation", "multi_session", "document_analysis"


class SessionInfo(BaseModel):
    name: str = "default"


# --- PANIC Engine (v5 Simplified) ---

class PanicEngine:
    """
    Core PANIC engine — v5 simplified.

    Retrieval uses embeddings + dual graph only:
      - Cosine similarity on blended embeddings (0.7 item / 0.3 turn)
      - Rule graph: sparse, precise — handles contradiction detection
      - LLM graph: dense, high recall — handles multi-hop connectivity
      - Graph boost blended by query type
      - Recency tiebreaker
      - Multi-hop expansion

    No reservoir, readout, reranker, pattern separator, or snapshots.
    """

    def __init__(self):
        self.encoder: Optional[PanicEncoder] = None

        # Dual graph stores
        self.rule_graph: Optional[GraphStorage] = None
        self.llm_graph: Optional[GraphStorage] = None
        self.rule_extractor: Optional[ExtractorPipeline] = None
        self.llm_extractor: Optional[LLMExtractorPipeline] = None

        self.translator: Optional[Translator] = None

        self.mode: str = "long_conversation"
        self.connected: bool = False
        self.provider: str = ""
        self.model: str = ""
        self.api_key: str = ""
        self.base_url: str = ""
        self.extraction_model: str = "claude-haiku-4-5-20251001"
        self.extraction_api_key: str = ""

        # Scoring weights (validated by eval harness — don't change without re-evaluating)
        # item_blend: ratio of item embedding vs turn embedding in blended candidates
        #   0.7 item / 0.3 turn was the single biggest improvement found (+28pp multi-hop)
        self.item_blend: float = 0.7
        # w_cosine + w_graph = 0.95, remaining 0.05 goes to recency tiebreaker
        self.w_cosine: float = 0.75   # cosine similarity weight (primary signal)
        self.w_graph: float = 0.20    # graph connectivity boost weight

        # Conversation state
        self.immediate_buffer: list[dict] = []
        self.buffer_size: int = 10
        self.turn: int = 0
        self.last_transparency: dict = {}
        self.chat_history: list[dict] = []

        # Per-turn embedding cache for candidate scoring
        self.turn_embeddings: dict[int, np.ndarray] = {}

        # LLM extraction flush interval (flush every N turns)
        self.llm_flush_interval: int = 5
        self._turns_since_flush: int = 0

        self._initialize()

    def _initialize(self):
        """Initialize all PANIC components."""
        self.encoder = PanicEncoder(device="cpu")

        # Dual graph stores
        self.rule_graph = GraphStorage(":memory:")
        self.llm_graph = GraphStorage(":memory:")

        self.rule_extractor = ExtractorPipeline(self.rule_graph)

        # LLM extractor with rule-based fallback on the LLM graph
        llm_rule_fallback = ExtractorPipeline(self.llm_graph)
        llm_config = LLMExtractorConfig(
            model=self.extraction_model,
            batch_size=20,
            temperature=0.0,
        )
        self.llm_extractor = LLMExtractorPipeline(
            self.llm_graph, config=llm_config, rule_fallback=llm_rule_fallback
        )

        self.translator = Translator(PromptConfig())

    def connect_llm(self, provider: str, model: str, api_key: str = "",
                    base_url: str = "", extraction_model: str = "",
                    extraction_api_key: str = ""):
        """Connect to an LLM provider."""
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.connected = True

        if extraction_model:
            self.extraction_model = extraction_model
        if extraction_api_key:
            self.extraction_api_key = extraction_api_key

        if extraction_model or extraction_api_key:
            self._reinit_llm_extractor()

    def _reinit_llm_extractor(self):
        """Reinitialize the LLM extractor with current config."""
        llm_rule_fallback = ExtractorPipeline(self.llm_graph)
        llm_config = LLMExtractorConfig(
            model=self.extraction_model,
            batch_size=20,
            temperature=0.0,
        )
        self.llm_extractor = LLMExtractorPipeline(
            self.llm_graph, config=llm_config, rule_fallback=llm_rule_fallback
        )

    def set_mode(self, mode: str):
        """Switch operating mode."""
        if mode not in ("long_conversation", "multi_session", "document_analysis"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

    def process_turn(self, user_message: str) -> dict:
        """
        Process a user message through the simplified PANIC pipeline.
        Returns dict with response and metadata.
        """
        start_time = time.perf_counter()
        self.turn += 1

        # 1. Encode user input
        t0 = time.perf_counter()
        user_embedding = self.encoder.encode(user_message)
        self.turn_embeddings[self.turn] = user_embedding
        encode_ms = (time.perf_counter() - t0) * 1000

        # 2. Extract graph updates (both engines)
        t0 = time.perf_counter()
        self.rule_extractor.extract_and_apply(user_message, self.turn, source="user")
        self.llm_extractor.add_turn(user_message, self.turn, source="user")
        self._turns_since_flush += 1

        # LLM extraction batches turns and sends them in one API call.
        # Flush every N turns to keep the graph up to date.
        if self._turns_since_flush >= self.llm_flush_interval:
            try:
                self.llm_extractor.flush_and_apply()
            except Exception as e:
                logger.warning(f"LLM extraction flush failed: {e}")
            self._turns_since_flush = 0

        extract_ms = (time.perf_counter() - t0) * 1000

        # 3. Retrieval scoring (working memory — current session graph)
        t0 = time.perf_counter()
        retrieved_items = self._retrieve(user_message, user_embedding)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # 4. Build layered prompt
        t0 = time.perf_counter()
        immediate_items = self._get_immediate_items()

        # Enrich working memory items with conversation-aware metadata
        formatter = ContextFormatter(
            rule_graph=self.rule_graph,
            chat_history=self.chat_history,
            total_turns=self.turn,
        )
        formatted_working = formatter.format_items(retrieved_items, query=user_message)

        # Query persistent memory layers from profile
        from panic.translation.translator import MemoryLayer
        layers = self._build_memory_layers(user_message, user_embedding, formatted_working)

        prompt_result = self.translator.construct_layered_prompt(
            query=user_message,
            layers=layers,
            immediate_buffer=immediate_items,
        )
        translate_ms = (time.perf_counter() - t0) * 1000

        # 5. Call LLM
        t0 = time.perf_counter()
        llm_response = self._call_llm(prompt_result.full_prompt)
        llm_ms = (time.perf_counter() - t0) * 1000

        # 6. Process LLM response back through PANIC
        # Skip error responses (they start with "[") — don't pollute the graph
        if llm_response and not llm_response.startswith("["):
            response_embedding = self.encoder.encode(llm_response)
            self.turn_embeddings[self.turn] = response_embedding  # Update with response context
            self.rule_extractor.extract_and_apply(llm_response, self.turn, source="llm")
            self.llm_extractor.add_turn(llm_response, self.turn, source="llm")
            confusion = self.translator.analyze_response(llm_response)
        else:
            confusion = []

        # 7. Update immediate buffer
        self.immediate_buffer.append({
            "turn": self.turn,
            "user": user_message,
            "assistant": llm_response,
        })
        if len(self.immediate_buffer) > self.buffer_size:
            self.immediate_buffer.pop(0)

        # 8. Store in chat history
        self.chat_history.append({"role": "user", "content": user_message, "turn": self.turn})
        self.chat_history.append({"role": "assistant", "content": llm_response, "turn": self.turn})

        total_ms = (time.perf_counter() - start_time) * 1000

        # Build transparency data
        rule_stats = self.rule_graph.stats()
        llm_stats = self.llm_graph.stats()
        self.last_transparency = {
            "engine": "simplified_v5",
            "layers": {
                l.name: {"items": len(l.items), "budget": l.budget}
                for l in layers
            } if layers else {},
            "working_retrieved_count": len(retrieved_items),
            "working_retrieved_items": [
                {"id": c.id, "text": c.text[:100], "score": c.relevance_score}
                for c in retrieved_items[:10]
            ],
            "immediate_turns": len(immediate_items),
            "tokens_used": prompt_result.token_usage,
            "rule_graph_stats": rule_stats,
            "llm_graph_stats": llm_stats,
            "query_multi_hop_score": is_multi_hop_query(user_message),
            "confusion_signals": [{"type": s.type, "detail": s.detail} for s in confusion],
            "latency": {
                "encode_ms": round(encode_ms, 1),
                "extract_ms": round(extract_ms, 1),
                "retrieval_ms": round(retrieval_ms, 1),
                "translate_ms": round(translate_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "total_panic_ms": round(total_ms - llm_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        }

        result = {
            "response": llm_response,
            "turn": self.turn,
            "panic_overhead_ms": round(total_ms - llm_ms, 1),
        }

        # Periodic intermediate extraction (~every 50 turns)
        if self.turn > 0 and self.turn % 50 == 0 and self.connected:
            try:
                from panic.extraction import ExtractionPipeline
                extractor = ExtractionPipeline(
                    profile_manager=_get_profile_manager(),
                    engine=self,
                )
                ext_result = extractor.run_intermediate()
                logger.info(f"Intermediate extraction at turn {self.turn}: {ext_result.get('status')}")
            except Exception as e:
                logger.warning(f"Intermediate extraction failed at turn {self.turn}: {e}")

        return result

    # --- Memory layer construction ---

    def _build_memory_layers(
        self,
        query: str,
        query_embedding: np.ndarray,
        working_items: list[ContextItem],
    ) -> list:
        """
        Build separate memory layers for layered context injection.

        Queries each persistent memory source (episodic, semantic, procedural)
        from the active profile's markdown files, scores them against the query,
        and returns MemoryLayer objects with per-layer token budgets.

        Working memory items come from the graph-based retrieval pipeline.
        """
        from panic.translation.translator import MemoryLayer

        try:
            pm = _get_profile_manager()
            cfg = pm.get_config()
        except Exception:
            # Fallback: single layer with all working items if profiles unavailable
            return [MemoryLayer(
                name="working", label="Current Session",
                items=working_items, budget=3800, priority=10,
            )]

        layers = []

        # Layer 1: Semantic memory (entities + facts + preferences)
        semantic_items = self._query_semantic_memory(query, query_embedding, pm)
        if semantic_items:
            layers.append(MemoryLayer(
                name="semantic", label="Long-term Knowledge",
                items=semantic_items, budget=cfg.budget_semantic, priority=8,
            ))

        # Layer 2: Episodic memory (past session summaries)
        episodic_items = self._query_episodic_memory(query, query_embedding, pm)
        if episodic_items:
            layers.append(MemoryLayer(
                name="episodic", label="Previous Sessions",
                items=episodic_items, budget=cfg.budget_episodic, priority=6,
            ))

        # Layer 3: Working memory (current session graph-retrieved items)
        if working_items:
            layers.append(MemoryLayer(
                name="working", label="Current Session Context",
                items=working_items, budget=cfg.budget_working, priority=10,
            ))

        # Layer 4: Procedural memory (workflows + failures)
        procedural_items = self._query_procedural_memory(query, query_embedding, pm)
        if procedural_items:
            layers.append(MemoryLayer(
                name="procedural", label="Relevant Patterns",
                items=procedural_items, budget=cfg.budget_procedural, priority=4,
            ))

        return layers

    def _query_semantic_memory(
        self, query: str, query_embedding: np.ndarray, pm
    ) -> list[ContextItem]:
        """Query semantic memory files (entities, facts, preferences) by embedding similarity."""
        chunks = []
        for file in ("entities.md", "facts.md", "preferences.md"):
            content = pm.read_semantic(file=file)
            if not content:
                continue
            # Split into sections by ## headings
            for chunk in self._split_markdown_sections(content, source_file=file):
                chunks.append(chunk)

        return self._score_and_rank_chunks(chunks, query_embedding, max_items=15)

    def _query_episodic_memory(
        self, query: str, query_embedding: np.ndarray, pm
    ) -> list[ContextItem]:
        """Query episodic memory (past session episode files) by embedding similarity."""
        chunks = []
        episodes = pm.list_episodes()
        # Read last 10 episodes (most recent first)
        for date in episodes[-10:]:
            content = pm.read_episode(date=date)
            if not content:
                continue
            # Each episode is one chunk (they're already summarized)
            chunks.append(ContextItem(
                id=f"episode_{date}",
                text=content[:1000],  # cap individual episode size
                source_turn=0,
                tier="episodic",
            ))

        return self._score_and_rank_chunks(chunks, query_embedding, max_items=5)

    def _query_procedural_memory(
        self, query: str, query_embedding: np.ndarray, pm
    ) -> list[ContextItem]:
        """Query procedural memory (workflows, failures) by embedding similarity."""
        chunks = []
        for file in ("workflows.md", "failures.md"):
            content = pm.read_procedural(file=file)
            if not content:
                continue
            for chunk in self._split_markdown_sections(content, source_file=file):
                chunks.append(chunk)

        return self._score_and_rank_chunks(chunks, query_embedding, max_items=8)

    def _split_markdown_sections(self, content: str, source_file: str = "") -> list[ContextItem]:
        """Split markdown content into sections by ## headings."""
        items = []
        current_heading = ""
        current_lines = []

        for line in content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_lines and current_heading:
                    text = "\n".join(current_lines).strip()
                    if text and len(text) > 10:  # skip trivially short sections
                        items.append(ContextItem(
                            id=f"semantic_{source_file}_{current_heading[:30]}",
                            text=f"{current_heading}\n{text}",
                            source_turn=0,
                            tier="semantic",
                        ))
                current_heading = line
                current_lines = []
            elif line.startswith("# ") and not line.startswith("## "):
                # Top-level heading, skip (it's the file title)
                continue
            else:
                current_lines.append(line)

        # Last section
        if current_lines and current_heading:
            text = "\n".join(current_lines).strip()
            if text and len(text) > 10:
                items.append(ContextItem(
                    id=f"semantic_{source_file}_{current_heading[:30]}",
                    text=f"{current_heading}\n{text}",
                    source_turn=0,
                    tier="semantic",
                ))

        return items

    def _score_and_rank_chunks(
        self, chunks: list[ContextItem], query_embedding: np.ndarray, max_items: int = 10
    ) -> list[ContextItem]:
        """Score chunks by cosine similarity and return top-N."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        chunk_embeddings = self.encoder.encode_batch(texts)

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = chunk_embeddings @ q_norm

        # Rank and take top-N
        top_indices = np.argsort(scores)[::-1][:max_items]

        result = []
        for i in top_indices:
            if scores[i] < 0.1:  # minimum relevance threshold
                continue
            chunk = chunks[i]
            chunk.relevance_score = float(scores[i])
            result.append(chunk)

        return result

    # --- Simplified retrieval (v5) ---

    def _retrieve(self, query: str, query_embedding: np.ndarray) -> list[ContextItem]:
        """
        Score and rank candidates using the v5 simplified approach:
          1. Cosine similarity on blended embeddings (primary signal, 0.75)
          2. Graph boost blended by query type (0.20)
          3. Contradiction penalty (rule graph)
          4. Recency tiebreaker (0.05)
          5. Multi-hop expansion
        """
        candidates = self._build_candidates(self.rule_graph)
        if not candidates:
            return []

        n_candidates = len(candidates)
        candidate_texts = [c.text for c in candidates]
        candidate_embeddings = self.encoder.encode_batch(candidate_texts)

        # Blended candidate embeddings: mix item text embedding with turn context embedding.
        # This preserves the needle signal (item text) while adding conversational context.
        # 0.7/0.3 ratio found via experiment — higher item weight prevents turn noise
        # from diluting precision on cosine similarity matches.
        turn_weight = 1.0 - self.item_blend
        blended_embeddings = []
        for i, c in enumerate(candidates):
            if c.source_turn in self.turn_embeddings:
                turn_emb = self.turn_embeddings[c.source_turn]
                blended = self.item_blend * candidate_embeddings[i] + turn_weight * turn_emb
                blended = blended / (np.linalg.norm(blended) + 1e-8)  # re-normalize
                blended_embeddings.append(blended)
            else:
                blended_embeddings.append(candidate_embeddings[i])  # no turn context available
        c_embs = np.stack(blended_embeddings)

        adaptive_top_k = min(n_candidates, max(10, n_candidates // 5))
        multi_hop_score = is_multi_hop_query(query)

        # --- 1. Cosine similarity ---
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cosine_scores = c_embs @ q_norm

        # --- 2. Graph boost (blended from both engines) ---
        rule_boosts = self._compute_graph_boost(self.rule_graph, query, query_embedding, candidates, hops=2)
        llm_boosts = self._compute_graph_boost(self.llm_graph, query, query_embedding, candidates, hops=1)

        graph_scores = np.zeros(n_candidates)
        for i in range(n_candidates):
            graph_scores[i] = (1.0 - multi_hop_score) * rule_boosts[i] + multi_hop_score * llm_boosts[i]

        # --- 3. Contradiction penalty (rule-based graph only) ---
        contradiction_penalty = np.zeros(n_candidates)
        for i, c in enumerate(candidates):
            node_ids = self.rule_graph.get_nodes_for_vector(c.source_turn)
            for node_id in node_ids:
                node = self.rule_graph.get_node(node_id)
                if node and node.status in (NodeStatus.SUPERSEDED, NodeStatus.CONTRADICTED):
                    contradiction_penalty[i] = max(contradiction_penalty[i], 0.3)
                    break

        # --- 4. Recency tiebreaker ---
        recency_boost = np.array([c.source_turn / max(1, self.turn) for c in candidates])
        recency_boost = recency_boost * 0.05

        # --- 5. Combine ---
        combined_scores = (
            self.w_cosine * cosine_scores +
            self.w_graph * graph_scores +
            recency_boost
            - contradiction_penalty
        )

        # --- 6. Multi-hop expansion ---
        # Take the top-8 candidates and boost anything connected to them in the graph.
        # For multi-hop queries, use the LLM graph (denser, higher recall).
        # For single-fact queries, use the rule graph (sparser, more precise).
        expand_graph = self.llm_graph if multi_hop_score > 0.3 else self.rule_graph
        expand_hops = 1 if multi_hop_score > 0.3 else 2  # fewer hops for denser graph
        top_k_expand = min(8, len(combined_scores))
        top_indices = np.argsort(combined_scores)[::-1][:top_k_expand]

        for i in top_indices:
            c = candidates[i]
            node_ids = expand_graph.get_nodes_for_vector(c.source_turn)
            for node_id in node_ids:
                connected = expand_graph.get_connected_nodes(node_id, max_hops=expand_hops)
                for conn_id in connected:
                    conn_refs = expand_graph.get_vector_refs(conn_id)
                    for ref in conn_refs:
                        # Boost candidates that are graph-connected to top results
                        for j, cc in enumerate(candidates):
                            if cc.source_turn == ref.vector_index:
                                combined_scores[j] += 0.1  # flat boost per connection
                                break

        # --- 7. Select top-K and assign scores ---
        top_k = min(adaptive_top_k, len(combined_scores))
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        retrieved = []
        for i in top_indices:
            c = candidates[i]
            c.relevance_score = float(combined_scores[i])
            retrieved.append(c)

        return retrieved

    def _build_candidates(self, graph: GraphStorage) -> list[ContextItem]:
        """Build candidate context items from a graph store."""
        candidates = []
        nodes = graph.get_all_nodes(active_only=True)

        for node in nodes:
            text = ""
            if node.type == NodeType.FACT:
                text = node.data.get("statement", "")
            elif node.type == NodeType.DECISION:
                text = node.data.get("statement", "")
            elif node.type == NodeType.ENTITY:
                name = node.data.get("name", "")
                props = {k: v for k, v in node.data.items()
                         if k not in ("name", "spacy_label", "category")}
                text = f"{name}: {json.dumps(props)}" if props else name

            if text:
                candidates.append(ContextItem(
                    id=node.id,
                    text=text,
                    source_turn=node.first_seen,
                    tier="cold",
                ))

        return candidates

    def _compute_graph_boost(
        self, graph: GraphStorage, query_text: str, query_emb: np.ndarray,
        candidates: list[ContextItem], hops: int = 2,
    ) -> np.ndarray:
        """Compute per-candidate boost from a specific graph."""
        boosts = np.zeros(len(candidates))

        query_words = set(query_text.lower().split())
        stopwords = {"what", "is", "the", "a", "an", "for", "of", "to", "in", "on",
                     "are", "was", "were", "do", "does", "did", "how", "when", "where",
                     "who", "which", "that", "this", "we", "our", "using", "about",
                     "earlier", "conversation", "mentioned", "considering", "imply",
                     "together", "these", "and", "with", "has", "have", "been"}
        content_words = query_words - stopwords

        # Find matching graph nodes
        matched_node_ids = set()
        for word in content_words:
            if len(word) < 3:
                continue
            nodes = graph.search_nodes(word)
            for node in nodes:
                matched_node_ids.add(node.id)

        # Bigram phrases
        content_list = list(content_words)
        for i in range(len(content_list)):
            for j in range(i + 1, min(i + 3, len(content_list))):
                phrase = f"{content_list[i]} {content_list[j]}"
                nodes = graph.search_nodes(phrase)
                for node in nodes:
                    matched_node_ids.add(node.id)

        # Expand via edges
        expanded_ids = set(matched_node_ids)
        for node_id in matched_node_ids:
            connected = graph.get_connected_nodes(node_id, max_hops=hops)
            expanded_ids.update(connected)

        # Map graph nodes back to candidate indices via vector_refs.
        # vector_refs link graph nodes to the turn they were extracted from.
        turn_to_idx = {}
        for i, c in enumerate(candidates):
            turn_to_idx.setdefault(c.source_turn, []).append(i)

        raw_boosts = {}
        for node_id in expanded_ids:
            refs = graph.get_vector_refs(node_id)
            for ref in refs:
                t = ref.vector_index
                if t in turn_to_idx:
                    # Direct keyword matches get full weight; graph-expanded get half
                    weight = 1.0 if node_id in matched_node_ids else 0.5
                    node = graph.get_node(node_id)
                    # Entities mentioned multiple times get a small boost (capped at 1.5x)
                    if node and node.mention_count > 1:
                        weight *= min(1.5, 1.0 + node.mention_count * 0.1)
                    raw_boosts[t] = raw_boosts.get(t, 0.0) + weight

        # Normalize and assign
        if raw_boosts:
            max_boost = max(raw_boosts.values())
            if max_boost > 0:
                for t, score in raw_boosts.items():
                    normalized = score / max_boost
                    for idx in turn_to_idx.get(t, []):
                        boosts[idx] = normalized

        return boosts

    def _get_immediate_items(self) -> list[ContextItem]:
        """Convert immediate buffer to ContextItems."""
        items = []
        for entry in self.immediate_buffer:
            text = f"User: {entry['user']}\nAssistant: {entry['assistant']}"
            items.append(ContextItem(
                id=f"buffer_{entry['turn']}",
                text=text,
                source_turn=entry["turn"],
                tier="immediate",
            ))
        return items

    def _call_llm(self, prompt: str) -> str:
        """Call the connected LLM. Returns response text."""
        if not self.connected:
            return "[No LLM connected. Connect one in the settings panel to get responses.]"

        try:
            import litellm
            litellm.drop_params = True

            kwargs = {
                "model": f"{self.provider}/{self.model}" if self.provider != "openai" else self.model,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["api_base"] = self.base_url

            response = litellm.completion(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def get_status(self, profile_name: str = "") -> dict:
        """Return current PANIC state."""
        rule_stats = self.rule_graph.stats() if self.rule_graph else {}
        llm_stats = self.llm_graph.stats() if self.llm_graph else {}
        return {
            "engine": "simplified_v5",
            "profile": profile_name,
            "mode": self.mode,
            "turn": self.turn,
            "connected": self.connected,
            "provider": self.provider,
            "model": self.model,
            "extraction_model": self.extraction_model,
            "buffer_size": len(self.immediate_buffer),
            "buffer_max": self.buffer_size,
            "rule_graph": rule_stats,
            "llm_graph": llm_stats,
        }

    def clear(self):
        """Reset everything."""
        self.turn = 0
        self.immediate_buffer.clear()
        self.chat_history.clear()
        self.last_transparency.clear()
        self.turn_embeddings.clear()
        self._turns_since_flush = 0
        self.rule_graph.clear()
        self.llm_graph.clear()

    def export_chat(self) -> list[dict]:
        return list(self.chat_history)


# --- FastAPI App ---

app = FastAPI(title="PANIC", description="Don't Panic — Persistent memory for AI assistants (v5 simplified)")

# Allow dashboard embeds to access the sidecar API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PanicEngine()
profile_manager = ProfileManager()


def _get_profile_manager() -> ProfileManager:
    """Module-level accessor for the profile manager (used by engine internals)."""
    return profile_manager

# Load active profile into engine at startup
try:
    profile_manager.switch(profile_manager.active_profile, engine)
    logger.info(f"Loaded profile: {profile_manager.active_profile}")
except Exception as e:
    logger.warning(f"Failed to load profile '{profile_manager.active_profile}': {e}")

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "PANIC API is running (v5 simplified). Frontend not found at /frontend/index.html"}


@app.post("/api/connect")
async def connect(req: ConnectRequest):
    engine.connect_llm(
        req.provider, req.model, req.api_key, req.base_url,
        extraction_model=req.extraction_model,
        extraction_api_key=req.extraction_api_key,
    )
    return {"status": "connected", "provider": req.provider, "model": req.model,
            "extraction_model": engine.extraction_model}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    result = engine.process_turn(req.message)
    return result


@app.post("/api/mode")
async def set_mode(req: ModeRequest):
    try:
        engine.set_mode(req.mode)
        return {"status": "ok", "mode": req.mode}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/status")
async def status():
    return engine.get_status(profile_name=profile_manager.active_profile)


@app.get("/api/transparency")
async def transparency():
    return engine.last_transparency


@app.get("/api/history")
async def history():
    return engine.chat_history


@app.post("/api/session/clear")
async def clear_session():
    engine.clear()
    return {"status": "cleared", "profile": profile_manager.active_profile}


@app.post("/api/session/export")
async def export_session():
    return engine.export_chat()


@app.post("/api/session/end")
async def end_session():
    """End the current session: run extraction pipeline, save state."""
    from panic.extraction import ExtractionPipeline

    result = {"profile": profile_manager.active_profile, "turn_count": engine.turn}

    # Run extraction pipeline if there's conversation data
    if engine.turn > 0 and engine.connected:
        try:
            extractor = ExtractionPipeline(
                profile_manager=profile_manager,
                engine=engine,
            )
            extraction_result = extractor.run_session_end()
            result["extraction"] = extraction_result
        except Exception as e:
            logger.warning(f"Extraction pipeline failed: {e}")
            result["extraction"] = {"status": "failed", "error": str(e)}

    # Save profile state
    try:
        profile_manager.save_state(profile_manager.active_profile, engine)
        result["status"] = "saved"
    except Exception as e:
        logger.warning(f"Profile save failed: {e}")
        result["status"] = "save_failed"
        result["error"] = str(e)

    return result


# --- Profile endpoints ---

class ProfileCreateRequest(BaseModel):
    name: str


class ProfileSwitchRequest(BaseModel):
    name: str


class ProfileCloneRequest(BaseModel):
    source: str
    target: str


class ProfileConfigUpdateRequest(BaseModel):
    budget_semantic: Optional[int] = None
    budget_episodic: Optional[int] = None
    budget_working: Optional[int] = None
    budget_procedural: Optional[int] = None
    item_blend: Optional[float] = None
    w_cosine: Optional[float] = None
    w_graph: Optional[float] = None
    buffer_size: Optional[int] = None
    extraction_model: Optional[str] = None


@app.get("/api/profiles")
async def list_profiles():
    """List all profiles with summary info."""
    profiles = profile_manager.list_profiles()
    return {
        "active": profile_manager.active_profile,
        "profiles": [
            {
                "name": p.name,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "turn_count": p.turn_count,
                "episode_count": p.episode_count,
                "entity_count": p.entity_count,
                "is_active": p.name == profile_manager.active_profile,
            }
            for p in profiles
        ],
    }


@app.post("/api/profiles")
async def create_profile(req: ProfileCreateRequest):
    """Create a new profile."""
    try:
        cfg = profile_manager.create(req.name)
        return {"status": "created", "profile": cfg.name}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/profiles/switch")
async def switch_profile(req: ProfileSwitchRequest):
    """Switch the active profile."""
    try:
        cfg = profile_manager.switch(req.name, engine)
        return {
            "status": "switched",
            "profile": cfg.name,
            "turn_count": engine.turn,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/profiles/clone")
async def clone_profile(req: ProfileCloneRequest):
    """Clone an existing profile."""
    try:
        cfg = profile_manager.clone(req.source, req.target)
        return {"status": "cloned", "source": req.source, "target": cfg.name}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/profiles/{name}")
async def get_profile_config(name: str):
    """Get a profile's configuration."""
    try:
        cfg = profile_manager.get_config(name)
        return {
            "name": cfg.name,
            "created_at": cfg.created_at,
            "updated_at": cfg.updated_at,
            "turn_count": cfg.turn_count,
            "extraction_model": cfg.extraction_model,
            "item_blend": cfg.item_blend,
            "w_cosine": cfg.w_cosine,
            "w_graph": cfg.w_graph,
            "budget_semantic": cfg.budget_semantic,
            "budget_episodic": cfg.budget_episodic,
            "budget_working": cfg.budget_working,
            "budget_procedural": cfg.budget_procedural,
            "buffer_size": cfg.buffer_size,
            "llm_flush_interval": cfg.llm_flush_interval,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.patch("/api/profiles/{name}")
async def update_profile_config(name: str, req: ProfileConfigUpdateRequest):
    """Update a profile's configuration."""
    try:
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")
        cfg = profile_manager.update_config(name, **updates)
        # If updating the active profile, apply changes to engine
        if name == profile_manager.active_profile:
            for k, v in updates.items():
                if hasattr(engine, k):
                    setattr(engine, k, v)
        return {"status": "updated", "profile": cfg.name}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/profiles/{name}")
async def delete_profile(name: str):
    """Delete a profile."""
    try:
        deleted = profile_manager.delete(name)
        if not deleted:
            raise HTTPException(404, f"Profile '{name}' not found")
        return {"status": "deleted", "profile": name}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/profiles/{name}/episodes")
async def list_episodes(name: str):
    """List episode dates for a profile."""
    try:
        episodes = profile_manager.list_episodes(name)
        return {"profile": name, "episodes": episodes}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/profiles/{name}/episodes/{date}")
async def get_episode(name: str, date: str):
    """Read an episode file."""
    content = profile_manager.read_episode(name=name, date=date)
    if not content:
        raise HTTPException(404, f"No episode for {date} in profile '{name}'")
    return {"profile": name, "date": date, "content": content}


@app.get("/api/profiles/{name}/semantic/{file}")
async def get_semantic(name: str, file: str):
    """Read a semantic memory file (entities, facts, preferences)."""
    if file not in ("entities.md", "facts.md", "preferences.md"):
        raise HTTPException(400, f"Unknown semantic file: {file}")
    content = profile_manager.read_semantic(name=name, file=file)
    return {"profile": name, "file": file, "content": content}


@app.get("/api/profiles/{name}/procedural/{file}")
async def get_procedural(name: str, file: str):
    """Read a procedural memory file (workflows, failures)."""
    if file not in ("workflows.md", "failures.md"):
        raise HTTPException(400, f"Unknown procedural file: {file}")
    content = profile_manager.read_procedural(name=name, file=file)
    return {"profile": name, "file": file, "content": content}


class MemoryFileWriteRequest(BaseModel):
    content: str


@app.put("/api/profiles/{name}/semantic/{file}")
async def put_semantic(name: str, file: str, req: MemoryFileWriteRequest):
    """Write a semantic memory file (entities, facts, preferences)."""
    if file not in ("entities.md", "facts.md", "preferences.md"):
        raise HTTPException(400, f"Unknown semantic file: {file}")
    profile_manager.write_semantic(req.content, name=name, file=file)
    return {"profile": name, "file": file, "status": "written", "bytes": len(req.content)}


@app.put("/api/profiles/{name}/procedural/{file}")
async def put_procedural(name: str, file: str, req: MemoryFileWriteRequest):
    """Write a procedural memory file (workflows, failures)."""
    if file not in ("workflows.md", "failures.md"):
        raise HTTPException(400, f"Unknown procedural file: {file}")
    profile_manager.write_procedural(req.content, name=name, file=file)
    return {"profile": name, "file": file, "status": "written", "bytes": len(req.content)}


class MemoryImportRequest(BaseModel):
    """Import raw text into PANIC's memory layers via LLM extraction."""
    content: str
    source_label: str = "imported"


@app.post("/api/profiles/{name}/import")
async def import_memory(name: str, req: MemoryImportRequest):
    """Import text (e.g. MEMORY.md) into a profile's memory layers via LLM extraction."""
    from panic.extraction import ExtractionPipeline, EXTRACTION_PROMPT

    if not engine.connected:
        raise HTTPException(503, "LLM not connected — cannot run extraction")

    # Build a fake engine with the import content as chat history
    class ImportEngine:
        chat_history = [{"role": "user", "content": req.content, "turn": 0}]
        turn = 1
        extraction_model = engine.extraction_model
        extraction_api_key = engine.extraction_api_key
        api_key = engine.api_key
        connected = True
        model = engine.model
        provider = engine.provider

    pm_copy = ProfileManager()
    if name != pm_copy.active_profile:
        pm_copy.switch(name, None)

    pipeline = ExtractionPipeline(profile_manager=pm_copy, engine=ImportEngine())
    result = pipeline.run_session_end()
    return {"profile": name, "source": req.source_label, "extraction": result}


# --- Legacy session persistence (kept for backward compat) ---

class SessionSaveRequest(BaseModel):
    name: str = "default"


@app.post("/api/session/save")
async def save_session(req: SessionSaveRequest):
    """Save current state to active profile."""
    try:
        profile_manager.save_state(profile_manager.active_profile, engine)
        return {
            "status": "saved",
            "profile": profile_manager.active_profile,
            "turn_count": engine.turn,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# --- Plugin integration endpoints ---
# These endpoints are called by the OpenClaw TypeScript plugin (engine.ts)
# to integrate PANIC with OpenClaw's ContextEngine lifecycle.


class IngestRequest(BaseModel):
    """A single message forwarded from OpenClaw's ingest() lifecycle hook."""
    message: str       # message content
    role: str = "user" # "user" or "assistant"
    turn: int = 0      # turn number from OpenClaw


class AssembleRequest(BaseModel):
    """Request to build layered context for a query."""
    query: str                   # the user's message (retrieval query)
    token_budget: int = 200_000  # total token budget from OpenClaw


@app.post("/api/chat/ingest")
async def ingest_message(req: IngestRequest):
    """
    Ingest a single message into PANIC's graph engine.

    Called by the OpenClaw plugin for each user/assistant message.
    This feeds the dual graph extractors (rule-based + LLM) so the
    knowledge graph stays up to date without going through the full
    process_turn() pipeline (which also calls the LLM for a response).
    """
    # Encode the message
    embedding = engine.encoder.encode(req.message)
    engine.turn = max(engine.turn, req.turn)
    engine.turn_embeddings[req.turn] = embedding

    # Feed both graph extractors
    source = "user" if req.role == "user" else "llm"
    engine.rule_extractor.extract_and_apply(req.message, req.turn, source=source)
    engine.llm_extractor.add_turn(req.message, req.turn, source=source)
    engine._turns_since_flush += 1

    # Flush LLM extractor periodically
    if engine._turns_since_flush >= engine.llm_flush_interval:
        try:
            engine.llm_extractor.flush_and_apply()
        except Exception as e:
            logger.warning(f"LLM extraction flush failed during ingest: {e}")
        engine._turns_since_flush = 0

    # Store in chat history for extraction pipeline
    engine.chat_history.append({
        "role": req.role,
        "content": req.message,
        "turn": req.turn,
    })

    return {"status": "ingested", "turn": req.turn}


@app.post("/api/assemble")
async def assemble_context(req: AssembleRequest):
    """
    Build layered memory context for a query.

    Called by the OpenClaw plugin's assemble() method. Returns the
    assembled context sections (semantic, episodic, working, procedural)
    formatted as labeled prompt text that gets prepended to the system prompt.

    This does NOT call the LLM — it only does retrieval and formatting.
    """
    # Encode the query
    query_embedding = engine.encoder.encode(req.query)

    # Retrieve from current session graph (working memory)
    working_items = engine._retrieve(req.query, query_embedding)

    # Enrich with conversation metadata
    formatter = ContextFormatter(
        rule_graph=engine.rule_graph,
        chat_history=engine.chat_history,
        total_turns=engine.turn,
    )
    formatted_working = formatter.format_items(working_items, query=req.query)

    # Build all 4 memory layers with per-layer budgets
    from panic.translation.translator import MemoryLayer
    layers = engine._build_memory_layers(req.query, query_embedding, formatted_working)

    # Get immediate buffer items
    immediate_items = engine._get_immediate_items()

    # Construct the layered prompt
    prompt_result = engine.translator.construct_layered_prompt(
        query=req.query,
        layers=layers,
        immediate_buffer=immediate_items,
    )

    return {
        "context_section": prompt_result.context_section,
        "immediate_section": prompt_result.immediate_section,
        "tokens_used": prompt_result.token_usage,
        "included_items": prompt_result.included_items,
        "dropped_items": prompt_result.dropped_items,
    }


# Mount static files for frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
