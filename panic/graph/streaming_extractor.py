"""
PANIC Streaming Graph Extractor — Per-Turn Incremental Extraction

Instead of batching 20 turns for one LLM call, this extracts from
each turn individually using a lightweight prompt. Benefits:
  - Lower latency (one turn at a time)
  - Deterministic graph structure (no batch-boundary effects)
  - Can use a cheaper/faster model (the prompt is tiny)
  - Graph builds up incrementally during conversation

Trade-off vs batch extraction:
  - More API calls (1 per turn vs 1 per 20)
  - Less cross-turn context (can't see contradictions spanning batch)
  - Slightly less accurate entity resolution

Designed for real-time use in the API. The batch extractor remains
for evaluation where speed matters less than consistency.
"""

import json
import re
from typing import Optional
from dataclasses import dataclass

import litellm

from panic.graph.storage import (
    GraphStorage, GraphNode, GraphEdge, VectorRef,
    NodeType, NodeStatus, EdgeType,
)
from panic.graph.extractors import ExtractionResult, ExtractorPipeline, _make_id


STREAMING_PROMPT = """Extract key information from this conversation turn.

Return JSON (no markdown fences):
{
  "facts": [{"statement": "<fact>", "subject": "<topic>", "value": "<value>"}],
  "entities": [{"name": "<name>", "category": "<person|tool|service|project|location|quantity|temporal>"}],
  "decisions": [{"statement": "<decision>", "choice": "<what was chosen>"}],
  "relations": [{"source": "<entity>", "target": "<entity>", "type": "<uses|depends_on|owns|causes|related_to>"}]
}

If nothing to extract, return: {"facts": [], "entities": [], "decisions": [], "relations": []}

Turn: """


@dataclass
class StreamingExtractorConfig:
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens: int = 1024
    fallback_to_rules: bool = True
    # Skip extraction for short/trivial turns
    min_turn_length: int = 20
    # Cache extracted turns to avoid re-processing
    cache_extracted: bool = True


class StreamingExtractor:
    """
    Per-turn incremental graph extraction using LLM.
    
    Each call to `extract_turn()` processes a single turn and
    immediately updates the graph. No batching, no flushing.
    """

    def __init__(
        self,
        storage: GraphStorage,
        config: Optional[StreamingExtractorConfig] = None,
        rule_fallback: Optional[ExtractorPipeline] = None,
    ):
        self.storage = storage
        self.config = config or StreamingExtractorConfig()
        self.rule_fallback = rule_fallback
        self._extracted_turns: set[int] = set()

    def extract_turn(self, text: str, turn: int, source: str = "user") -> ExtractionResult:
        """
        Extract and apply graph updates from a single turn.
        
        Falls back to rule-based extraction on LLM failure.
        Skips trivially short turns.
        """
        # Skip if already extracted
        if self.config.cache_extracted and turn in self._extracted_turns:
            return ExtractionResult()

        # Skip trivially short turns
        if len(text.strip()) < self.config.min_turn_length:
            if self.rule_fallback:
                result = self.rule_fallback.extract(text, turn, source)
                self.rule_fallback.apply(result)
                self._extracted_turns.add(turn)
                return result
            return ExtractionResult()

        try:
            result = self._extract_single(text, turn, source)
            self._apply_result(result, turn)
            self._extracted_turns.add(turn)
            return result
        except Exception as e:
            # Fallback to rules
            if self.config.fallback_to_rules and self.rule_fallback:
                result = self.rule_fallback.extract(text, turn, source)
                self.rule_fallback.apply(result)
                self._extracted_turns.add(turn)
                return result
            self._extracted_turns.add(turn)
            return ExtractionResult()

    def _extract_single(self, text: str, turn: int, source: str) -> ExtractionResult:
        """Call LLM for a single turn extraction."""
        response = litellm.completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Extract key facts, entities, decisions, and relations. Return valid JSON only."},
                {"role": "user", "content": STREAMING_PROMPT + text},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)
        return self._parse_response(data, turn, source)

    def _parse_response(self, data: dict, turn: int, source: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        result = ExtractionResult()

        for fact in data.get("facts", []):
            statement = fact.get("statement", "")
            if not statement:
                continue
            node_id = _make_id("fact", statement)
            node = GraphNode(
                id=node_id, type=NodeType.FACT,
                data={
                    "statement": statement,
                    "subject": fact.get("subject", ""),
                    "value": fact.get("value", ""),
                    "source": "streaming_llm",
                },
                first_seen=turn, last_seen=turn, source=source,
            )
            result.nodes.append(node)
            result.vector_refs.append(VectorRef(node_id=node_id, vector_index=turn, turn=turn))

        for entity in data.get("entities", []):
            name = entity.get("name", "")
            if not name:
                continue
            node_id = _make_id("entity", name)
            node = GraphNode(
                id=node_id, type=NodeType.ENTITY,
                data={
                    "name": name,
                    "category": entity.get("category", "other"),
                    "source": "streaming_llm",
                },
                first_seen=turn, last_seen=turn, source=source,
            )
            result.nodes.append(node)
            result.vector_refs.append(VectorRef(node_id=node_id, vector_index=turn, turn=turn))

        for decision in data.get("decisions", []):
            statement = decision.get("statement", "")
            if not statement:
                continue
            node_id = _make_id("decision", statement)
            node = GraphNode(
                id=node_id, type=NodeType.DECISION,
                data={
                    "statement": statement,
                    "choice": decision.get("choice", ""),
                    "source": "streaming_llm",
                },
                first_seen=turn, last_seen=turn, source=source,
            )
            result.nodes.append(node)
            result.vector_refs.append(VectorRef(node_id=node_id, vector_index=turn, turn=turn))

        for rel in data.get("relations", []):
            src_name = rel.get("source", "")
            tgt_name = rel.get("target", "")
            if src_name and tgt_name:
                edge_type_map = {
                    "uses": EdgeType.RELATED_TO,
                    "depends_on": EdgeType.DEPENDS_ON,
                    "owns": EdgeType.OWNS,
                    "causes": EdgeType.CAUSES,
                    "related_to": EdgeType.RELATED_TO,
                }
                edge_type = edge_type_map.get(rel.get("type", "related_to"), EdgeType.RELATED_TO)
                result.edges.append(GraphEdge(
                    source_id=_make_id("entity", src_name),
                    target_id=_make_id("entity", tgt_name),
                    type=edge_type, turn=turn,
                ))

        return result

    def _apply_result(self, result: ExtractionResult, turn: int):
        """Apply extraction result to graph storage."""
        known_ids = {node.id for node in result.nodes}
        for node in result.nodes:
            self.storage.upsert_node(node)
        for edge in result.edges:
            for eid in (edge.source_id, edge.target_id):
                if eid not in known_ids and not self.storage.get_node(eid):
                    placeholder = GraphNode(
                        id=eid, type=NodeType.ENTITY,
                        data={"name": eid.replace("entity_", "").replace("_", " ")},
                        first_seen=turn, last_seen=turn, source="llm",
                    )
                    self.storage.upsert_node(placeholder)
                    known_ids.add(eid)
            self.storage.upsert_edge(edge)
        for ref in result.vector_refs:
            self.storage.add_vector_ref(ref)
