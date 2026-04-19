"""
PANIC Graph Extractors — LLM-Assisted

Uses litellm to extract structured facts, relations, and contradictions
from conversation turns. Much higher recall than the rule-based pipeline.

Design:
  - Batches turns for efficiency (1 LLM call per N turns)
  - Extracts: facts, entities, relations, decisions, contradictions
  - Returns standard ExtractionResult objects for graph storage
  - Falls back to rule-based extraction on LLM failure
"""

import json
import hashlib
import re
import asyncio
import concurrent.futures
from typing import Optional
from dataclasses import dataclass, field

import litellm

from panic.graph.storage import (
    GraphStorage, GraphNode, GraphEdge, VectorRef,
    NodeType, NodeStatus, EdgeType,
)
from panic.graph.extractors import ExtractionResult, ExtractorPipeline, _make_id


EXTRACTION_PROMPT = """Extract structured information from these conversation turns.

For each turn, identify:
1. **Facts**: Concrete statements of fact (deadlines, names, numbers, tools, ports, preferences, budgets, locations, assignments)
2. **Entities**: Named things (people, tools, services, projects, places)
3. **Decisions**: Choices that were made ("let's go with X", "we decided Y")
4. **Contradictions**: If a new statement contradicts an earlier one (e.g., deadline changed from X to Y)
5. **Relations**: How entities relate to each other (X uses Y, X depends on Y, X owns Y)

Return JSON (no markdown fences):
{
  "facts": [
    {"turn": <turn_number>, "statement": "<the fact>", "subject": "<what it's about>", "value": "<the value>"}
  ],
  "entities": [
    {"turn": <turn_number>, "name": "<entity name>", "category": "<person|tool|service|project|location|quantity|temporal>"}
  ],
  "decisions": [
    {"turn": <turn_number>, "statement": "<the decision>", "choice": "<what was chosen>"}
  ],
  "contradictions": [
    {"new_turn": <turn_number>, "old_subject": "<what changed>", "old_value": "<previous value>", "new_value": "<new value>"}
  ],
  "relations": [
    {"turn": <turn_number>, "source": "<entity>", "target": "<entity>", "type": "<uses|depends_on|owns|causes|related_to>"}
  ]
}

Be thorough. Extract EVERY factual statement, even if it seems trivial. Include "by the way" asides, port numbers, timezone mentions, tool assignments, budget figures, deadlines — everything.

Conversation turns:
"""


@dataclass
class LLMExtractorConfig:
    model: str = "claude-haiku-4-5-20251001"
    batch_size: int = 20  # turns per LLM call
    temperature: float = 0.0
    max_tokens: int = 4096
    fallback_to_rules: bool = True


class LLMExtractorPipeline:
    """
    LLM-assisted graph extraction pipeline.
    
    Batches conversation turns, calls LLM for structured extraction,
    and converts results to graph nodes/edges/refs.
    """

    def __init__(self, storage: GraphStorage, config: Optional[LLMExtractorConfig] = None,
                 rule_fallback: Optional[ExtractorPipeline] = None):
        self.storage = storage
        self.config = config or LLMExtractorConfig()
        self.rule_fallback = rule_fallback
        self._pending_turns: list[dict] = []  # buffered turns awaiting extraction
        self._extracted_turns: set[int] = set()  # turns already processed

    def add_turn(self, text: str, turn: int, source: str = "user"):
        """Buffer a turn for batch extraction."""
        self._pending_turns.append({
            "text": text,
            "turn": turn,
            "source": source,
        })

    def flush(self) -> list[ExtractionResult]:
        """Process all pending turns through LLM extraction."""
        if not self._pending_turns:
            return []

        results = []
        # Process in batches
        for i in range(0, len(self._pending_turns), self.config.batch_size):
            batch = self._pending_turns[i:i + self.config.batch_size]
            try:
                batch_results = self._extract_batch(batch)
                results.extend(batch_results)
            except Exception as e:
                print(f"  LLM extraction failed: {e}")
                if self.config.fallback_to_rules and self.rule_fallback:
                    for turn_data in batch:
                        result = self.rule_fallback.extract(
                            turn_data["text"], turn_data["turn"], turn_data["source"]
                        )
                        self.rule_fallback.apply(result)
                        results.append(result)

        self._pending_turns.clear()
        return results

    def flush_parallel(self, max_workers: int = 4) -> list[ExtractionResult]:
        """
        Process all pending turns through LLM extraction with parallel API calls.
        Uses ThreadPoolExecutor for concurrent batch processing.
        """
        if not self._pending_turns:
            return []

        # Split into batches
        batches = []
        for i in range(0, len(self._pending_turns), self.config.batch_size):
            batches.append(self._pending_turns[i:i + self.config.batch_size])

        results = []
        batch_results_map = {}

        def process_batch(batch_idx, batch):
            try:
                return batch_idx, self._extract_batch(batch), None
            except Exception as e:
                return batch_idx, None, (e, batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                batch_idx, batch_results, error = future.result()
                if batch_results is not None:
                    batch_results_map[batch_idx] = batch_results
                elif error is not None:
                    e, batch = error
                    print(f"  LLM extraction failed (batch {batch_idx}): {e}")
                    if self.config.fallback_to_rules and self.rule_fallback:
                        fallback_results = []
                        for turn_data in batch:
                            result = self.rule_fallback.extract(
                                turn_data["text"], turn_data["turn"], turn_data["source"]
                            )
                            self.rule_fallback.apply(result)
                            fallback_results.append(result)
                        batch_results_map[batch_idx] = fallback_results

        # Reassemble in order
        for idx in sorted(batch_results_map.keys()):
            results.extend(batch_results_map[idx])

        self._pending_turns.clear()
        return results

    def flush_and_apply_parallel(self, max_workers: int = 4) -> list[ExtractionResult]:
        """Flush with parallel API calls and apply to graph."""
        results = self.flush_parallel(max_workers=max_workers)
        for r in results:
            self._apply_result(r)
        return results

    def extract_and_apply(self, text: str, turn: int, source: str = "user") -> Optional[ExtractionResult]:
        """
        Add turn and flush if batch is full.
        Returns result only when a flush happens.
        """
        self.add_turn(text, turn, source)
        if len(self._pending_turns) >= self.config.batch_size:
            results = self.flush()
            for r in results:
                self._apply_result(r)
            return results[-1] if results else None
        return None

    def flush_and_apply(self) -> list[ExtractionResult]:
        """Flush remaining turns and apply to graph."""
        results = self.flush()
        for r in results:
            self._apply_result(r)
        return results

    def _extract_batch(self, batch: list[dict]) -> list[ExtractionResult]:
        """Call LLM to extract structured info from a batch of turns."""
        # Format turns for the prompt
        turns_text = ""
        for td in batch:
            turns_text += f"\n[Turn {td['turn']}] ({td['source']}): {td['text']}"

        response = litellm.completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are a precise information extraction system. Extract ALL factual content. Return valid JSON only."},
                {"role": "user", "content": EXTRACTION_PROMPT + turns_text},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)

        # Convert to ExtractionResults grouped by turn
        results_by_turn: dict[int, ExtractionResult] = {}
        for td in batch:
            results_by_turn[td["turn"]] = ExtractionResult()

        # Process facts
        for fact in data.get("facts", []):
            turn = fact.get("turn")
            if turn not in results_by_turn:
                continue
            statement = fact.get("statement", "")
            if not statement:
                continue

            node_id = _make_id("fact", statement)
            node = GraphNode(
                id=node_id,
                type=NodeType.FACT,
                data={
                    "statement": statement,
                    "subject": fact.get("subject", ""),
                    "value": fact.get("value", ""),
                    "source": "llm_extraction",
                },
                first_seen=turn,
                last_seen=turn,
                source="user",
            )
            results_by_turn[turn].nodes.append(node)
            results_by_turn[turn].vector_refs.append(
                VectorRef(node_id=node_id, vector_index=turn, turn=turn)
            )

        # Process entities
        for entity in data.get("entities", []):
            turn = entity.get("turn")
            if turn not in results_by_turn:
                continue
            name = entity.get("name", "")
            if not name:
                continue

            node_id = _make_id("entity", name)
            node = GraphNode(
                id=node_id,
                type=NodeType.ENTITY,
                data={
                    "name": name,
                    "category": entity.get("category", "other"),
                    "source": "llm_extraction",
                },
                first_seen=turn,
                last_seen=turn,
                source="user",
            )
            results_by_turn[turn].nodes.append(node)
            results_by_turn[turn].vector_refs.append(
                VectorRef(node_id=node_id, vector_index=turn, turn=turn)
            )

        # Process decisions
        for decision in data.get("decisions", []):
            turn = decision.get("turn")
            if turn not in results_by_turn:
                continue
            statement = decision.get("statement", "")
            if not statement:
                continue

            node_id = _make_id("decision", statement)
            node = GraphNode(
                id=node_id,
                type=NodeType.DECISION,
                data={
                    "statement": statement,
                    "choice": decision.get("choice", ""),
                    "source": "llm_extraction",
                },
                first_seen=turn,
                last_seen=turn,
                source="user",
            )
            results_by_turn[turn].nodes.append(node)
            results_by_turn[turn].vector_refs.append(
                VectorRef(node_id=node_id, vector_index=turn, turn=turn)
            )

        # Process contradictions
        for contra in data.get("contradictions", []):
            new_turn = contra.get("new_turn")
            if new_turn not in results_by_turn:
                continue
            old_subject = contra.get("old_subject", "")
            old_value = contra.get("old_value", "")
            new_value = contra.get("new_value", "")

            if old_subject and old_value and new_value:
                # Find and supersede old fact nodes that match
                old_statement = f"{old_subject} is {old_value}"
                old_id = _make_id("fact", old_statement)
                new_statement = f"{old_subject} changed to {new_value}"
                new_id = _make_id("fact", new_statement)

                # Create new fact node
                new_node = GraphNode(
                    id=new_id,
                    type=NodeType.FACT,
                    data={
                        "statement": new_statement,
                        "subject": old_subject,
                        "value": new_value,
                        "supersedes": old_id,
                        "source": "llm_extraction",
                    },
                    first_seen=new_turn,
                    last_seen=new_turn,
                    source="user",
                )
                results_by_turn[new_turn].nodes.append(new_node)
                results_by_turn[new_turn].vector_refs.append(
                    VectorRef(node_id=new_id, vector_index=new_turn, turn=new_turn)
                )
                results_by_turn[new_turn].contradictions.append((old_id, new_id))

        # Process relations
        for rel in data.get("relations", []):
            turn = rel.get("turn")
            if turn not in results_by_turn:
                continue
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            rel_type = rel.get("type", "related_to")

            if source_name and target_name:
                source_id = _make_id("entity", source_name)
                target_id = _make_id("entity", target_name)

                edge_type_map = {
                    "uses": EdgeType.RELATED_TO,
                    "depends_on": EdgeType.DEPENDS_ON,
                    "owns": EdgeType.OWNS,
                    "causes": EdgeType.CAUSES,
                    "related_to": EdgeType.RELATED_TO,
                    "decided_on": EdgeType.DECIDED_ON,
                }
                edge_type = edge_type_map.get(rel_type, EdgeType.RELATED_TO)

                results_by_turn[turn].edges.append(
                    GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        type=edge_type,
                        turn=turn,
                    )
                )

        return list(results_by_turn.values())

    def _apply_result(self, result: ExtractionResult):
        """Apply extraction result to graph storage."""
        # Collect all node IDs we're about to insert
        known_ids = {node.id for node in result.nodes}
        for node in result.nodes:
            self.storage.upsert_node(node)
        # Ensure edge endpoints exist before inserting edges
        for edge in result.edges:
            for endpoint_id in (edge.source_id, edge.target_id):
                if endpoint_id not in known_ids and not self.storage.get_node(endpoint_id):
                    # Create a placeholder entity node so FK constraint is satisfied
                    placeholder = GraphNode(
                        id=endpoint_id,
                        type=NodeType.ENTITY,
                        data={"name": endpoint_id.replace("entity_", "").replace("_", " ")},
                        first_seen=edge.turn,
                        last_seen=edge.turn,
                        source="llm",
                    )
                    self.storage.upsert_node(placeholder)
                    known_ids.add(endpoint_id)
            self.storage.upsert_edge(edge)
        for ref in result.vector_refs:
            self.storage.add_vector_ref(ref)
        for old_id, new_id in result.contradictions:
            # Try to supersede the old node
            old_node = self.storage.get_node(old_id)
            if old_node:
                self.storage.update_node_status(old_id, NodeStatus.SUPERSEDED)
                self.storage.penalize_vector_refs(old_id)
            # Also search for any existing fact with similar subject
            self._supersede_matching_facts(old_id, new_id, result)
            # Ensure both endpoints exist before creating SUPERSEDES edge
            for endpoint_id in (new_id, old_id):
                if endpoint_id not in known_ids and not self.storage.get_node(endpoint_id):
                    placeholder = GraphNode(
                        id=endpoint_id,
                        type=NodeType.FACT,
                        data={"subject": endpoint_id.replace("fact_", "").replace("_", " ")},
                        first_seen=0,
                        last_seen=0,
                        source="llm",
                    )
                    self.storage.upsert_node(placeholder)
                    known_ids.add(endpoint_id)
            self.storage.upsert_edge(GraphEdge(
                source_id=new_id, target_id=old_id,
                type=EdgeType.SUPERSEDES, turn=0,
            ))

    def _supersede_matching_facts(self, old_id: str, new_id: str, result: ExtractionResult):
        """
        Search for existing facts that might match the old contradiction target,
        even if the hash doesn't match exactly.
        """
        # Get the new node's subject from the result
        new_node = None
        for node in result.nodes:
            if node.id == new_id:
                new_node = node
                break

        if not new_node:
            return

        subject = new_node.data.get("subject", "").lower()
        if not subject:
            return

        # Search existing facts for matching subject
        existing = self.storage.search_nodes(subject, node_type=NodeType.FACT)
        for old_fact in existing:
            if old_fact.id != new_id and old_fact.status == NodeStatus.ACTIVE:
                self.storage.update_node_status(old_fact.id, NodeStatus.SUPERSEDED)
                self.storage.penalize_vector_refs(old_fact.id)
