"""
PANIC Graph Extractors

Six-stage pipeline that runs on every turn (user input and LLM response):
  1. Entity Extractor (NER)
  2. Fact Extractor (pattern matching)
  3. Relation Extractor (co-occurrence + patterns)
  4. Coreference Resolver (alias/pronoun tracking)
  5. Contradiction Detector (fact conflict detection)
  6. Decision Extractor (commitment/action detection)

MVP: spaCy NER + rule-based patterns. No trained classifiers.
"""

import hashlib
import re
import spacy
from typing import Optional
from dataclasses import dataclass, field

from panic.graph.storage import (
    GraphStorage, GraphNode, GraphEdge, VectorRef,
    NodeType, NodeStatus, EdgeType,
)


@dataclass
class ExtractionResult:
    """Output of running the full extraction pipeline on one text."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    vector_refs: list[VectorRef] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)  # (old_id, new_id)


def _make_id(prefix: str, text: str) -> str:
    """Create a stable hash ID for a node."""
    h = hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]
    return f"{prefix}_{h}"


class ExtractorPipeline:
    """Runs all six extractors in sequence on input text."""

    def __init__(self, storage: GraphStorage, spacy_model: str = "en_core_web_sm"):
        self.storage = storage
        self._nlp = spacy.load(spacy_model)

        # Pattern matchers for fact extraction
        self._fact_patterns = [
            # Definitions
            re.compile(r"(.+?)\s+(?:is|are|was|were)\s+(?:defined as|known as)\s+(.+)", re.IGNORECASE),
            # Preferences
            re.compile(r"(?:i |we )?(?:prefer|want|like|choose|pick)\s+(.+?)(?:\s+(?:over|instead of|rather than)\s+(.+))?$", re.IGNORECASE),
            # Constraints
            re.compile(r"(?:don'?t|do not|never|always|must|should|must not|should not)\s+(.+)", re.IGNORECASE),
        ]

        # Patterns for decision extraction
        self._decision_patterns = [
            re.compile(r"(?:let'?s|we(?:'ll| will| should)?|i(?:'ll| will)?)\s+(?:go with|use|choose|pick|decide on|stick with|switch to)\s+(.+)", re.IGNORECASE),
            re.compile(r"(?:the |our )?(?:plan|decision|approach|strategy)\s+(?:is|will be)\s+(?:to\s+)?(.+)", re.IGNORECASE),
            re.compile(r"(?:we |i )?(?:decided|agreed|committed)\s+(?:to|on|that)\s+(.+)", re.IGNORECASE),
        ]

        # Relation verb patterns
        self._relation_patterns = [
            (re.compile(r"(.+?)\s+(?:depends on|requires|needs)\s+(.+)", re.IGNORECASE), EdgeType.DEPENDS_ON),
            (re.compile(r"(.+?)\s+(?:caused|led to|resulted in|broke)\s+(.+)", re.IGNORECASE), EdgeType.CAUSES),
            (re.compile(r"(.+?)(?:'s| belongs? to| is owned by| is part of)\s+(.+)", re.IGNORECASE), EdgeType.OWNS),
        ]

        # Alias tracking for coreference
        self._aliases: dict[str, str] = {}  # alias -> canonical entity id

    def extract(self, text: str, turn: int, source: str = "user") -> ExtractionResult:
        """
        Run the full extraction pipeline on a text.

        Args:
            text: Raw text input.
            turn: Current turn number.
            source: "user" or "llm".

        Returns:
            ExtractionResult with all extracted nodes, edges, refs, and contradictions.
        """
        result = ExtractionResult()
        doc = self._nlp(text)

        # 1. Entity extraction
        entities = self._extract_entities(doc, text, turn, source)
        result.nodes.extend(entities)

        # 2. Fact extraction
        facts = self._extract_facts(text, turn, source)
        result.nodes.extend(facts)

        # 3. Relation extraction (edges between entities found in this turn)
        entity_ids = [e.id for e in entities]
        relations = self._extract_relations(text, entity_ids, turn)
        result.edges.extend(relations)

        # Add co-occurrence edges for entities in same turn
        cooccurrence = self._extract_cooccurrence(entity_ids, turn)
        result.edges.extend(cooccurrence)

        # 4. Coreference resolution (update aliases)
        self._resolve_coreferences(doc, entities)

        # 5. Contradiction detection
        for fact in facts:
            contradiction = self._detect_contradiction(fact)
            if contradiction:
                result.contradictions.append(contradiction)

        # 6. Decision extraction
        decisions = self._extract_decisions(text, turn, source)
        result.nodes.extend(decisions)

        # Create vector refs for all nodes
        for node in result.nodes:
            result.vector_refs.append(VectorRef(
                node_id=node.id, vector_index=turn, turn=turn,
            ))

        return result

    def apply(self, result: ExtractionResult):
        """Apply extraction results to the graph storage."""
        for node in result.nodes:
            self.storage.upsert_node(node)

        for edge in result.edges:
            self.storage.upsert_edge(edge)

        for ref in result.vector_refs:
            self.storage.add_vector_ref(ref)

        for old_id, new_id in result.contradictions:
            self.storage.update_node_status(old_id, NodeStatus.SUPERSEDED)
            self.storage.penalize_vector_refs(old_id)
            self.storage.upsert_edge(GraphEdge(
                source_id=new_id, target_id=old_id,
                type=EdgeType.SUPERSEDES, turn=0,
            ))

    def extract_and_apply(self, text: str, turn: int, source: str = "user") -> ExtractionResult:
        """Extract and immediately apply to storage. Convenience method."""
        result = self.extract(text, turn, source)
        self.apply(result)
        return result

    # --- Individual extractors ---

    def _extract_entities(self, doc, text: str, turn: int, source: str) -> list[GraphNode]:
        """Stage 1: NER-based entity extraction."""
        entities = []
        seen = set()

        for ent in doc.ents:
            canonical = ent.text.strip()
            if not canonical or canonical.lower() in seen:
                continue
            seen.add(canonical.lower())

            node_id = _make_id("entity", canonical)
            entity_type = self._map_spacy_label(ent.label_)

            entities.append(GraphNode(
                id=node_id,
                type=NodeType.ENTITY,
                data={
                    "name": canonical,
                    "spacy_label": ent.label_,
                    "category": entity_type,
                },
                first_seen=turn,
                last_seen=turn,
                source=source,
            ))

            # Track alias
            self._aliases[canonical.lower()] = node_id

        return entities

    def _extract_facts(self, text: str, turn: int, source: str) -> list[GraphNode]:
        """Stage 2: Pattern-based fact extraction."""
        facts = []

        # Split into sentences for finer extraction
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        for sentence in sentences:
            for pattern in self._fact_patterns:
                match = pattern.search(sentence)
                if match:
                    statement = sentence.strip()
                    node_id = _make_id("fact", statement)

                    facts.append(GraphNode(
                        id=node_id,
                        type=NodeType.FACT,
                        data={
                            "statement": statement,
                            "pattern": pattern.pattern[:50],
                        },
                        first_seen=turn,
                        last_seen=turn,
                        source=source,
                    ))
                    break  # one fact per sentence max

        return facts

    def _extract_relations(self, text: str, entity_ids: list[str], turn: int) -> list[GraphEdge]:
        """Stage 3: Verb-pattern based relation extraction."""
        edges = []

        for pattern, edge_type in self._relation_patterns:
            match = pattern.search(text)
            if match and len(entity_ids) >= 2:
                # Link the first two entities found with this relation type
                edges.append(GraphEdge(
                    source_id=entity_ids[0],
                    target_id=entity_ids[1],
                    type=edge_type,
                    turn=turn,
                ))

        return edges

    def _extract_cooccurrence(self, entity_ids: list[str], turn: int) -> list[GraphEdge]:
        """Stage 3b: Co-occurrence edges for entities in the same turn."""
        edges = []
        for i, src in enumerate(entity_ids):
            for tgt in entity_ids[i + 1:]:
                if src != tgt:
                    edges.append(GraphEdge(
                        source_id=src,
                        target_id=tgt,
                        type=EdgeType.RELATED_TO,
                        weight=0.5,  # weaker than explicit relations
                        turn=turn,
                    ))
        return edges

    def _resolve_coreferences(self, doc, entities: list[GraphNode]):
        """Stage 4: Track aliases for coreference resolution."""
        # MVP: Track noun chunks that map to known entities
        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.strip().lower()

            # Skip if it's already a known alias
            if chunk_lower in self._aliases:
                continue

            # Check if this chunk refers to a recently extracted entity
            for entity in entities:
                name = entity.data.get("name", "").lower()
                if name and (name in chunk_lower or chunk_lower in name):
                    self._aliases[chunk_lower] = entity.id
                    break

    def _detect_contradiction(self, new_fact: GraphNode) -> Optional[tuple[str, str]]:
        """
        Stage 5: Check if a new fact contradicts an existing active fact.

        Returns (old_fact_id, new_fact_id) if contradiction found, else None.
        """
        statement = new_fact.data.get("statement", "").lower()
        if not statement:
            return None

        existing_facts = self.storage.get_nodes_by_type(NodeType.FACT, status=NodeStatus.ACTIVE)

        for existing in existing_facts:
            if existing.id == new_fact.id:
                continue

            old_statement = existing.data.get("statement", "").lower()
            if not old_statement:
                continue

            # Check for direct negation patterns
            if self._is_negation(old_statement, statement):
                return (existing.id, new_fact.id)

            # Check for value changes on same subject
            if self._is_value_change(old_statement, statement):
                return (existing.id, new_fact.id)

        return None

    def _extract_decisions(self, text: str, turn: int, source: str) -> list[GraphNode]:
        """Stage 6: Pattern-based decision extraction."""
        decisions = []

        for pattern in self._decision_patterns:
            match = pattern.search(text)
            if match:
                statement = match.group(0).strip()
                node_id = _make_id("decision", statement)

                decisions.append(GraphNode(
                    id=node_id,
                    type=NodeType.DECISION,
                    data={
                        "statement": statement,
                        "detail": match.group(1).strip() if match.lastindex else "",
                    },
                    first_seen=turn,
                    last_seen=turn,
                    source=source,
                ))
                break  # one decision per text max

        return decisions

    # --- Helpers ---

    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy NER labels to PANIC categories."""
        mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "DATE": "temporal",
            "TIME": "temporal",
            "MONEY": "quantity",
            "PRODUCT": "technical",
            "WORK_OF_ART": "technical",
            "LAW": "technical",
            "LANGUAGE": "technical",
        }
        return mapping.get(label, "other")

    def _is_negation(self, old: str, new: str) -> bool:
        """Check if two statements are direct negations."""
        # Simple heuristic: same core content but one has negation
        negation_words = {"not", "don't", "dont", "never", "no", "won't", "wont", "shouldn't", "cant", "can't"}

        old_words = set(old.split())
        new_words = set(new.split())

        old_has_neg = bool(old_words & negation_words)
        new_has_neg = bool(new_words & negation_words)

        if old_has_neg == new_has_neg:
            return False  # both positive or both negative

        # Check if the non-negation content overlaps significantly
        old_content = old_words - negation_words - {"i", "we", "the", "a", "an", "is", "are", "was", "were", "to", "it"}
        new_content = new_words - negation_words - {"i", "we", "the", "a", "an", "is", "are", "was", "were", "to", "it"}

        if not old_content or not new_content:
            return False

        overlap = len(old_content & new_content) / max(len(old_content), len(new_content))
        return overlap > 0.5

    def _is_value_change(self, old: str, new: str) -> bool:
        """Check if two statements assign different values to the same subject."""
        # Look for "X is Y" patterns where X matches but Y differs
        is_pattern = re.compile(r"(.+?)\s+(?:is|are|was|will be|should be)\s+(.+)")

        old_match = is_pattern.match(old)
        new_match = is_pattern.match(new)

        if not old_match or not new_match:
            return False

        old_subject = old_match.group(1).strip()
        new_subject = new_match.group(1).strip()
        old_value = old_match.group(2).strip()
        new_value = new_match.group(2).strip()

        # Same subject, different value
        subject_words = set(old_subject.split()) & set(new_subject.split())
        if len(subject_words) < 1:
            return False

        return old_value != new_value

    def resolve_alias(self, text: str) -> Optional[str]:
        """Look up a text alias and return the canonical entity ID, if known."""
        return self._aliases.get(text.lower().strip())

    def get_aliases(self) -> dict[str, str]:
        """Return current alias map."""
        return dict(self._aliases)
