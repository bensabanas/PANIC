"""
PANIC Context Formatter — Conversation-Aware Prompt Construction

Enriches retrieved context items with:
  - Turn numbers and temporal markers
  - Speaker labels (user vs assistant)
  - Contradiction annotations (when newer info supersedes older)
  - Temporal ordering cues
  - Grouping by topic/entity for coherence

This sits between retrieval and the translator, transforming raw
ContextItems into structured, LLM-friendly context blocks.
"""

from dataclasses import dataclass, field
from typing import Optional
from panic.translation.translator import ContextItem
from panic.graph.storage import GraphStorage, NodeStatus


@dataclass
class FormattedContext:
    """A context item enriched with conversation metadata."""
    item: ContextItem
    turn_label: str = ""          # e.g. "Turn 42"
    speaker: str = ""             # "user" or "assistant"
    temporal_marker: str = ""     # e.g. "early", "mid", "recent"
    is_superseded: bool = False   # newer info exists
    superseded_by_turn: Optional[int] = None
    contradiction_note: str = ""  # e.g. "⚠ Updated at turn 150: deadline changed to April"
    related_turns: list[int] = field(default_factory=list)


class ContextFormatter:
    """
    Formats retrieved context items into structured, conversation-aware blocks.

    The formatter enriches raw retrieval results with metadata that helps
    the LLM reason about temporal ordering, contradictions, and relationships
    between pieces of information.
    """

    def __init__(
        self,
        rule_graph: Optional[GraphStorage] = None,
        chat_history: Optional[list[dict]] = None,
        total_turns: int = 0,
    ):
        self.rule_graph = rule_graph
        self.chat_history = chat_history or []
        self.total_turns = total_turns

        # Build turn->speaker index from chat history
        self._turn_speakers: dict[int, str] = {}
        self._turn_texts: dict[int, str] = {}
        for entry in self.chat_history:
            t = entry.get("turn", 0)
            role = entry.get("role", "user")
            self._turn_speakers[t] = role
            self._turn_texts[t] = entry.get("content", "")

    def format_items(
        self,
        retrieved_items: list[ContextItem],
        query: str = "",
    ) -> list[ContextItem]:
        """
        Enrich retrieved items with conversation-aware metadata and
        return new ContextItems with formatted text.

        The formatted text includes turn numbers, speaker labels,
        temporal markers, and contradiction annotations — giving the
        LLM explicit structural cues about the conversation flow.

        Args:
            retrieved_items: Raw retrieved context items from dual-engine.
            query: Current user query (for relevance context).

        Returns:
            New list of ContextItems with enriched text, same order.
        """
        if not retrieved_items:
            return []

        # Detect contradictions from graph
        superseded_turns = self._find_superseded_turns()

        formatted = []
        for item in retrieved_items:
            fc = self._format_single(item, superseded_turns)
            formatted.append(fc)

        # Sort by turn number for temporal coherence
        formatted.sort(key=lambda fc: fc.item.source_turn)

        # Build formatted ContextItems
        result = []
        for fc in formatted:
            new_text = self._build_formatted_text(fc)
            result.append(ContextItem(
                id=fc.item.id,
                text=new_text,
                source_turn=fc.item.source_turn,
                relevance_score=fc.item.relevance_score,
                tier=fc.item.tier,
            ))

        return result

    def _format_single(
        self,
        item: ContextItem,
        superseded_turns: dict[int, dict],
    ) -> FormattedContext:
        """Format a single context item with metadata."""
        turn = item.source_turn

        # Speaker label
        speaker = self._turn_speakers.get(turn, "user")

        # Temporal marker
        temporal = self._temporal_marker(turn)

        # Check if superseded
        is_superseded = turn in superseded_turns
        superseded_info = superseded_turns.get(turn, {})
        superseded_by = superseded_info.get("superseded_by_turn")
        contradiction_note = ""

        if is_superseded and superseded_by:
            # Look up what the newer info says
            newer_text = self._turn_texts.get(superseded_by, "")
            if newer_text:
                snippet = newer_text[:120].strip()
                contradiction_note = f"[Note: Updated at turn {superseded_by}: {snippet}]"
            else:
                contradiction_note = f"[Note: This was updated later at turn {superseded_by}]"

        # Find related turns via graph
        related = self._find_related_turns(item)

        return FormattedContext(
            item=item,
            turn_label=f"Turn {turn}",
            speaker=speaker,
            temporal_marker=temporal,
            is_superseded=is_superseded,
            superseded_by_turn=superseded_by,
            contradiction_note=contradiction_note,
            related_turns=related,
        )

    def _build_formatted_text(self, fc: FormattedContext) -> str:
        """Build the enriched text string for a formatted context item."""
        parts = []

        # Header: [Turn 42 · user · early in conversation]
        header_parts = [fc.turn_label]
        if fc.speaker:
            header_parts.append(fc.speaker)
        if fc.temporal_marker:
            header_parts.append(fc.temporal_marker)
        header = "[" + " · ".join(header_parts) + "]"
        parts.append(header)

        # Main text
        parts.append(fc.item.text)

        # Contradiction annotation
        if fc.contradiction_note:
            parts.append(fc.contradiction_note)

        # Related turns hint (only if there are a few, not too noisy)
        if fc.related_turns and len(fc.related_turns) <= 5:
            related_str = ", ".join(str(t) for t in sorted(fc.related_turns))
            parts.append(f"[Related: turns {related_str}]")

        return "\n".join(parts)

    def _temporal_marker(self, turn: int) -> str:
        """Generate a human-readable temporal marker.
        Tells the LLM roughly when something happened in the conversation.
        The LLM can't count turns — this gives it a sense of time.
        """
        if self.total_turns <= 0:
            return ""

        ratio = turn / self.total_turns

        if ratio < 0.15:
            return "early in conversation"
        elif ratio < 0.35:
            return "earlier"
        elif ratio < 0.65:
            return "mid-conversation"
        elif ratio < 0.85:
            return "later"
        else:
            return "recent"

    def _find_superseded_turns(self) -> dict[int, dict]:
        """
        Find turns whose information has been superseded by later turns.
        Uses the rule-based graph (precise contradiction detection).

        Returns dict mapping turn -> {superseded_by_turn: int}
        """
        superseded = {}
        if not self.rule_graph:
            return superseded

        # Look for SUPERSEDED/CONTRADICTED nodes
        all_nodes = self.rule_graph.get_all_nodes(active_only=False)
        for node in all_nodes:
            if node.status in (NodeStatus.SUPERSEDED, NodeStatus.CONTRADICTED):
                # Get the vector refs for this node (maps to turns)
                refs = self.rule_graph.get_vector_refs(node.id)
                # Find what supersedes it via edges
                edges_to = self.rule_graph.get_edges_to(node.id)
                superseder_turn = None
                for edge in edges_to:
                    if edge.type.value == "supersedes":
                        # The source of the supersedes edge is the newer node
                        newer_node = self.rule_graph.get_node(edge.source_id)
                        if newer_node:
                            superseder_turn = newer_node.last_seen

                for ref in refs:
                    superseded[ref.vector_index] = {
                        "superseded_by_turn": superseder_turn,
                    }

        return superseded

    def _find_related_turns(self, item: ContextItem) -> list[int]:
        """Find turns related to this item via graph edges."""
        if not self.rule_graph:
            return []

        related_turns = set()
        node_ids = self.rule_graph.get_nodes_for_vector(item.source_turn)

        for node_id in node_ids:
            connected = self.rule_graph.get_connected_nodes(node_id, max_hops=1)
            for conn_id in connected:
                refs = self.rule_graph.get_vector_refs(conn_id)
                for ref in refs:
                    if ref.vector_index != item.source_turn:
                        related_turns.add(ref.vector_index)

        # Limit to avoid noise
        return sorted(related_turns)[:5]
