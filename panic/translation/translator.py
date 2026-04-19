"""
PANIC Translation Layer

Two directions:
  Outbound: PANIC state → optimized LLM prompt
  Inbound:  LLM response → PANIC state update

Manages token budgets, context ordering (exploiting LLM attention patterns),
and structural cues.
"""

import tiktoken
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextItem:
    """A single item that could go into the LLM prompt."""
    id: str
    text: str
    source_turn: int
    relevance_score: float = 0.0
    tier: str = "cold"  # "immediate", "working", "cold"
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0 and self.text:
            # Lazy token count — will be set properly by the translator
            self.token_count = len(self.text.split())  # rough estimate


@dataclass
class PromptConfig:
    """Configuration for prompt construction."""
    max_context_tokens: int = 200_000  # LLM context window
    system_prompt_tokens: int = 1500   # reserved for system prompt
    response_headroom: int = 6000      # reserved for LLM response
    immediate_buffer_max_ratio: float = 0.3  # max % of budget for immediate buffer
    tiktoken_encoding: str = "cl100k_base"


@dataclass
class ConstructedPrompt:
    """Output of prompt construction."""
    system_prompt: str
    context_section: str
    immediate_section: str
    query: str
    full_prompt: str
    token_usage: dict = field(default_factory=dict)
    included_items: list[str] = field(default_factory=list)  # IDs of included context items
    dropped_items: list[str] = field(default_factory=list)   # IDs that didn't fit


@dataclass
class ConfusionSignal:
    """Signal from response analysis indicating potential context gaps."""
    type: str  # "missing_info", "contradiction", "clarification_needed"
    detail: str
    boost_topics: list[str] = field(default_factory=list)


@dataclass
class MemoryLayer:
    """A named memory layer with its own items and token budget.
    
    Each layer gets an independent section in the prompt. Layers don't share budgets —
    if semantic uses 300 of its 500 token budget, the remaining 200 are NOT given to
    episodic. This prevents any single layer from dominating the context.
    """
    name: str           # e.g. "semantic", "episodic", "working", "procedural"
    label: str          # Human-readable label shown in the prompt section header
    items: list         # list[ContextItem] — pre-scored, highest relevance first
    budget: int         # Max tokens this layer can use in the prompt
    priority: int = 0   # Higher priority layers are packed first (get budget priority)


class Translator:
    """
    Builds optimized prompts from PANIC state and processes LLM responses.
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self._tokenizer = tiktoken.get_encoding(self.config.tiktoken_encoding)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def construct_prompt(
        self,
        query: str,
        immediate_buffer: list[ContextItem],
        retrieved_items: list[ContextItem],
        working_summaries: list[ContextItem],
        system_prompt: str = "You are a helpful assistant.",
    ) -> ConstructedPrompt:
        """
        Build an optimized prompt from PANIC state.

        Ordering exploits LLM attention patterns:
          - Top: highest-relevance cold store retrievals (high attention)
          - Middle: working summaries + lower-relevance retrievals (lower attention)
          - Bottom: immediate buffer verbatim turns (high attention + recency)
          - End: current query (highest attention)

        Args:
            query: Current user message.
            immediate_buffer: Recent turns, verbatim. Already ordered chronologically.
            retrieved_items: Cold store retrievals, ordered by relevance (highest first).
            working_summaries: Compressed summaries from working state.
            system_prompt: System prompt text.

        Returns:
            ConstructedPrompt with assembled sections and usage stats.
        """
        # Accurate token counts
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)

        # Available budget
        total_budget = self.config.max_context_tokens
        reserved = system_tokens + query_tokens + self.config.response_headroom
        available = max(0, total_budget - reserved)

        if available == 0:
            return ConstructedPrompt(
                system_prompt=system_prompt,
                context_section="",
                immediate_section="",
                query=query,
                full_prompt=self._assemble("", "", query, system_prompt),
                token_usage={
                    "system": system_tokens,
                    "query": query_tokens,
                    "context": 0,
                    "immediate": 0,
                    "headroom": self.config.response_headroom,
                    "total": system_tokens + query_tokens,
                    "budget": total_budget,
                },
            )

        # Budget split: immediate buffer gets up to 30% of available
        immediate_budget = int(available * self.config.immediate_buffer_max_ratio)
        context_budget = available - immediate_budget

        # Pack immediate buffer (bottom of prompt, high attention zone)
        immediate_items, immediate_text, immediate_used = self._pack_items(
            immediate_buffer, immediate_budget
        )

        # Reclaim unused immediate budget
        context_budget += (immediate_budget - immediate_used)

        # Pack retrieved items (top of prompt, high attention zone)
        retrieved_packed, retrieved_text, retrieved_used = self._pack_items(
            retrieved_items, context_budget
        )
        context_budget -= retrieved_used

        # Pack working summaries (middle, lower attention — fine for supplementary info)
        summary_packed, summary_text, summary_used = self._pack_items(
            working_summaries, context_budget
        )

        # Build context section (retrieved first, then summaries)
        context_parts = []
        if retrieved_text:
            context_parts.append("[Historical context — relevant prior discussion]")
            context_parts.append(retrieved_text)
        if summary_text:
            context_parts.append("[Context summaries]")
            context_parts.append(summary_text)
        context_section = "\n\n".join(context_parts)

        # Build immediate section
        immediate_section = ""
        if immediate_text:
            immediate_section = "[Recent conversation]\n" + immediate_text

        # Assemble
        full_prompt = self._assemble(context_section, immediate_section, query, system_prompt)

        included = [i.id for i in retrieved_packed + summary_packed + immediate_items]
        all_ids = set(i.id for i in retrieved_items + working_summaries + immediate_buffer)
        dropped = [i for i in all_ids if i not in set(included)]

        total_tokens = system_tokens + query_tokens + retrieved_used + summary_used + immediate_used

        return ConstructedPrompt(
            system_prompt=system_prompt,
            context_section=context_section,
            immediate_section=immediate_section,
            query=query,
            full_prompt=full_prompt,
            token_usage={
                "system": system_tokens,
                "query": query_tokens,
                "retrieved": retrieved_used,
                "summaries": summary_used,
                "immediate": immediate_used,
                "headroom": self.config.response_headroom,
                "total": total_tokens,
                "budget": total_budget,
                "utilization": total_tokens / total_budget if total_budget > 0 else 0,
            },
            included_items=included,
            dropped_items=dropped,
        )

    def construct_layered_prompt(
        self,
        query: str,
        layers: list["MemoryLayer"],
        immediate_buffer: list[ContextItem],
        system_prompt: str = "You are a helpful assistant.",
    ) -> ConstructedPrompt:
        """
        Build a prompt with separate labeled sections per memory layer.

        Each layer gets its own token budget and labeled section in the prompt.
        Layers are packed in priority order (highest first). Unused budget from
        one layer is NOT redistributed — each layer is independently capped.

        Prompt ordering exploits LLM attention patterns ("lost in the middle" effect):
          - Top: working memory (highest priority, strong primacy attention)
          - Upper: semantic memory (durable knowledge)
          - Middle: episodic memory (past sessions — lower attention zone, acceptable)
          - Lower: procedural memory (patterns — supplementary, ok if partially missed)
          - Bottom: immediate buffer (recent turns, strong recency attention)
          - End: current query (highest attention)

        Priority controls pack order: higher priority layers are packed first,
        so if the total budget is tight, lower priority layers get squeezed.
        """
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)

        total_budget = self.config.max_context_tokens
        reserved = system_tokens + query_tokens + self.config.response_headroom
        available = max(0, total_budget - reserved)

        if available == 0:
            return ConstructedPrompt(
                system_prompt=system_prompt,
                context_section="",
                immediate_section="",
                query=query,
                full_prompt=self._assemble("", "", query, system_prompt),
                token_usage={
                    "system": system_tokens, "query": query_tokens,
                    "context": 0, "immediate": 0,
                    "headroom": self.config.response_headroom,
                    "total": system_tokens + query_tokens, "budget": total_budget,
                },
            )

        # Reserve immediate buffer budget (up to 30% of available)
        immediate_budget = int(available * self.config.immediate_buffer_max_ratio)
        layer_budget_total = available - immediate_budget

        # Pack immediate buffer
        imm_packed, imm_text, imm_used = self._pack_items(immediate_buffer, immediate_budget)

        # Pack each layer independently within its own budget.
        # Sort by priority descending so high-priority layers get packed first
        # and claim their share of the total available space.
        sorted_layers = sorted(layers, key=lambda l: l.priority, reverse=True)

        layer_sections = []  # (name, label, text, tokens_used, included, dropped)
        total_layer_used = 0
        all_included = []
        all_dropped = []

        for layer in sorted_layers:
            if not layer.items:
                continue

            # Cap at both the layer's own budget and remaining total
            effective_budget = min(layer.budget, layer_budget_total - total_layer_used)
            if effective_budget <= 0:
                all_dropped.extend(i.id for i in layer.items)
                continue

            packed, text, used = self._pack_items(layer.items, effective_budget)
            total_layer_used += used

            all_included.extend(i.id for i in packed)
            dropped_ids = [i.id for i in layer.items if i.id not in {p.id for p in packed}]
            all_dropped.extend(dropped_ids)

            if text:
                layer_sections.append((layer.name, layer.label, text, used))

        # Build context section with labeled layers
        context_parts = []
        for name, label, text, used in layer_sections:
            context_parts.append(f"## {label}\n[{name} memory — {used} tokens]")
            context_parts.append(text)
        context_section = "\n\n".join(context_parts)

        # Build immediate section
        immediate_section = ""
        if imm_text:
            immediate_section = "## Current Conversation\n[working memory]\n" + imm_text

        all_included.extend(i.id for i in imm_packed)

        full_prompt = self._assemble(context_section, immediate_section, query, system_prompt)

        total_tokens = system_tokens + query_tokens + total_layer_used + imm_used

        # Per-layer usage breakdown
        layer_usage = {}
        for name, label, text, used in layer_sections:
            layer_usage[name] = used

        return ConstructedPrompt(
            system_prompt=system_prompt,
            context_section=context_section,
            immediate_section=immediate_section,
            query=query,
            full_prompt=full_prompt,
            token_usage={
                "system": system_tokens,
                "query": query_tokens,
                "immediate": imm_used,
                "layers": layer_usage,
                "headroom": self.config.response_headroom,
                "total": total_tokens,
                "budget": total_budget,
                "utilization": total_tokens / total_budget if total_budget > 0 else 0,
            },
            included_items=all_included,
            dropped_items=all_dropped,
        )

    def analyze_response(self, response: str) -> list[ConfusionSignal]:
        """
        Analyze LLM response for confusion signals.

        Detects patterns that suggest PANIC should adjust retrieval on the next turn.
        """
        signals = []
        response_lower = response.lower()

        # Missing information signals
        missing_patterns = [
            "i don't have information",
            "i don't have access to",
            "i'm not sure about",
            "i don't recall",
            "you haven't mentioned",
            "could you clarify",
            "could you provide more",
            "i need more context",
            "without more information",
            "based on what you've told me",
        ]
        for pattern in missing_patterns:
            if pattern in response_lower:
                signals.append(ConfusionSignal(
                    type="missing_info",
                    detail=f"LLM indicated missing information: '{pattern}'",
                ))
                break  # one signal per type

        # Clarification request signals
        clarification_patterns = [
            "did you mean",
            "are you referring to",
            "which one do you mean",
            "can you be more specific",
            "to clarify",
        ]
        for pattern in clarification_patterns:
            if pattern in response_lower:
                signals.append(ConfusionSignal(
                    type="clarification_needed",
                    detail=f"LLM requested clarification: '{pattern}'",
                ))
                break

        return signals

    def _pack_items(
        self, items: list[ContextItem], budget: int
    ) -> tuple[list[ContextItem], str, int]:
        """
        Pack as many items as possible into the token budget.

        Returns (packed_items, combined_text, tokens_used).
        """
        packed = []
        parts = []
        used = 0

        for item in items:
            item_tokens = self.count_tokens(item.text)
            if used + item_tokens > budget:
                break  # no more room
            packed.append(item)
            parts.append(item.text)
            used += item_tokens

        return packed, "\n\n".join(parts), used

    def _assemble(
        self,
        context_section: str,
        immediate_section: str,
        query: str,
        system_prompt: str,
    ) -> str:
        """Assemble the final prompt string."""
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if context_section:
            parts.append(context_section)

        if immediate_section:
            parts.append(immediate_section)

        parts.append(f"[Current message]\n{query}")

        return "\n\n".join(parts)
