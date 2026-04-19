"""
PANIC Extraction Pipeline — Session-End Memory Consolidation

At session end, one LLM call generates:
  1. Episodic summary (→ episodes/YYYY-MM-DD.md)
  2. Semantic updates (→ semantic/entities.md, facts.md, preferences.md)
  3. Procedural updates (→ procedural/workflows.md, failures.md)

Also supports periodic intermediate extraction (~every 50 turns) for
crash protection during long sessions.

Design principles:
  - One LLM call produces all three outputs (cost efficient)
  - Existing markdown content is preserved; new content is appended
  - Soft deduplication: only skip obviously identical entries
  - The pipeline is stateless — reads engine state + profile files, writes results
"""

import json
import re
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger("panic.extraction")


# --- Extraction prompt ---

# This prompt is sent once at session end. It produces ALL memory updates in one call:
# episode summary, entity updates, fact updates, preference updates, workflow updates, failure updates.
# Keeping it to one call minimizes LLM cost (vs separate calls per memory type).
EXTRACTION_PROMPT = """You are a memory consolidation system. Analyze this conversation and extract structured memory updates.

Return a JSON object (no markdown fences) with these sections:

{{
  "summary": "<2-4 sentence summary of what happened in this conversation>",
  "key_facts": ["<fact 1>", "<fact 2>", ...],
  "important_moments": ["<turn N: what happened>", ...],
  "entities_mentioned": ["<entity1>", "<entity2>", ...],
  "entity_updates": [
    {{"name": "<entity>", "property": "<key>", "value": "<value>"}}
  ],
  "fact_updates": [
    {{"category": "<topic>", "fact": "<statement>"}}
  ],
  "preference_updates": [
    {{"category": "<area>", "preference": "<what the user prefers>"}}
  ],
  "workflow_updates": [
    {{"name": "<workflow>", "step": "<what to do>"}}
  ],
  "failure_updates": [
    {{"name": "<what failed>", "reason": "<why>"}}
  ]
}}

Rules:
- Only extract information that was actually discussed or decided
- Prefer specific facts over vague summaries
- For entities, include relationships and properties mentioned
- For preferences, capture explicit user preferences and working style
- For workflows, capture tools, commands, or procedures that worked
- For failures, capture what was tried and why it didn't work
- If a section has nothing to extract, use an empty array []

EXISTING ENTITIES (do not duplicate these, only add new info):
{existing_entities}

EXISTING FACTS (do not duplicate these):
{existing_facts}

CONVERSATION ({turn_count} turns):
{conversation}"""


# --- Data classes ---

@dataclass
class ExtractionResult:
    """Result of running the extraction pipeline."""
    status: str = "ok"
    episode_written: bool = False
    episode_date: str = ""
    entities_added: int = 0
    facts_added: int = 0
    preferences_added: int = 0
    workflows_added: int = 0
    failures_added: int = 0
    llm_tokens_used: int = 0
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "episode_written": self.episode_written,
            "episode_date": self.episode_date,
            "entities_added": self.entities_added,
            "facts_added": self.facts_added,
            "preferences_added": self.preferences_added,
            "workflows_added": self.workflows_added,
            "failures_added": self.failures_added,
            "llm_tokens_used": self.llm_tokens_used,
            "duration_ms": round(self.duration_ms, 1),
        }


# --- Pipeline ---

class ExtractionPipeline:
    """
    Runs memory extraction at session end or periodically.

    Reads the engine's chat history + profile's existing memory files,
    calls an LLM to extract structured updates, and writes them to
    the profile's markdown files.
    """

    def __init__(self, profile_manager, engine):
        self.pm = profile_manager
        self.engine = engine
        self.profile_name = profile_manager.active_profile

    def run_session_end(self) -> dict:
        """
        Full session-end extraction.
        Returns a dict summary of what was extracted.
        """
        start = time.perf_counter()
        result = ExtractionResult()

        if not self.engine.chat_history:
            result.status = "skipped"
            result.error = "No conversation data"
            return result.to_dict()

        try:
            # Build the conversation text
            conv_text = self._build_conversation_text()

            # Read existing memory for dedup context
            existing_entities = self.pm.read_semantic(name=self.profile_name, file="entities.md")
            existing_facts = self.pm.read_semantic(name=self.profile_name, file="facts.md")

            # Call LLM
            prompt = EXTRACTION_PROMPT.format(
                existing_entities=self._truncate(existing_entities, 2000),
                existing_facts=self._truncate(existing_facts, 2000),
                turn_count=self.engine.turn,
                conversation=self._truncate(conv_text, 12000),
            )

            llm_result = self._call_extraction_llm(prompt)
            if not llm_result:
                result.status = "failed"
                result.error = "LLM returned empty result"
                return result.to_dict()

            result.llm_tokens_used = llm_result.get("_tokens", 0)

            # Write episodic memory
            self._write_episode(llm_result, result)

            # Update semantic memory
            self._update_entities(llm_result, result)
            self._update_facts(llm_result, result)
            self._update_preferences(llm_result, result)

            # Update procedural memory
            self._update_workflows(llm_result, result)
            self._update_failures(llm_result, result)

        except Exception as e:
            logger.error(f"Extraction pipeline failed: {e}", exc_info=True)
            result.status = "failed"
            result.error = str(e)

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result.to_dict()

    def run_intermediate(self) -> dict:
        """
        Intermediate extraction (every ~50 turns).
        Only writes an episode append — doesn't update semantic/procedural.
        """
        start = time.perf_counter()
        result = ExtractionResult()

        if not self.engine.chat_history:
            result.status = "skipped"
            return result.to_dict()

        try:
            conv_text = self._build_conversation_text(last_n_turns=50)

            prompt = f"""Summarize the last portion of this conversation in 2-3 sentences.
Return JSON: {{"summary": "<summary>", "key_facts": ["<fact>", ...]}}

CONVERSATION (last ~50 turns):
{self._truncate(conv_text, 8000)}"""

            llm_result = self._call_extraction_llm(prompt)
            if llm_result:
                result.llm_tokens_used = llm_result.get("_tokens", 0)
                # Append to today's episode
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                existing = self.pm.read_episode(name=self.profile_name, date=date_str)

                summary = llm_result.get("summary", "")
                facts = llm_result.get("key_facts", [])

                if summary:
                    append_text = f"\n\n## Intermediate (turn {self.engine.turn})\n\n{summary}\n"
                    if facts:
                        append_text += "\n### Key Facts\n"
                        for f in facts:
                            append_text += f"- {f}\n"

                    self.pm.write_episode(
                        (existing + append_text) if existing else append_text,
                        date_str,
                        name=self.profile_name,
                    )
                    result.episode_written = True
                    result.episode_date = date_str

        except Exception as e:
            logger.error(f"Intermediate extraction failed: {e}", exc_info=True)
            result.status = "failed"
            result.error = str(e)

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result.to_dict()

    # --- LLM call ---

    def _call_extraction_llm(self, prompt: str) -> Optional[dict]:
        """Call the extraction LLM and parse JSON response."""
        model = self.engine.extraction_model
        api_key = self.engine.extraction_api_key or self.engine.api_key

        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a memory extraction system. Return valid JSON only, no markdown fences."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                api_key=api_key if api_key else None,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            data = json.loads(raw)

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                data["_tokens"] = response.usage.total_tokens

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Extraction LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Extraction LLM call failed: {e}")
            return None

    # --- Conversation builder ---

    def _build_conversation_text(self, last_n_turns: Optional[int] = None) -> str:
        """Build a text representation of the conversation."""
        history = self.engine.chat_history
        if last_n_turns:
            # Get last N entries (each turn has user + assistant = 2 entries)
            history = history[-(last_n_turns * 2):]

        parts = []
        for entry in history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            turn = entry.get("turn", 0)
            # Truncate very long individual messages
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"[Turn {turn} - {role}]: {content}")

        return "\n".join(parts)

    # --- Episode writer ---

    def _write_episode(self, data: dict, result: ExtractionResult):
        """Write episodic memory (session summary).
        If an episode already exists for today (e.g. multiple sessions), appends with a separator.
        """
        summary = data.get("summary", "")
        if not summary:
            return

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key_facts = data.get("key_facts", [])
        important_moments = data.get("important_moments", [])
        entities = data.get("entities_mentioned", [])

        # Build episode markdown
        lines = [
            f"# Episode: {date_str} — Session Summary",
            "",
            "## Metadata",
            f"- **Date:** {date_str}",
            f"- **Profile:** {self.profile_name}",
            f"- **Turns:** {self.engine.turn}",
            f"- **Model:** {self.engine.model or 'not connected'}",
            "",
            "## Summary",
            summary,
            "",
        ]

        if key_facts:
            lines.append("## Key Facts Established")
            for fact in key_facts:
                lines.append(f"- {fact}")
            lines.append("")

        if important_moments:
            lines.append("## Important Moments")
            for moment in important_moments:
                lines.append(f"- {moment}")
            lines.append("")

        if entities:
            lines.append("## Entities")
            lines.append(f"- {', '.join(entities)}")
            lines.append("")

        episode_content = "\n".join(lines)

        # Check if there's already an episode for today (append)
        existing = self.pm.read_episode(name=self.profile_name, date=date_str)
        if existing:
            # Append as a new session section
            episode_content = existing.rstrip() + "\n\n---\n\n" + episode_content

        self.pm.write_episode(episode_content, date_str, name=self.profile_name)
        result.episode_written = True
        result.episode_date = date_str

    # --- Semantic updaters ---

    def _update_entities(self, data: dict, result: ExtractionResult):
        """Append new entity info to semantic/entities.md.
        Uses simple case-insensitive dedup: if 'name: value' already appears in
        the file, skip it. Intentionally permissive — storage is cheap, nuance matters.
        """
        updates = data.get("entity_updates", [])
        if not updates:
            return

        existing = self.pm.read_semantic(name=self.profile_name, file="entities.md")
        existing_lower = existing.lower()

        new_entries = []
        for update in updates:
            name = update.get("name", "").strip()
            prop = update.get("property", "").strip()
            value = update.get("value", "").strip()
            if not name or not value:
                continue

            # Simple dedup: check if this property+value combo already exists.
            # We check the lowercased file content to catch case-insensitive matches.
            check = f"{name.lower()}: {value.lower()}" if not prop else f"{prop.lower()}: {value.lower()}"
            if check in existing_lower:
                continue

            new_entries.append((name, prop, value))

        if not new_entries:
            return

        # Group by entity name
        by_name = {}
        for name, prop, value in new_entries:
            by_name.setdefault(name, []).append((prop, value))

        # Build append text
        lines = []
        for name, props in by_name.items():
            # Check if entity heading already exists
            heading = f"## {name}"
            if heading.lower() not in existing_lower:
                lines.append(f"\n## {name}")

            for prop, value in props:
                if prop:
                    lines.append(f"- {prop}: {value}")
                else:
                    lines.append(f"- {value}")

        if lines:
            append_text = "\n".join(lines) + "\n"
            self.pm.append_semantic(append_text, name=self.profile_name, file="entities.md")
            result.entities_added = len(new_entries)

    def _update_facts(self, data: dict, result: ExtractionResult):
        """Append new facts to semantic/facts.md."""
        updates = data.get("fact_updates", [])
        if not updates:
            return

        existing = self.pm.read_semantic(name=self.profile_name, file="facts.md")
        existing_lower = existing.lower()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        new_facts = []
        for update in updates:
            fact = update.get("fact", "").strip()
            category = update.get("category", "General").strip()
            if not fact:
                continue

            # Simple dedup
            if fact.lower() in existing_lower:
                continue

            new_facts.append((category, fact))

        if not new_facts:
            return

        by_category = {}
        for cat, fact in new_facts:
            by_category.setdefault(cat, []).append(fact)

        lines = []
        for cat, facts in by_category.items():
            heading = f"## {cat}"
            if heading.lower() not in existing_lower:
                lines.append(f"\n## {cat}")
            for fact in facts:
                lines.append(f"- {fact} — Established: {date_str}")

        if lines:
            self.pm.append_semantic("\n".join(lines) + "\n", name=self.profile_name, file="facts.md")
            result.facts_added = len(new_facts)

    def _update_preferences(self, data: dict, result: ExtractionResult):
        """Append new preferences to semantic/preferences.md."""
        updates = data.get("preference_updates", [])
        if not updates:
            return

        existing = self.pm.read_semantic(name=self.profile_name, file="preferences.md")
        existing_lower = existing.lower()

        new_prefs = []
        for update in updates:
            pref = update.get("preference", "").strip()
            category = update.get("category", "General").strip()
            if not pref:
                continue

            if pref.lower() in existing_lower:
                continue

            new_prefs.append((category, pref))

        if not new_prefs:
            return

        by_category = {}
        for cat, pref in new_prefs:
            by_category.setdefault(cat, []).append(pref)

        lines = []
        for cat, prefs in by_category.items():
            heading = f"## {cat}"
            if heading.lower() not in existing_lower:
                lines.append(f"\n## {cat}")
            for pref in prefs:
                lines.append(f"- {pref}")

        if lines:
            self.pm.append_semantic("\n".join(lines) + "\n", name=self.profile_name, file="preferences.md")
            result.preferences_added = len(new_prefs)

    # --- Procedural updaters ---

    def _update_workflows(self, data: dict, result: ExtractionResult):
        """Append new workflows to procedural/workflows.md."""
        updates = data.get("workflow_updates", [])
        if not updates:
            return

        existing = self.pm.read_procedural(name=self.profile_name, file="workflows.md")
        existing_lower = existing.lower()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        new_workflows = []
        for update in updates:
            name = update.get("name", "").strip()
            step = update.get("step", "").strip()
            if not step:
                continue

            if step.lower() in existing_lower:
                continue

            new_workflows.append((name or "General", step))

        if not new_workflows:
            return

        by_name = {}
        for name, step in new_workflows:
            by_name.setdefault(name, []).append(step)

        lines = []
        for name, steps in by_name.items():
            heading = f"## {name}"
            if heading.lower() not in existing_lower:
                lines.append(f"\n## {name}")
            for step in steps:
                lines.append(f"- {step}")
            lines.append(f"- Established: {date_str}")

        if lines:
            self.pm.write_procedural(
                existing.rstrip() + "\n" + "\n".join(lines) + "\n",
                name=self.profile_name,
                file="workflows.md",
            )
            result.workflows_added = len(new_workflows)

    def _update_failures(self, data: dict, result: ExtractionResult):
        """Append new failures to procedural/failures.md."""
        updates = data.get("failure_updates", [])
        if not updates:
            return

        existing = self.pm.read_procedural(name=self.profile_name, file="failures.md")
        existing_lower = existing.lower()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        new_failures = []
        for update in updates:
            name = update.get("name", "").strip()
            reason = update.get("reason", "").strip()
            if not name or not reason:
                continue

            if name.lower() in existing_lower and reason.lower() in existing_lower:
                continue

            new_failures.append((name, reason))

        if not new_failures:
            return

        lines = []
        for name, reason in new_failures:
            lines.append(f"\n## {name}")
            lines.append(f"- {reason}")
            lines.append(f"- Date: {date_str}")

        if lines:
            self.pm.write_procedural(
                existing.rstrip() + "\n" + "\n".join(lines) + "\n",
                name=self.profile_name,
                file="failures.md",
            )
            result.failures_added = len(new_failures)

    # --- Helpers ---

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text to max characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[... truncated ...]"
