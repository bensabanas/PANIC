# PANIC — Product Specification

(PANIC is a reference to the Hitchhiker's Guide to the Galaxy)
All feedback and comments are welcome.

## Overview

PANIC is a persistent memory system for AI assistants. It sits between the user and any LLM, providing genuine long-term conversational memory that persists across sessions, improves over time, and runs entirely locally.

The LLM has no awareness of PANIC. From its perspective, it receives a well-constructed prompt with relevant context. PANIC's intelligence is in deciding *what* goes into that prompt and *where* that context comes from — the current conversation, previous sessions, or accumulated knowledge.

PANIC is designed as an OpenClaw plugin but architecturally model-agnostic and runtime-agnostic.

## What Makes PANIC Different

Standard memory solutions (RAG, MemGPT, LangChain memory) are stateless retrieval: store text chunks, search by similarity, inject matches. They treat turn 5 and turn 500 identically if their embeddings are similar.

PANIC adds:
- **Dual-engine knowledge graphs** — rule-based (precise, contradiction-aware) + LLM-based (dense, multi-hop) extraction running in parallel
- **Multi-layer memory** — working memory, episodic memory, semantic knowledge, and procedural patterns, each serving a different retrieval need
- **Layered context injection** — the LLM sees clearly labeled sections showing where each piece of context came from
- **Importance heuristics** — recency weighting, graph connectivity, and contradiction detection identify what matters without heavyweight ML components
- **Persistent knowledge** — entity graphs and accumulated facts carry across sessions, not just embeddings

## Architecture

```
User message
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    PANIC LAYER                        │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  WORKING MEMORY (current session, volatile)      │ │
│  │  ├── Turn embeddings (all-MiniLM-L6-v2, 384d)   │ │
│  │  ├── Recency-weighted recent turns               │ │
│  │  └── Session entity graph (rule + LLM extracted) │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  EPISODIC MEMORY (per session, stored as .md)    │ │
│  │  ├── Session summaries (LLM-generated)           │ │
│  │  ├── Key facts established per session           │ │
│  │  ├── Important moments (high-relevance turns)    │ │
│  │  ├── Entities mentioned                          │ │
│  │  └── Metadata (date, duration, model, profile)   │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  SEMANTIC MEMORY (cross-session, accumulated)    │ │
│  │  ├── Entity knowledge (people, projects, places) │ │
│  │  ├── Established facts (soft-deduplicated)       │ │
│  │  └── User preferences & patterns                 │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  PROCEDURAL MEMORY (cross-session, learned)      │ │
│  │  ├── Workflows (tools, approaches, solutions)    │ │
│  │  └── Failures (what didn't work and why)         │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  RETRIEVAL + CONTEXT INJECTION                   │ │
│  │  ├── Cosine similarity + graph boost (0.75/0.20) │ │
│  │  ├── Blended embeddings (0.7 item / 0.3 turn)   │ │
│  │  ├── Contradiction penalty + recency tiebreaker  │ │
│  │  ├── Multi-hop graph expansion                   │ │
│  │  └── Layered injection with per-layer budgets    │ │
│  └──────────────────────────────────────────────────┘ │
│                                                       │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
                    LLM prompt
```

## Memory Layers — Detailed

### Layer 1: Working Memory (per-session, volatile until checkpointed)

Working memory tracks the current session's conversation state using embeddings and knowledge graphs.

**Encoder:**
- all-MiniLM-L6-v2 (384-dimensional embeddings)
- Each turn is encoded and stored with its turn number

**Blended embeddings:**
- Candidate embeddings blend item text (70%) with turn context (30%)
- This ratio was the single biggest retrieval improvement found across all experiments (+28pp multi-hop)
- Rationale: pure item embeddings preserve needle signal; turn embeddings add context but dilute precision at higher ratios

**Session entity graph:**
- Rule-based + LLM-based entity/relationship extraction per turn
- Two parallel graphs: rule graph (sparse, precise, contradiction-aware) and LLM graph (dense, high-recall multi-hop)
- Enables multi-hop reasoning within the current session
- Graph-boosted retrieval augments cosine similarity with entity connectivity

**Immediate buffer:**
- Last N turns kept verbatim for the LLM's immediate context window
- Configurable buffer size (default: 10 turns)

**Persistence:**
- Entity graph state is checkpointed to disk at session end and periodically
- Turn embeddings are stored for cross-session retrieval
- On session resume (same profile), graph state is restored

### Layer 2: Episodic Memory (per session, stored as markdown)

Each session produces an episode file:

```markdown
# Episode: 2026-04-15 — PANIC Product Architecture

## Metadata
- **Date:** 2026-04-15
- **Profile:** panic-dev
- **Turns:** 127
- **Duration:** 2h 15m
- **Model:** claude-opus-4

## Summary
Discussed refactoring PANIC from eval harness to OpenClaw plugin.
Decided on 4-layer memory architecture: working, episodic, semantic,
procedural. Confirmed layered context injection with token budgets
and markdown storage.

## Key Facts Established
- PANIC will be an OpenClaw plugin
- Storage format: markdown files (not JSON/SQLite)
- Profiles enable separate memory stores per context
- Retrieval uses layered injection with per-layer token budgets

## Important Moments
- Turn 23: Decision to use markdown over JSON (shifted storage architecture)
- Turn 45: Confirmed persistent state across sessions as product differentiator
- Turn 61: Confirmation of layered injection with budgets

## Entities
- PANIC, OpenClaw, Ben, profiles, markdown
```

**Extraction timing:**
- LLM summary generated at session end
- Intermediate extraction every ~50 turns during long sessions (crash protection)
- Intermediate extractions are appended; final extraction consolidates

### Layer 3: Semantic Memory (cross-session, accumulated)

Accumulated knowledge stored in markdown files within the profile:

**entities.md** — People, projects, places, tools, and their relationships.
```markdown
## Ben
- Role: Developer, PANIC creator
- Timezone: Europe/Vienna
- Communication: Blunt, factual, prefers no filler
- First mentioned: 2026-04-14

## PANIC
- Type: Software project
- Location: /Users/ben/Desktop/panic/
- Language: Python 3.13
- Purpose: Persistent memory system for AI assistants
- Related: OpenClaw (target platform)
```

**facts.md** — Established knowledge, soft-deduplicated.
```markdown
## Architecture
- Cosine similarity on embeddings does 55-75% of retrieval work
- Graph boost adds 15-20% improvement via entity connectivity
- Blended embeddings (0.7 item / 0.3 turn) is the key retrieval lever
- Established: 2026-04-19

## Decisions
- Markdown files for storage, not JSON/SQLite — Established: 2026-04-15
- No automatic memory decay — Established: 2026-04-15
```

**preferences.md** — User patterns and preferences.
```markdown
## Work Style
- Experiments tested one at a time with report + confirmation before proceeding
- Prefers manual control with good UI (dropdown) over full automation
- Values transparency: wants to see and edit stored memory

## Communication
- Blunt, factual, no filler
- Wants polite formal behavior, not cheerful
```

**Extraction:** Semantic facts are extracted heuristically per turn (from the graph engine's entity/relationship triples — zero extra LLM cost). Preferences are extracted by LLM at session end, bundled with the episodic summary call.

**Soft deduplication:** Only merge when obviously the same entity + relationship + value. Slightly different phrasings are kept as separate entries. Storage is cheap; nuance is not.

### Layer 4: Procedural Memory (cross-session, learned)

**workflows.md** — How the user works.
```markdown
## PANIC Development
- Run eval: `python evaluate_v5.py smoke|standard|stress|marathon`
- Always use PTY and DYLD_LIBRARY_PATH for Python
- Test one experiment at a time, report results, wait for confirmation
- Established: 2026-04-14
```

**failures.md** — What didn't work and why.
```markdown
## Reservoir / LSM for Retrieval
- Extensive testing (phases 1-5, experiments #1-#8, TenTenTen flux)
- Reservoir state has NO discriminative power for retrieval
- Cannot tell planted facts from filler
- Readout MLP, reranker, pattern separator, cerebellar expansion — all dead weight
- Simplification to embeddings + graph matched or beat full reservoir pipeline
- Date: 2026-04-19

## Experiment #4: Cerebellar Expansion
- 384→12288 over-separated inputs, killed working memory surprise discriminative power
- Date: 2026-04-14
```

**Extraction:** LLM-generated at session end, bundled with episodic summary. One LLM call produces episodic + preferences + procedural updates.

## Retrieval & Context Injection

### Retrieval Pipeline

When the user sends a message, PANIC queries all memory layers:

1. **Working memory:** Cosine similarity on blended embeddings + graph boost + contradiction penalty + recency, scored against current session turns
2. **Episodic:** Embed the query, search episode summaries and key facts across all sessions
3. **Semantic:** Keyword + embedding search over entities, facts, preferences
4. **Procedural:** Match against current context (working directory, active tools, recent patterns)

Each layer returns its top candidates with relevance scores.

### Scoring Formula (Working Memory)

```
score = 0.75 * cosine_similarity
      + 0.20 * graph_boost
      + 0.05 * recency
      - contradiction_penalty (0.3 for superseded/contradicted facts)
```

**Graph boost blending:** The rule-based and LLM-based graph boosts are blended based on query type:
- Single-fact queries → lean toward rule graph (precise)
- Multi-hop queries → lean toward LLM graph (higher recall)
- A lightweight keyword classifier scores query multi-hop likelihood (0.0–1.0)

**Multi-hop expansion:** Top-8 candidates are expanded through the entity graph, boosting connected facts by +0.1 per hop.

### Layered Context Injection

Results are NOT merged into a single ranked list. Each layer gets its own labeled section and token budget:

```markdown
## Long-term Knowledge
[semantic memory — budget: 500 tokens]
- Cosine similarity on embeddings does 55-75% of retrieval work
- Graph boost adds entity connectivity for multi-hop reasoning
- Ben prefers experiments tested one at a time

## Previous Sessions
[episodic memory — budget: 1000 tokens]
- Apr 14: Tested lateral inhibition experiment #8. Result: 200t multi-hop
  improved to avg 72% (from 55% baseline). Kept despite 500t regression.
- Apr 14: Discarded experiment #4 (cerebellar expansion) — over-separation.

## Current Conversation
[working memory — budget: 2000 tokens]
- Turn 34: Asked about embedding blend ratios
- Turn 41: Decided on markdown storage over JSON
- Turn 45: Confirmed persistent state across sessions

## Relevant Patterns
[procedural memory — budget: 300 tokens]
- PANIC dev workflow: test one thing → report → wait for confirmation
```

**Benefits:**
- LLM sees provenance: "I know this from a previous session" vs "you said this 20 turns ago"
- Debuggable: users can inspect exactly what was injected
- Tunable: budgets can be adjusted per profile (coding profile → more procedural; personal → more episodic)
- Prevents any single layer from dominating the context

**Token budgets are configurable per profile.** Defaults:
- Semantic: 500 tokens
- Episodic: 1000 tokens
- Working (current session): 2000 tokens
- Procedural: 300 tokens
- Total injected context: ~3800 tokens (well within any model's capacity)

## Profiles

Profiles enable separate memory stores for different contexts.

### Storage Structure
```
~/.openclaw/panic/
├── profiles/
│   ├── default/
│   │   ├── graph.state              # Entity graph (rule + LLM)
│   │   ├── embeddings.state         # Turn embeddings index
│   │   ├── episodes/
│   │   │   ├── 2026-04-14.md
│   │   │   └── 2026-04-15.md
│   │   ├── semantic/
│   │   │   ├── entities.md
│   │   │   ├── facts.md
│   │   │   └── preferences.md
│   │   └── procedural/
│   │       ├── workflows.md
│   │       └── failures.md
│   ├── panic-dev/
│   │   └── ... (same structure)
│   └── personal/
│       └── ... (same structure)
└── config.md                        # Global PANIC settings
```

### Profile Operations
- **Create** — new name, empty memory stores
- **Switch** — load different profile (manual, via dropdown UI)
- **Clone** — copy a profile for branching
- **Delete** — remove profile and all memory (with confirmation)
- **Export / Import** — backup or share (all markdown + state files)

### Cross-Profile Queries
By default, retrieval is scoped to the active profile. Optional cross-profile search can surface relevant context from other profiles ("I discussed something similar in another project"), but this is off by default.

## Extraction Pipeline

### Per-Turn (zero LLM cost)
- Embedding generation (all-MiniLM-L6-v2)
- Entity/relationship extraction (heuristic, from graph engine)
- Semantic fact accumulation (heuristic)

### Periodic (~every 50 turns during long sessions)
- Intermediate episodic summary (LLM call)
- Graph state checkpoint

### Session End (one LLM call)
- Final episodic summary (consolidates any intermediate summaries)
- Preference extraction
- Procedural pattern extraction
- Final state checkpoint (graph + embeddings)

## Performance Benchmarks (v5 simplified, from eval harness)

| Metric | 50 turns | 200 turns | 500 turns |
|---|---|---|---|
| Fact retrieval | 100% | 100% | 97% |
| Multi-hop reasoning | 100% | ~60-80% (LLM graph variance) | 74% |
| Contradiction detection | 100% | 80-100% | 66.7% |
| Context inclusion | 100% | ~93% | 89.2% |

**Latency:** 8-11ms per retrieval (was 21-60ms with the old reservoir-based pipeline).

Note: These benchmarks are from synthetic evaluation data. Real conversation performance will differ and needs separate validation.

### Why No Reservoir

The reservoir (Liquid State Machine) was extensively tested across 5 phases, 8 bio-inspired experiments, and a dedicated research branch (TenTenTen). Key findings:

- Reservoir state has **no discriminative power** for retrieval — cannot tell planted facts from filler
- Readout MLP, reranker, pattern separator, lateral inhibition, cerebellar expansion — all dead weight for retrieval
- Removing everything except embeddings + dual graph produces **equal or better accuracy at 5-6x the speed**
- The candidate embedding blend ratio (0.7/0.3) was the single biggest improvement (+28pp multi-hop), not any reservoir component
- LLM graph non-determinism is the dominant variance source, not architecture

The reservoir research continues separately in the TenTenTen project for non-retrieval applications (behavioral fingerprinting).

## Open Items

- [ ] OpenClaw plugin integration — hook points, message pipeline, UI surface
- [ ] Real conversation benchmarking (vs. synthetic eval)
- [ ] Cross-profile search implementation
- [ ] Token budget tuning per profile type
- [ ] LLM extraction model selection (cost vs quality tradeoff)
- [ ] Error handling: corrupt state files, partial writes, migration between versions
- [ ] Session end detection in OpenClaw (idle timeout vs explicit close)
