# PANIC — Project Specification

*"Don't Panic"* — Douglas Adams

## Overview

PANIC is a middleware layer that sits between a user and any LLM. It uses a hybrid Liquid State Machine (the "PANIC layer") to manage, compress, and selectively retrieve conversational context — effectively extending the LLM's usable memory far beyond its native context window.

The LLM has no awareness of PANIC. From its perspective, it receives a well-constructed prompt. PANIC's intelligence is in deciding *what* goes into that prompt.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              PANIC LAYER                     │
│                                              │
│  ┌────────────┐  ┌────────────────────────┐  │
│  │  Encoder   │──│   Reservoir (fixed)     │  │
│  │            │  │   80% vector-space      │  │
│  └────────────┘  │   20% structured graph  │  │
│                  └──────────┬─────────────┘  │
│                             │                │
│  ┌──────────────────────────┴──────────┐     │
│  │         Memory Tiers                │     │
│  │  ┌───────────┐ ┌─────────┐ ┌─────┐ │     │
│  │  │ Immediate │ │ Working │ │Cold │ │     │
│  │  │ (verbatim)│ │ (dense) │ │(KV) │ │     │
│  │  └───────────┘ └─────────┘ └─────┘ │     │
│  └──────────────────────────┬──────────┘     │
│                             │                │
│  ┌──────────────────────────┴──────────┐     │
│  │     Readout (learned selector)      │     │
│  └──────────────────────────┬──────────┘     │
│                             │                │
│  ┌──────────────────────────┴──────────┐     │
│  │     Compressor (learned)            │     │
│  └─────────────────────────────────────┘     │
│                                              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Translation Layer│
          │ (PANIC → Prompt) │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    LLM (any)    │
          │  via API (Opt A)│
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Translation Layer│
          │ (Response → PANIC│
          │  state update)   │
          └────────┬────────┘
                   │
                   ▼
            User Output
```

## LLM Integration

**Current: Option A — Prompt Stuffing (API wrapping)**
- PANIC constructs optimized prompts and sends them via standard LLM APIs
- Compatible with any LLM (OpenAI, Anthropic, local models)
- No model modifications required
- PANIC's value is in intelligent context selection, not in delivery mechanism

**Future: Option C — Attention Injection**
- Soft-prompt embeddings or auxiliary attention masks
- Requires local model access
- Higher performance ceiling, more complex implementation
- Planned as a later mode, not MVP

## PANIC Layer — Core Design

### 1. State Representation (80/20 Hybrid)

**Vector-space component (80%)**
The primary state representation. High-dimensional dense vectors that capture semantic meaning of conversational content. This is the core LSM advantage — a rich, continuous state space that captures patterns, associations, and temporal dynamics that discrete representations miss.

- Incoming text is encoded into the reservoir's vector space
- State evolves continuously as new input arrives
- Captures fuzzy associations, tone, topic drift, implicit context
- Lossy by nature — trades precision for density and pattern recognition

**Structured graph component (20%)**
Explicit entities, facts, decisions, and their relationships. The "cortical regions" — predefined functional areas that give structure to the chaos.

- Named entities and their properties
- Stated facts and decisions
- Causal relationships (X led to Y)
- References and coreferences (when "it" means the database, track that)
- Contradictions and corrections (user changed their mind about X)

The graph anchors the vector space. Without it, the reservoir drifts. Without the reservoir, the graph is too rigid to capture nuance.

### 2. State Update Function

**Approach: Hybrid (learned + rule-based), tuned empirically**

The update function determines how PANIC state changes when new input arrives. Initial design, subject to iteration:

- **Rule-based triggers:** Entity extraction, fact detection, contradiction detection feed the structured graph directly. These are deterministic.
- **Learned dynamics:** The reservoir's vector state updates through a learned transformation. The reservoir itself is fixed (random, stable projections), but the input encoding and state mixing weights are learned.
- **Compression trigger:** When working memory reaches a threshold, the learned compressor fires — summarizing and merging older state entries into denser representations.

This split will be iterated on. The boundary between rule-based and learned will shift based on what performs better in practice.

### 3. Reservoir Architecture

#### Core Dynamics

The reservoir is the "liquid" in the Liquid State Machine. It works like dropping pebbles into water — each input creates ripples that interact with ripples from previous inputs. The surface pattern at any moment encodes a temporal history of everything that's passed through.

Reservoir state update per turn:

```
S(t) = f(W_res · S(t-1) + W_in · x(t) + noise)
```

- **S(t)** = reservoir state after turn t
- **S(t-1)** = previous reservoir state
- **W_res** = reservoir weight matrix (fixed, random, initialized once, never trained)
- **W_in** = input projection matrix (fixed, random, maps encoder output into reservoir space)
- **x(t)** = encoded input at turn t
- **f** = tanh activation (keeps state bounded)
- **noise** = small stochastic perturbation (prevents state collapse into fixed points)

#### Why Fixed Weights

W_res and W_in are never trained. This is the core LSM property:

1. **Stability.** Trained recurrent matrices drift over time. Fixed ones don't. PANIC processes thousands of turns — trained weights would accumulate gradient artifacts.
2. **Separation property.** Random high-dimensional projections naturally map different input sequences to different state trajectories (Johnson-Lindenstrauss territory).
3. **Fading memory.** Old inputs naturally decay as new inputs arrive. Decay rate is controlled by the spectral radius of W_res.

#### Spectral Radius (ρ) — The Memory Knob

The spectral radius (largest eigenvalue) of W_res controls how long past inputs influence current state:

- ρ < 1.0: State decays. Short memory, stable dynamics.
- ρ ≈ 1.0: Maximum memory capacity. Edge of chaos. Sensitive to input differences.
- ρ > 1.0: State explodes. Unusable.

#### Multi-Timescale Sub-Reservoirs

One reservoir with one ρ captures one timescale. Real conversations have multiple. PANIC uses three parallel sub-reservoirs concatenated into one state vector:

```
S(t) = [S_fast(t) | S_medium(t) | S_slow(t)]
```

| Sub-reservoir | ρ | Dimension | Temporal focus | Analogy |
|---|---|---|---|---|
| S_fast | 0.80 | 1,024 | Last ~5-10 turns | Working memory |
| S_medium | 0.93 | 1,024 | Last ~30-50 turns | Short-term memory |
| S_slow | 0.98 | 2,048 | Last ~200+ turns | Long-term memory |

Total state dimension: **D = 4,096**

The readout layer learns which timescale to weight for a given query. Biologically plausible — different brain regions operate at different timescales.

Spectral radius values per operating mode:

| Mode | S_fast ρ | S_medium ρ | S_slow ρ |
|---|---|---|---|
| Long Conversation (default) | 0.80 | 0.93 | 0.98 |
| Multi-Session Memory | 0.85 | 0.95 | 0.99 |
| Document Analysis | 0.75 | 0.88 | 0.95 |

These are starting values. The evaluation harness will determine final tuning.

#### Echo State Property

For correct function, the reservoir must satisfy the echo state property: the influence of initial state S(0) must wash out, so behavior is determined only by input history.

Requirements:
- Spectral radius < 1.0 (guaranteed by design)
- W_res is **sparse at 10% connectivity** (non-zero connections). Sparse reservoirs produce richer dynamics than dense ones at the same dimension. Sparsity level subject to tuning.
- W_in is dense but scaled so inputs don't saturate the tanh activation

#### Reservoir Dimensionality and Cost

| Target | Total D | Memory per state snapshot (float32) | Compute per turn |
|---|---|---|---|
| 200 turns | 4,096 | ~16KB | ~16M multiply-adds (<1ms on modern CPU) |
| 500 turns | 4,096 | ~16KB | Same |
| 1000+ turns | 4,096 (or 8,192 if needed) | ~16-32KB | <2ms |

The reservoir state is cheap. The matrix multiplications are not a bottleneck.

#### Input Encoding

**MVP: Off-the-shelf sentence embeddings**
- Use existing sentence-transformer (e.g., `all-MiniLM-L6-v2`, dimension 384)
- W_in projects from 384 → D (per sub-reservoir)
- Each turn encoded independently
- Fast, no training needed

**Future: Contextual encoder (learned)**
- Small transformer or RNN encoding current turn conditioned on recent turns
- Context-aware embedding before it hits the reservoir
- Captures cross-turn relationships at the input stage
- Requires training data
- Planned expansion, not MVP

The reservoir's temporal mixing partially compensates for independent per-turn encoding. If evaluation shows insufficient cross-turn structure capture, the contextual encoder becomes priority.

#### State Snapshots

Reservoir state S(t) is computed incrementally — S(200) cannot be recomputed without replaying all 200 inputs.

- **Snapshot every N turns** (default: every 10 turns). Store S(t) alongside turn data.
- On session resume (Mode 2), load latest snapshot instead of replaying full history.
- Enables "time travel" — readout can query reservoir state at past points, not just current moment.
- Snapshot storage: ~16KB per snapshot at D=4096. At every-10-turns, a 1000-turn session = 100 snapshots = ~1.6MB. Negligible.

#### Learned Readout

- Trained selector/ranker
- Given the current reservoir state + user query, decides which memory tier entries to retrieve
- Learns to weight sub-reservoirs (fast/medium/slow) differently depending on query type
- Outputs a relevance-ranked list of context items to include in the LLM prompt
- This is the primary trained component

#### Learned Compressor

- Merges and summarizes older state entries
- Reduces token footprint while preserving key information
- Operates on the boundary between immediate buffer and working state
- Also operates on working state → cold store transitions
- Implementation approach TBD — will be iterated empirically

### 4. Memory Tiers

**Immediate buffer**
- Last N turns, stored verbatim (full text)
- No compression, no loss
- Directly included in LLM prompts when context window allows
- Size: configurable, default ~last 5-10 turns

**Working state**
- The reservoir's current dense activation
- Compressed representations of recent-but-not-immediate history
- Lossy but captures patterns and recency
- Updated every turn

**Cold store (structured)**
- Extracted entities, facts, decisions, relationships
- Key-value pairs and graph edges
- Sparse, precise, queryable
- Retrieved on demand by the readout layer when relevant to current query
- Persists across sessions (for multi-session mode)

### 5. Decay and Eviction

**Primary: Usage-based**
- State entries that are accessed (retrieved by readout and included in prompts) get their relevance score boosted
- Entries that are never retrieved decay over time
- "Use it or lose it" — mirrors biological memory consolidation

**Secondary: Relevance-based**
- When eviction is needed, lowest-relevance entries are compressed or dropped first
- Relevance is a function of: recency, usage frequency, semantic similarity to recent topics, graph connectivity (highly-connected entities are more important)

**Eviction cascade:**
1. Immediate buffer overflows → oldest turns compressed into working state
2. Working state exceeds capacity → oldest/least-relevant compressed into cold store
3. Cold store exceeds capacity → lowest-relevance entries pruned (logged, not silently dropped)

### 6. Session Persistence

**Per-user session files**
- PANIC state serialized to a user-specific session file
- File contains: serialized reservoir state, structured graph, memory tier contents, metadata (timestamps, access counts)
- Updated at configurable intervals (per-turn, per-N-turns, or on session close)

**Open questions (to be resolved during implementation):**
- File format: binary (compact, fast) vs JSON/MessagePack (inspectable, portable)
- Growth management: session files will grow over time. Need a compaction strategy.
- Archival: old sessions — keep, summarize, or prune?
- Multi-device sync: if user accesses from multiple places, how to handle state conflicts

## Operating Modes

### Mode 1: Long Conversation (default)
- Optimized for extended single-session dialogue
- Immediate buffer is larger
- Working state prioritizes recency and conversational flow
- Decay is slower
- Session file optional (conversation can be ephemeral)

### Mode 2: Multi-Session Memory
- Optimized for persistent knowledge across separate conversations
- Cold store is primary
- Session file is mandatory and loaded on startup
- Working state initialized from cold store based on conversation opening
- Decay is much slower — information persists across sessions
- Graph component is more important here (facts and entities need to survive)

### Mode 3: Large Document Analysis
- Optimized for ingesting and querying large documents
- Immediate buffer is smaller (document content lives in cold store, not buffer)
- Document is chunked, encoded, and stored in cold store with structural metadata (section, page, position)
- Readout layer biased toward document retrieval over conversation history
- Working state tracks the user's "reading position" and current focus area

## Translation Layer

Sits between PANIC state and the LLM. Two directions: outbound (prompt construction) and inbound (response processing).

### Outbound: PANIC → Prompt Construction

**Token budget allocation**

The LLM has a finite context window. PANIC spends it by priority:

```
Total context window (e.g., 200k tokens)
├── System prompt (fixed, ~500-2000 tokens)
├── PANIC context section (variable, bulk of budget)
│   ├── Cold store retrievals (graph-boosted vector hits)
│   ├── Working state summaries
│   └── Immediate buffer (verbatim recent turns)
├── Current user query (variable, usually small)
└── Response headroom (reserved for LLM output, ~4k-8k tokens)
```

Budget allocation rules:
1. System prompt + current query + response headroom subtracted first. Non-negotiable.
2. Immediate buffer gets next priority. Recent turns verbatim until buffer exhausted or configurable max (~30% of remaining budget).
3. Cold store retrievals fill remaining space. Ranked by readout relevance scores. Inserted in order until budget exhausted.
4. Working state summaries fill gaps. Medium-relevance items that didn't make the cut as full retrievals get compressed summaries slotted in.

Overflow trimming: bottom-up by relevance score. Lowest-ranked cold store retrieval drops first, then working state summaries, then (last resort) oldest immediate buffer turns.

**Ordering within the prompt**

LLMs attend more to beginning and end of context, less to the middle ("lost in the middle" problem). PANIC exploits this:

```
[System prompt]
[Highest-relevance cold store retrievals]     ← beginning: high attention
[Working state summaries]                      ← middle: lower attention, OK for summaries
[Lower-relevance cold store retrievals]        ← middle
[Immediate buffer — verbatim recent turns]     ← end: high attention, recency matters
[Current user query]                           ← very end: highest attention
```

Most important retrieved context at top (strong attention). Most recent conversation at bottom (strong attention + recency). Summaries and lower-priority retrievals in the middle (weakest attention — acceptable since they're supplementary).

**Structural cues**

The LLM has no awareness of PANIC. Lightweight markers help it understand what it's reading:

```
[Historical context — relevant prior discussion]
<retrieved content here>

[Recent conversation]
<immediate buffer here>

[Current message]
<user query>
```

Minimal markup. No PANIC metadata leaks into the visible prompt.

**Token counting**

- MVP: tiktoken cl100k_base as approximation across models (slightly overestimates for some — safer than underestimating)
- Future: pluggable tokenizer per connected LLM

### Inbound: LLM Response → PANIC State Update

Processing pipeline:

1. **Encode response** → same encoder as user input → feeds into reservoir state update. Reservoir now reflects both sides of the conversation.
2. **Extract graph updates** → same rule-based extractors run on response. New entities, LLM-asserted facts (tagged differently from user-stated), decisions, references to earlier context.
3. **Confusion detection** → heuristic checks:
   - "I don't have information about..." → context retrieval may have missed something
   - Contradicts a known graph fact → potential retrieval gap or hallucination
   - Asks for clarification about something in cold store → readout should boost that topic next turn
   - These signals feed back as relevance adjustments for the next turn
4. **Store response in immediate buffer** alongside the user turn.

### Latency Budget

| Step | Target | Notes |
|---|---|---|
| Encode user input | <50ms | Off-the-shelf embedding inference |
| Reservoir state update | <1ms | Matrix multiply, D=4096 |
| Graph extraction | <30ms | NER + rule matching |
| Readout (vector search + graph rerank) | <100ms | Depends on cold store size |
| Prompt assembly + token counting | <20ms | String concatenation + tokenizer |
| **Total PANIC overhead (outbound)** | **<200ms** | Before LLM API call |
| Encode LLM response | <50ms | Same encoder |
| Response graph extraction | <30ms | Same extractors |
| State persistence (if triggered) | <50ms | Async, non-blocking |
| **Total PANIC overhead (inbound)** | **<130ms** | After LLM response |
| **Total added latency per turn** | **<350ms** | Well within Tier 2 target of <500ms |

LLM API calls are typically 1-10 seconds. PANIC adds <350ms. Negligible in practice.

## Graph Extractors

Rule-based pipeline feeding the 20% structured component. Runs on every input (user turns and LLM responses).

### Extractor Pipeline

```
Raw text
  │
  ├─→ [1. Entity Extractor]     → graph nodes
  ├─→ [2. Fact Extractor]       → graph nodes + edges
  ├─→ [3. Relation Extractor]   → graph edges
  ├─→ [4. Coreference Resolver] → graph edge merges
  ├─→ [5. Contradiction Detector] → graph node annotations
  └─→ [6. Decision Extractor]   → graph nodes + edges
```

### 1. Entity Extractor (NER)

Identifies named entities and typed concepts.

MVP: spaCy pre-trained NER model. Fast, accurate for common types, no training.

Entity types:
- **People** — names, roles, "my boss", "the client"
- **Organizations** — companies, teams, projects
- **Technical terms** — libraries, frameworks, APIs, model names, protocols
- **Locations** — if relevant
- **Temporal** — dates, deadlines, "last week", "next sprint"
- **Custom** — domain-specific recurring terms (detected by frequency, not NER)

Each entity becomes a graph node:
- `id`: stable hash of canonical name
- `type`: entity category
- `first_seen`: turn number
- `last_seen`: turn number
- `mention_count`: how often referenced
- `vector_refs`: list of reservoir vector indices this entity was extracted from
- `properties`: key-value pairs accumulated over time

### 2. Fact Extractor

Identifies declarative statements that should persist.

MVP: Pattern matching + lightweight classifier. Targets:
- Statements with "is", "are", "was", "has", "should", "must", "will"
- Definitions: "X means Y", "X is defined as Y"
- Preferences: "I prefer X", "use Y instead of Z"
- Constraints: "don't do X", "always Y", "never Z"

Each fact becomes a graph node:
- `id`: content hash
- `statement`: extracted text
- `source`: "user" or "llm"
- `turn`: when stated
- `confidence`: how explicit (direct = high, implied = low)
- `status`: active / superseded / contradicted
- `vector_refs`: linked reservoir vectors

### 3. Relation Extractor

Identifies connections between entities and facts.

Edge types:
- **owns/belongs_to**: "my project", "Ben's API key"
- **causes/leads_to**: "X broke because of Y"
- **depends_on**: "X requires Y"
- **related_to**: weak association, co-occurrence in same turn
- **decided_on**: links a decision to entities involved

MVP: Co-occurrence within same turn creates `related_to` edge. Explicit relational verbs create typed edges via pattern matching.

Edge schema:
- `source_id`, `target_id`: linked node ids
- `type`: edge category
- `weight`: starts at 1.0, boosted by repeated mention
- `turn`: when established

### 4. Coreference Resolver

Determines when different mentions refer to the same entity.

MVP: spaCy or lightweight coreference model. Key cases:
- Pronoun resolution: "it" → the database, "they" → the team
- Alias resolution: "the project" → PANIC, "the spec" → SPEC.md
- Temporal reference: "what we discussed earlier" → links to specific turns

When resolved, graph merges mentions into same entity node. Prevents entity fragmentation.

### 5. Contradiction Detector

Identifies when a new statement conflicts with an existing graph fact.

MVP: For each new fact, check against existing facts involving same entities:
- Direct negation: "X is Y" vs. "X is not Y"
- Value change: "deadline is Friday" vs. "deadline is Monday"
- Preference reversal: "use React" vs. "actually, use Vue"

When detected:
- Old fact status → `superseded`
- New fact links to old via `supersedes` edge
- Old fact's vector refs get penalty flag (readout demotes them)
- Both facts kept — history of the change matters

### 6. Decision Extractor

Identifies commitments, choices, and action items.

Patterns:
- "Let's go with X"
- "We decided to Y"
- "The plan is Z"
- "I'll do X", "you should do Y"

Decision nodes get permanent relevance boost in decay function. Decisions rarely become irrelevant until explicitly reversed.

### Graph Storage

**MVP: SQLite with JSON columns**

- Nodes table: id, type, data (JSON), first_seen, last_seen, mention_count, status
- Edges table: source_id, target_id, type, weight, turn
- Vector refs table: node_id, vector_index, turn

SQLite handles our scale (thousands of nodes, not millions). Recursive CTEs handle multi-hop graph traversal. Neo4j unnecessary unless graph complexity warrants it. Interface is abstracted so backend is swappable.

## Readout Training

The readout is the only component trained end-to-end in MVP. Decides what to retrieve from PANIC state for a given query. Everything else is fixed or rule-based.

### What the Readout Does

Input:
- Current reservoir state S(t) = [S_fast | S_medium | S_slow] (4096-dim)
- Current query embedding q (384-dim from encoder)
- Candidate context items with metadata (from all memory tiers)

Output:
- Relevance score per candidate (0 to 1)
- Timescale weights (how much to trust fast/medium/slow for this query)

### Architecture

Small, fast neural network. Runs every turn, must be <100ms.

```
Input: concat(S(t), q) → 4480-dim
  │
  ├─→ Linear(4480, 512) + ReLU
  ├─→ Linear(512, 256) + ReLU
  ├─→ Linear(256, 128) + ReLU
  │
  ├─→ Scoring head: Linear(128, 1) per candidate → relevance score
  └─→ Timescale head: Linear(128, 3) + softmax → [w_fast, w_medium, w_slow]
```

Timescale weights modulate which sub-reservoir dimensions the scoring head attends to. "What did I just say" → weight S_fast. "What was the original requirement" → weight S_slow.

### Training Data

Synthetic generation using the evaluation harness:

1. Run Synthetic Conversation Generator
2. For each planted fact at turn T, generate probe queries that should retrieve it
3. Ground-truth label: for this query, these context items should score highest
4. Generate negative examples: topically similar but wrong context items

| Source | Volume | Purpose |
|---|---|---|
| Synthetic single-hop probes | 10,000+ | Basic retrieval |
| Synthetic multi-hop probes | 5,000+ | Graph-assisted retrieval |
| Synthetic temporal probes | 3,000+ | Timescale weighting |
| Synthetic negation/contradiction probes | 2,000+ | Contradiction awareness |
| Real conversation data (if available) | As much as possible | Distribution matching |

### Loss Function

- Binary cross-entropy on relevance scores
- KL divergence on timescale weights against oracle timescale labels (which sub-reservoir should dominate per query type)

### Training Regime

- Train on synthetic data first
- Fine-tune on real data when available
- Retrain periodically as graph extractors improve (better graph = better training labels)
- Regression gate: every retrain must pass Standard evaluation config

### What We Don't Train (MVP)

- W_res (reservoir weights) — fixed
- W_in (input projection) — fixed
- Encoder — off-the-shelf, frozen
- Graph extractors — rule-based
- Compressor — separate training loop (TBD)

The readout is intentionally the only trained component in MVP. If results are bad, it's either the readout or the graph extractors. Not a tangled mess of interacting learned components.

## User Interface

### Layout

```
┌────────────────────────────────────────────────────────────────┐
│  PANIC                                          [Settings]  │
├────────────────┬───────────────────────────────────────────────┤
│                │                                               │
│  LLM Config    │              Chat Area                        │
│  ──────────   │                                               │
│  Provider: [v] │   User: How does the reservoir work?          │
│  Model:  [v]   │                                               │
│  API Key: [..] │   PANIC: The reservoir uses fixed random...    │
│  [Connect]     │                                               │
│                │                                               │
│  Mode          │                                               │
│  ──────────   │                                               │
│  (·) Long Conv  │                                               │
│  ( ) Multi-Sess │                                               │
│  ( ) Document   │                                               │
│                │                                               │
│  Session       │                                               │
│  ──────────   │                                               │
│  [Save]        │                                               │
│  [Load]        │                                               │
│  [Clear]       │                                               │
│  [Export]      │                                               │
│                │───────────────────────────────────────────────┤
│  Status        │  [Type your message...]           [Send]     │
│  ──────────   ├───────────────────────────────────────────────┤
│  Latency: 45ms │  [Show PANIC Context ▼]  (expandable panel)    │
│  Turns: 42     │  ┌───────────────────────────────────────────┐ │
│  Buffer: 5/10  │  │ Retrieved: 3 items from cold store        │ │
│  Cold: 128 ent │  │ Buffer: turns 38-42 (verbatim)            │ │
│  Graph: 89 nod │  │ Tokens used: 4,200 / 200,000             │ │
│                │  └───────────────────────────────────────────┘ │
└────────────────┴───────────────────────────────────────────────┘
```

### Panels

**Left sidebar:**
- LLM Configuration — provider dropdown, model selector, API key field, [Connect] button with connection status indicator
- Mode selector — radio buttons: Long Conversation (default), Multi-Session Memory, Document Analysis. Each with one-line description of what it does.
- Session management — [Save Session], [Load Session], [Clear Session], [Export Chat]. All labeled text buttons.
- Status panel — live stats: PANIC overhead latency, turn count, immediate buffer usage, cold store entity count, graph node count

**Main area:**
- Chat interface — scrollable message history with clear user/response visual separation
- Input bar — text input + [Send] button. Enter key sends.
- Transparency panel — expandable (collapsed by default). Shows what PANIC fed to the LLM on the last turn: retrieved items, buffer contents, token usage breakdown. For debugging and trust-building.

**Settings (top-right):**
- Immediate buffer size
- Reservoir parameters (advanced, collapsed by default)
- Token budget allocation overrides
- Export/import PANIC state

### Document Analysis Mode (additional UI)

When Document Analysis mode is selected:
- [Upload Document] button appears in sidebar
- Supported formats: .txt, .md, .pdf (text extraction)
- Upload progress indicator
- Document chunk list (scrollable, shows how PANIC segmented the document)
- Current focus indicator (which chunk PANIC thinks is relevant)

### Design Principles

- Every interactive element has a visible text label
- No icon-only buttons
- Clear state indicators (connected/disconnected, processing/idle, mode active)
- Functional-first layout — aesthetic polish is a separate phase
- Responsive enough for desktop use; mobile optimization deferred

## Evaluation Targets

### Baseline: LLM without PANIC

Measured behavior of current LLMs with long context:

| Scenario | Typical Performance |
|----------|--------------------|
| Fact retrieval at 10% context window | 95-99% accuracy |
| Fact retrieval at 50% context window | 85-95% accuracy |
| Fact retrieval at 80%+ context window | 60-85% accuracy ("lost in the middle" degradation) |
| Multi-hop reasoning across 30k+ token gap | <50% accuracy |
| Contradiction rate after 100 turns | 15-25% of responses |
| Summarization-based context management | Preserves 40-60% of facts; compounds on re-summarization |
| Naive truncation | Everything before cutoff is gone entirely |

### Tier 1 — MVP (proves the concept)

| Metric | Without PANIC | PANIC Target |
|--------|--------------|-------------|
| Fact retrieval at 200 turns | 30-40% (truncated) / 60% (summarized) | 85%+ |
| Multi-hop reasoning across 100+ turn gap | 25-35% | 65%+ |
| Contradiction rate after 100 turns | 15-25% | <5% |
| Relevant context inclusion rate | N/A | 80%+ |

### Tier 2 — Solid Product

| Metric | PANIC Target |
|--------|--------------|
| Fact retrieval at 500 turns | 80%+ |
| Multi-hop reasoning across 250+ turn gap | 60%+ |
| Latency overhead per turn | <500ms |
| Session file size after 500 turns | <50MB |
| Context compression ratio (effective context / tokens sent) | 10:1 or better |

### Tier 3 — Ambitious

| Metric | PANIC Target |
|--------|--------------|
| Fact retrieval at 1000+ turns | 75%+ |
| User-blind test: can user tell PANIC is managing context? | >80% "no difference" |
| Cross-session knowledge retention (Mode 2, across 10+ sessions) | 70%+ of key facts |

### Key commercial metric

The **10:1 compression ratio** is the number that matters most. If PANIC gives the user the experience of 2M tokens of context while sending 200k to the LLM, that's the value proposition.

## Evaluation Harness

Built before PANIC itself. Runs identical tests with and without PANIC for direct comparison.

### Test Suite Components

**1. Synthetic Conversation Generator**
- Generates multi-hundred-turn conversations with realistic topic drift
- Plants specific retrievable facts at known turn positions ("needles")
- Plants entity introductions, decisions, contradictions, and corrections at known positions
- Tags each planted item with: turn number, category (fact/entity/decision/contradiction), difficulty (explicit statement vs. implied)
- Configurable: conversation length, topic count, needle density, ambiguity level

**2. Retrieval Probe Set**
- Single-hop: "What did I say about X in the earlier discussion?" (fact is at a known turn)
- Multi-hop: "Given what I said about X and what we decided about Y, what should we do about Z?"
- Temporal: "Did I change my mind about X at any point?"
- Negation: "What did I say we should NOT do?"
- Implicit: "What was the concern I raised but never explicitly named?"
- Each probe has a ground-truth answer derived from the planted facts

**3. Scoring Functions**

```
fact_retrieval_accuracy = correct_retrievals / total_probes
multi_hop_accuracy = correct_multi_hop / total_multi_hop_probes
contradiction_rate = contradicted_responses / total_responses
context_inclusion = probes_where_relevant_context_was_in_prompt / total_probes
compression_ratio = total_conversation_tokens / tokens_sent_to_llm
latency_overhead = (panic_turn_time - baseline_turn_time) / baseline_turn_time
```

**4. Test Configurations**

| Config | Turns | Needles | Multi-hop pairs | Purpose |
|--------|-------|---------|----------------|---------|
| Smoke | 50 | 10 | 5 | Fast sanity check |
| Standard | 200 | 40 | 20 | MVP validation |
| Stress | 500 | 100 | 50 | Product-grade validation |
| Marathon | 1000 | 200 | 100 | Tier 3 ambition test |

**5. Comparison Modes**

Every test runs in these configurations:
- **Baseline A — Full context:** All turns sent to LLM (only possible within context window)
- **Baseline B — Naive truncation:** Only last N turns sent, rest dropped
- **Baseline C — Summarization:** Older turns summarized, summary + recent turns sent
- **PANIC:** PANIC layer manages context selection

Results are reported as a comparison table across all four modes.

**6. Regression Gate**

Any code change to PANIC must pass the Standard test config without regressing any Tier 1 metric by more than 2 percentage points. Enforced in CI.

## 80/20 Interface Design

The vector space (80%) and structured graph (20%) are not parallel retrieval paths. The graph annotates and constrains the vector space. The graph never returns results on its own — it shapes which vector results surface and in what order.

### On Input (State Update)

1. Text hits the encoder → produces dense vector x(t) → feeds into reservoir (vector space)
2. Simultaneously, rule-based extractors pull entities/facts/relations → feeds into graph
3. Each graph node is linked to the reservoir vector(s) it was extracted from (pointer, not copy)
4. Graph edges encode relationships between vectors that pure cosine similarity would miss

### On Retrieval (Readout)

1. Query encodes to a vector
2. Vector similarity search returns top-K candidates from the reservoir
3. Graph **reranks** those candidates:
   - **Boost:** Candidates connected to entities mentioned in the query get relevance boost
   - **Boost:** Candidates connected to currently-active graph clusters (recent topics) get boosted
   - **Penalize:** Candidates whose graph nodes are marked contradicted/superseded get demoted
   - **Expand:** If a graph edge connects a top candidate to another vector not in top-K, pull it in (this is how multi-hop reasoning works — the graph bridges gaps vector similarity alone can't)
4. Final ranked list goes to the translation layer

### On Compression (Eviction)

1. Graph connectivity determines survival. Isolated vectors with no graph edges decay fastest.
2. When vectors are compressed/merged, their graph edges are preserved and repointed to the merged representation.
3. Graph nodes without any vector backing are flagged "orphaned" — kept in cold store as bare facts but can't be expanded into full context.

### Failure Modes

- If the graph extractor misses something: vectors still work (degraded, not broken)
- If vectors drift: graph anchors them to concrete facts
- If both miss: the immediate buffer (verbatim recent turns) is the safety net

### Key Principle

Vectors do the heavy lifting. The graph is metadata that makes vector retrieval smarter. The graph is cheap to maintain relative to the reservoir. Multi-hop reasoning comes from graph edge traversal, not from hoping vector similarity catches it.

## Tech Stack

**Language: Python**

**Compute: CPU-only for MVP.** Switch to GPU only if a measurable speed bottleneck emerges.

### Core Dependencies

| Component | Library | Purpose |
|---|---|---|
| Reservoir math | NumPy / SciPy | Matrix ops, sparse matrices. CPU-only, sufficient for D=4096 |
| Encoder | sentence-transformers (all-MiniLM-L6-v2) | Off-the-shelf embeddings, 384-dim, CPU inference |
| Graph extractors | spaCy (en_core_web_trf) | NER, coreference, dependency parsing |
| Graph storage | SQLite (stdlib) | Zero-config, embedded, sufficient scale |
| Token counting | tiktoken | cl100k_base as default approximation across models |
| Readout model | PyTorch (CPU) | Small MLP, easy to train and serialize |
| LLM API calls | litellm | Multi-provider compatibility (OpenAI, Anthropic, local models) |
| Backend | FastAPI | REST API serving the PANIC pipeline |
| Frontend | React or plain HTML/JS | Web UI with buttons, clear descriptions, functional-first |
| Session persistence | MessagePack | Binary serialization for reservoir state snapshots |
| Evaluation harness | pytest + custom generators | Synthetic conversation generation, scoring, regression gates |

### UI Approach

Web UI from the start. Functional-first, aesthetic polish later.

Requirements:
- Clear labeled buttons for all actions (send, clear, mode switch, session management)
- Visible mode selector (Long Conversation / Multi-Session / Document Analysis)
- LLM connection panel (API key input, model selector, provider dropdown)
- Chat interface with clear input/output separation
- Status indicators (PANIC state health, current memory tier usage, latency per turn)
- Optional transparency view (expandable panel showing what PANIC fed to the LLM)
- Session management (save, load, clear, export)
- All interactive elements have clear text labels — no icon-only buttons

Aesthetic design is a separate phase after core functionality works.

## Open Questions

1. Reservoir dimensionality — how large does the vector space need to be?
2. Training data for readout and compressor — what do we train on?
3. Latency budget per turn — what's acceptable overhead?
4. Embedding model for encoder — use existing (e.g., sentence-transformers) or train custom?
5. Graph database for cold store — embedded (SQLite + JSON) or dedicated (Neo4j)?

---

*Last updated: 2026-04-12*
*Status: Specification phase — no code yet*
