# PANIC — Persistent Memory for AI Assistants

A memory engine that gives your AI assistant persistent, structured, **local**  memory across conversations.

PANIC is an [OpenClaw](https://github.com/openclaw/openclaw) context engine plugin. It replaces the default context assembly pipeline with a 4-layer memory system that remembers facts, preferences, workflows, and conversation history across sessions.

## How It Works

PANIC sits between your conversations and the LLM. Every message passes through PANIC, which:

1. **Extracts** entities, facts, and relationships using both rule-based and LLM-assisted graph extraction
2. **Stores** knowledge in 4 memory layers (working, semantic, episodic, procedural) as plain markdown files
3. **Retrieves** relevant context for each new message using embedding similarity + graph boosting
4. **Injects** the retrieved memory into the LLM's system prompt so it "remembers" previous conversations

```
User message → PANIC ingest → graph extraction → knowledge stored
                                                         ↓
LLM prompt  ← PANIC assemble ← retrieval + ranking ← memory layers
```

### Memory Layers

| Layer | What It Stores | Persists Across Sessions |
|-------|---------------|-------------------------|
| **Working** | Current session entities, facts, decisions | No (session-scoped) |
| **Semantic** | Entities, facts, preferences | Yes |
| **Episodic** | Session summaries with key moments | Yes |
| **Procedural** | Workflows, tools, failure patterns | Yes |

All persistent memory is stored as markdown files in `~/.openclaw/panic/profiles/<name>/` — human-readable, editable, and portable.

## Architecture

```
┌─────────────────────────────────────────────┐
│  OpenClaw Gateway                           │
│  ┌─────────────────────────────────────┐    │
│  │  PANIC Plugin (TypeScript)          │    │
│  │  • ContextEngine interface          │    │
│  │  • ingest / assemble / afterTurn    │    │
│  │  • Sidecar process manager          │    │
│  └──────────────┬──────────────────────┘    │
│                 │ HTTP (localhost)           │
│  ┌──────────────▼──────────────────────┐    │
│  │  PANIC Sidecar (Python/FastAPI)     │    │
│  │  • Dual graph engine (rule + LLM)   │    │
│  │  • all-MiniLM-L6-v2 encoder (384d) │    │
│  │  • Cosine + graph-boost retrieval   │    │
│  │  • Session-end LLM extraction       │    │
│  │  • Profile management               │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
         │
         ▼
~/.openclaw/panic/profiles/default/
├── config.json
├── embeddings.npz
├── episodes/2024-01-15.md
├── graphs/rule.db
├── graphs/llm.db
├── semantic/
│   ├── entities.md
│   ├── facts.md
│   └── preferences.md
├── procedural/
│   ├── workflows.md
│   └── failures.md
└── session_state.json
```

## Requirements

- **OpenClaw** 2026.3.24-beta.2 or later
- **Python** 3.11+
- **Node.js** 18+
- **~2 GB disk** for Python dependencies + sentence-transformers model
- An **Anthropic API key** (for LLM-assisted extraction; set as `ANTHROPIC_API_KEY` env var)

## Install

```bash
git clone https://github.com/bensabanas/panic.git
cd panic
bash install.sh
```

The installer will:
1. Create a Python virtual environment and install dependencies
2. Download the sentence-transformers encoder model
3. Build and install the OpenClaw plugin
4. Configure OpenClaw and restart the gateway

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PANIC_PORT` | `7420` | Sidecar API port |
| `PANIC_PROFILE` | `default` | Initial profile name |
| `ANTHROPIC_API_KEY` | — | Required for LLM extraction |

## Dashboard

PANIC includes a web dashboard for monitoring and management:

```
http://127.0.0.1:7420
```

Features:
- **Dashboard** — live metrics, memory layer visualization, graph stats
- **Context Preview** — test retrieval queries and see what PANIC injects
- **Profiles** — switch, create, or clone memory profiles
- **Memory Editor** — edit entities, facts, preferences, workflows directly
- **Import** — paste or drag-and-drop files for LLM-powered memory extraction
- **Advanced Settings** — tune token budgets, retrieval weights, extraction parameters

## Profiles

Profiles are isolated memory containers. Each profile has its own entities, facts, episodes, and settings.

```bash
# Via the dashboard at http://127.0.0.1:7420 → Profiles tab
# Or via the API:
curl -X POST http://127.0.0.1:7420/api/profiles -d '{"name":"work"}'
curl -X POST http://127.0.0.1:7420/api/profiles/switch -d '{"name":"work"}'
```

Profile data lives in `~/.openclaw/panic/profiles/<name>/` as plain markdown — you can edit files directly, back them up, or sync them across machines.

## API

The sidecar exposes a REST API on `http://127.0.0.1:7420`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Engine status, graph stats, connection info |
| `/api/chat/ingest` | POST | Ingest a message for graph extraction |
| `/api/assemble` | POST | Build layered memory context for a query |
| `/api/session/end` | POST | Run session-end extraction (LLM call) |
| `/api/session/clear` | POST | Clear current session state |
| `/api/profiles` | GET | List profiles |
| `/api/profiles` | POST | Create a profile |
| `/api/profiles/switch` | POST | Switch active profile |
| `/api/profiles/{name}` | GET | Profile config and stats |
| `/api/profiles/{name}` | PATCH | Update profile settings |
| `/api/profiles/{name}/semantic/{file}` | GET/PUT | Read/write semantic memory files |
| `/api/profiles/{name}/procedural/{file}` | GET/PUT | Read/write procedural memory files |
| `/api/profiles/{name}/episodes` | GET | List episodes |
| `/api/profiles/{name}/import` | POST | Import text via LLM extraction |

## How Retrieval Works

1. **Encode** the user's message with all-MiniLM-L6-v2 (384-dimensional embedding)
2. **Build candidates** from both the rule-based and LLM-assisted knowledge graphs
3. **Score** each candidate: `score = 0.75 × cosine_similarity + 0.2 × graph_boost + recency_bonus - contradiction_penalty`
4. **Blend** item-level and turn-level candidates at a 70/30 ratio
5. **Retrieve** cross-session memory from semantic, episodic, and procedural markdown files
6. **Assemble** into labeled sections with per-layer token budgets
7. **Inject** as a `systemPromptAddition` that the LLM sees at the top of every turn

## Uninstall

```bash
cd panic
bash uninstall.sh                  # interactive — asks before deleting profiles
bash uninstall.sh --keep-profiles  # remove plugin, preserve memory data
bash uninstall.sh --purge          # remove everything including Python venv
```

## Performance

Tested at 50, 200, and 500 conversation turns:

| Turns | Fact Recall | Multi-hop | Contradiction Detection | Retrieval Latency |
|-------|-------------|-----------|------------------------|-------------------|
| 50 | 100% | 100% | 100% | 8ms |
| 200 | 100% | 60-80% | 80-100% | 9ms |
| 500 | 97% | 74% | 67% | 11ms |

Multi-hop variance is due to LLM graph extraction non-determinism — the retrieval engine itself is deterministic.
