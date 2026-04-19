"""
PANIC Profile Manager — Isolated Memory Silos

Each profile is a self-contained directory with its own:
  - Entity graphs (rule + LLM) as SQLite files
  - Turn embeddings index (SQLite)
  - Episodic session logs (markdown)
  - Semantic memory (entities.md, facts.md, preferences.md)
  - Procedural memory (workflows.md, failures.md)
  - Profile-specific config

Profiles live under a root directory (default: ~/.openclaw/panic/profiles/).
The ProfileManager handles creation, switching, deletion, cloning, and listing.

Usage:
    pm = ProfileManager()  # uses ~/.openclaw/panic/
    pm.create("work")
    pm.create("personal")
    pm.switch("work", engine)   # loads work profile into engine
    pm.switch("personal", engine)
    pm.list_profiles()          # ["default", "personal", "work"]
"""

import json
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from panic.graph.storage import GraphStorage


# --- Constants ---

DEFAULT_ROOT = Path.home() / ".openclaw" / "panic"
DEFAULT_PROFILE = "default"

# Subdirectories created for each new profile
PROFILE_DIRS = [
    "graphs",       # SQLite graph databases (rule.db, llm.db)
    "episodes",     # Per-session episode summaries (YYYY-MM-DD.md)
    "semantic",     # Accumulated knowledge (entities.md, facts.md, preferences.md)
    "procedural",   # Learned patterns (workflows.md, failures.md)
]

# Template files created in semantic/ for each new profile.
# These are append-only during normal operation — extraction pipeline adds to them.
SEMANTIC_FILES = {
    "entities.md": "# Entities\n\nAccumulated entity knowledge across sessions.\n",
    "facts.md": "# Facts\n\nEstablished knowledge, soft-deduplicated.\n",
    "preferences.md": "# Preferences\n\nUser patterns and preferences.\n",
}

# Template files created in procedural/ for each new profile.
PROCEDURAL_FILES = {
    "workflows.md": "# Workflows\n\nHow things get done — tools, approaches, solutions.\n",
    "failures.md": "# Failures\n\nWhat didn't work and why.\n",
}


# --- Data classes ---

@dataclass
class ProfileConfig:
    """Per-profile configuration."""
    name: str
    created_at: float = 0.0
    updated_at: float = 0.0
    turn_count: int = 0

    # LLM extraction
    extraction_model: str = "claude-haiku-4-5-20251001"

    # Retrieval scoring weights
    item_blend: float = 0.7
    w_cosine: float = 0.75
    w_graph: float = 0.20

    # Token budgets for layered context injection
    budget_semantic: int = 500
    budget_episodic: int = 1000
    budget_working: int = 2000
    budget_procedural: int = 300

    # Buffer
    buffer_size: int = 10
    llm_flush_interval: int = 5

    def save(self, path: Path):
        """Write config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProfileConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProfileInfo:
    """Summary info about a profile (for listing)."""
    name: str
    created_at: float
    updated_at: float
    turn_count: int
    episode_count: int
    entity_count: int  # approximate, from semantic/entities.md


# --- Profile Manager ---

class ProfileManager:
    """
    Manages PANIC memory profiles.

    Each profile is a directory under root/profiles/<name>/ containing
    graphs, embeddings, episodes, semantic memory, and procedural memory.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or DEFAULT_ROOT
        self.profiles_dir = self.root / "profiles"
        self._active_profile: Optional[str] = None
        self._global_config_path = self.root / "config.json"

        # Ensure root structure exists
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        # Create default profile if none exist
        if not any(self.profiles_dir.iterdir()):
            self.create(DEFAULT_PROFILE)

        # Load global config to find active profile
        self._load_global_config()

    def _load_global_config(self):
        """Load global PANIC config (active profile, etc.)."""
        if self._global_config_path.exists():
            with open(self._global_config_path) as f:
                data = json.load(f)
            self._active_profile = data.get("active_profile", DEFAULT_PROFILE)
        else:
            self._active_profile = DEFAULT_PROFILE
            self._save_global_config()

    def _save_global_config(self):
        """Save global PANIC config."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "active_profile": self._active_profile,
            "updated_at": time.time(),
        }
        with open(self._global_config_path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def active_profile(self) -> str:
        return self._active_profile or DEFAULT_PROFILE

    def profile_path(self, name: str) -> Path:
        """Get the directory path for a profile."""
        return self.profiles_dir / name

    def profile_exists(self, name: str) -> bool:
        return self.profile_path(name).is_dir()

    # --- CRUD ---

    def create(self, name: str, config: Optional[ProfileConfig] = None) -> ProfileConfig:
        """
        Create a new profile with empty memory stores.

        Args:
            name: Profile name (alphanumeric + hyphens + underscores).
            config: Optional custom config. Defaults are used if not provided.

        Returns:
            ProfileConfig for the new profile.

        Raises:
            ValueError: If profile already exists or name is invalid.
        """
        name = self._validate_name(name)
        pdir = self.profile_path(name)
        if pdir.exists():
            raise ValueError(f"Profile '{name}' already exists")

        # Create directory structure
        pdir.mkdir(parents=True)
        for subdir in PROFILE_DIRS:
            (pdir / subdir).mkdir()

        # Create semantic memory files
        for filename, template in SEMANTIC_FILES.items():
            (pdir / "semantic" / filename).write_text(template)

        # Create procedural memory files
        for filename, template in PROCEDURAL_FILES.items():
            (pdir / "procedural" / filename).write_text(template)

        # Create profile config
        now = time.time()
        cfg = config or ProfileConfig(name=name, created_at=now, updated_at=now)
        cfg.name = name
        if cfg.created_at == 0.0:
            cfg.created_at = now
        cfg.updated_at = now
        cfg.save(pdir / "config.json")

        return cfg

    def delete(self, name: str) -> bool:
        """
        Delete a profile and all its data.

        Cannot delete the active profile or the last remaining profile.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If trying to delete the active or last profile.
        """
        name = self._validate_name(name)
        pdir = self.profile_path(name)
        if not pdir.exists():
            return False

        if name == self._active_profile:
            raise ValueError(f"Cannot delete the active profile '{name}'. Switch to another first.")

        profiles = self.list_profiles()
        if len(profiles) <= 1:
            raise ValueError("Cannot delete the last remaining profile.")

        shutil.rmtree(pdir)
        return True

    def clone(self, source: str, target: str) -> ProfileConfig:
        """
        Clone a profile (full copy of all data).

        Args:
            source: Existing profile to clone from.
            target: New profile name.

        Returns:
            ProfileConfig of the new profile.

        Raises:
            ValueError: If source doesn't exist or target already exists.
        """
        source = self._validate_name(source)
        target = self._validate_name(target)

        src_dir = self.profile_path(source)
        tgt_dir = self.profile_path(target)

        if not src_dir.exists():
            raise ValueError(f"Source profile '{source}' does not exist")
        if tgt_dir.exists():
            raise ValueError(f"Target profile '{target}' already exists")

        # Deep copy
        shutil.copytree(src_dir, tgt_dir)

        # Update config in the clone
        cfg = ProfileConfig.load(tgt_dir / "config.json")
        cfg.name = target
        cfg.created_at = time.time()
        cfg.updated_at = cfg.created_at
        cfg.save(tgt_dir / "config.json")

        return cfg

    def list_profiles(self) -> list[ProfileInfo]:
        """List all profiles with summary info."""
        profiles = []
        for pdir in sorted(self.profiles_dir.iterdir()):
            if not pdir.is_dir():
                continue
            cfg_path = pdir / "config.json"
            if cfg_path.exists():
                cfg = ProfileConfig.load(cfg_path)
            else:
                cfg = ProfileConfig(name=pdir.name, created_at=0, updated_at=0)

            # Count episodes
            episodes_dir = pdir / "episodes"
            episode_count = len(list(episodes_dir.glob("*.md"))) if episodes_dir.exists() else 0

            # Count entities (rough: count ## headings in entities.md)
            entities_path = pdir / "semantic" / "entities.md"
            entity_count = 0
            if entities_path.exists():
                content = entities_path.read_text()
                entity_count = content.count("\n## ") + (1 if content.startswith("## ") else 0)

            profiles.append(ProfileInfo(
                name=cfg.name,
                created_at=cfg.created_at,
                updated_at=cfg.updated_at,
                turn_count=cfg.turn_count,
                episode_count=episode_count,
                entity_count=entity_count,
            ))
        return profiles

    def get_config(self, name: Optional[str] = None) -> ProfileConfig:
        """Get config for a profile (default: active profile)."""
        name = name or self.active_profile
        name = self._validate_name(name)
        pdir = self.profile_path(name)
        if not pdir.exists():
            raise ValueError(f"Profile '{name}' does not exist")
        cfg_path = pdir / "config.json"
        if cfg_path.exists():
            return ProfileConfig.load(cfg_path)
        return ProfileConfig(name=name)

    def update_config(self, name: str, **kwargs) -> ProfileConfig:
        """Update specific config fields for a profile."""
        name = self._validate_name(name)
        cfg = self.get_config(name)
        for k, v in kwargs.items():
            if hasattr(cfg, k) and k != "name":
                setattr(cfg, k, v)
        cfg.updated_at = time.time()
        cfg.save(self.profile_path(name) / "config.json")
        return cfg

    # --- Engine integration ---

    def switch(self, name: str, engine) -> ProfileConfig:
        """
        Switch the active profile and load it into the engine.

        This:
          1. Saves the current profile state (if any active)
          2. Loads the new profile's graphs and embeddings
          3. Updates the global active profile setting

        Args:
            name: Profile to switch to.
            engine: PanicEngine instance.

        Returns:
            ProfileConfig of the newly active profile.
        """
        name = self._validate_name(name)
        pdir = self.profile_path(name)
        if not pdir.exists():
            raise ValueError(f"Profile '{name}' does not exist")

        # Save current state if there's an active profile
        if self._active_profile and self._active_profile != name:
            try:
                self.save_state(self._active_profile, engine)
            except Exception:
                pass  # Best-effort save on switch

        # Load the new profile
        cfg = self._load_profile_into_engine(name, engine)

        # Update active
        self._active_profile = name
        self._save_global_config()

        return cfg

    def save_state(self, name: Optional[str], engine):
        """
        Persist the engine's current state to the profile directory.

        Saves: graphs (as SQLite), turn embeddings, chat history, buffer,
        and updates the profile config with current turn count.
        """
        name = name or self.active_profile
        name = self._validate_name(name)
        pdir = self.profile_path(name)
        if not pdir.exists():
            raise ValueError(f"Profile '{name}' does not exist")

        graphs_dir = pdir / "graphs"
        graphs_dir.mkdir(exist_ok=True)

        # Save graphs by exporting current in-memory graphs to on-disk SQLite
        self._save_graph_to_disk(engine.rule_graph, graphs_dir / "rule.db")
        self._save_graph_to_disk(engine.llm_graph, graphs_dir / "llm.db")

        # Save turn embeddings
        self._save_embeddings(engine.turn_embeddings, pdir / "embeddings.npz")

        # Save chat history + buffer as JSON
        state = {
            "turn": engine.turn,
            "chat_history": engine.chat_history,
            "immediate_buffer": engine.immediate_buffer,
            "turns_since_flush": engine._turns_since_flush,
        }
        with open(pdir / "session_state.json", "w") as f:
            json.dump(state, f)

        # Update profile config
        cfg = self.get_config(name)
        cfg.turn_count = engine.turn
        cfg.updated_at = time.time()
        cfg.save(pdir / "config.json")

    def _load_profile_into_engine(self, name: str, engine) -> ProfileConfig:
        """Load a profile's state into the engine."""
        pdir = self.profile_path(name)
        cfg = self.get_config(name)

        # Reset engine state
        engine.turn = 0
        engine.chat_history.clear()
        engine.immediate_buffer.clear()
        engine.turn_embeddings.clear()
        engine._turns_since_flush = 0

        # Apply profile config to engine
        engine.item_blend = cfg.item_blend
        engine.w_cosine = cfg.w_cosine
        engine.w_graph = cfg.w_graph
        engine.buffer_size = cfg.buffer_size
        engine.llm_flush_interval = cfg.llm_flush_interval
        engine.extraction_model = cfg.extraction_model

        # Load graphs
        graphs_dir = pdir / "graphs"
        rule_db = graphs_dir / "rule.db"
        llm_db = graphs_dir / "llm.db"

        # Close existing in-memory graphs
        if engine.rule_graph:
            engine.rule_graph.close()
        if engine.llm_graph:
            engine.llm_graph.close()

        if rule_db.exists():
            engine.rule_graph = self._load_graph_from_disk(rule_db)
        else:
            engine.rule_graph = GraphStorage(":memory:")

        if llm_db.exists():
            engine.llm_graph = self._load_graph_from_disk(llm_db)
        else:
            engine.llm_graph = GraphStorage(":memory:")

        # Reinitialize extractors with new graphs
        from panic.graph.extractors import ExtractorPipeline
        from panic.graph.llm_extractors import LLMExtractorPipeline, LLMExtractorConfig

        engine.rule_extractor = ExtractorPipeline(engine.rule_graph)
        llm_rule_fallback = ExtractorPipeline(engine.llm_graph)
        llm_config = LLMExtractorConfig(
            model=cfg.extraction_model,
            batch_size=20,
            temperature=0.0,
        )
        engine.llm_extractor = LLMExtractorPipeline(
            engine.llm_graph, config=llm_config, rule_fallback=llm_rule_fallback
        )

        # Load embeddings
        emb_path = pdir / "embeddings.npz"
        if emb_path.exists():
            engine.turn_embeddings = self._load_embeddings(emb_path)

        # Load session state
        state_path = pdir / "session_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            engine.turn = state.get("turn", 0)
            engine.chat_history = state.get("chat_history", [])
            engine.immediate_buffer = state.get("immediate_buffer", [])
            engine._turns_since_flush = state.get("turns_since_flush", 0)

        return cfg

    # --- Graph persistence helpers ---

    def _save_graph_to_disk(self, graph: GraphStorage, db_path: Path):
        """
        Export an in-memory graph to an on-disk SQLite file.
        Creates a fresh file each time (atomic replacement via .tmp rename).
        This avoids partial writes — the old file stays intact until the new one is ready.
        """
        from panic.graph.storage import (
            GraphNode, GraphEdge, VectorRef,
            NodeType, NodeStatus, EdgeType,
        )

        tmp_path = db_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        disk_graph = GraphStorage(str(tmp_path))

        # Copy all nodes
        all_nodes = graph.get_all_nodes(active_only=False)
        for node in all_nodes:
            disk_graph.upsert_node(node)
            # Fix mention_count (upsert increments)
            if node.mention_count > 1:
                disk_graph._conn.execute(
                    "UPDATE nodes SET mention_count = ? WHERE id = ?",
                    (node.mention_count, node.id),
                )

        # Copy all edges
        seen_edges = set()
        for node in all_nodes:
            for edge in graph.get_edges_from(node.id):
                edge_key = (edge.source_id, edge.target_id, edge.type.value)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    disk_graph.upsert_edge(edge)
                    # Fix weight
                    disk_graph._conn.execute(
                        "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ? AND type = ?",
                        (edge.weight, edge.source_id, edge.target_id, edge.type.value),
                    )

        # Copy vector refs
        for node in all_nodes:
            for ref in graph.get_vector_refs(node.id):
                disk_graph.add_vector_ref(ref)

        disk_graph._conn.commit()
        disk_graph.close()

        # Atomic replace
        if db_path.exists():
            db_path.unlink()
        tmp_path.rename(db_path)

    def _load_graph_from_disk(self, db_path: Path) -> GraphStorage:
        """
        Load a graph from an on-disk SQLite file into an in-memory GraphStorage.
        We copy into memory (not use the file directly) because the graph engine
        does frequent small writes during conversation — in-memory is much faster.
        """
        from panic.graph.storage import (
            GraphNode, GraphEdge, VectorRef,
            NodeType, NodeStatus, EdgeType,
        )

        disk_graph = GraphStorage(str(db_path))
        mem_graph = GraphStorage(":memory:")

        all_nodes = disk_graph.get_all_nodes(active_only=False)
        for node in all_nodes:
            mem_graph.upsert_node(node)
            if node.mention_count > 1:
                mem_graph._conn.execute(
                    "UPDATE nodes SET mention_count = ? WHERE id = ?",
                    (node.mention_count, node.id),
                )

        seen_edges = set()
        for node in all_nodes:
            for edge in disk_graph.get_edges_from(node.id):
                edge_key = (edge.source_id, edge.target_id, edge.type.value)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    mem_graph.upsert_edge(edge)
                    mem_graph._conn.execute(
                        "UPDATE edges SET weight = ? WHERE source_id = ? AND target_id = ? AND type = ?",
                        (edge.weight, edge.source_id, edge.target_id, edge.type.value),
                    )

        for node in all_nodes:
            for ref in disk_graph.get_vector_refs(node.id):
                mem_graph.add_vector_ref(ref)

        mem_graph._conn.commit()
        disk_graph.close()

        return mem_graph

    # --- Embedding persistence ---

    def _save_embeddings(self, turn_embeddings: dict[int, np.ndarray], path: Path):
        """Save turn embeddings to a compressed numpy archive."""
        if not turn_embeddings:
            if path.exists():
                path.unlink()
            return
        turns = sorted(turn_embeddings.keys())
        turn_array = np.array(turns, dtype=np.int64)
        emb_array = np.stack([turn_embeddings[t] for t in turns])
        np.savez_compressed(str(path), turns=turn_array, embeddings=emb_array)

    def _load_embeddings(self, path: Path) -> dict[int, np.ndarray]:
        """Load turn embeddings from a numpy archive."""
        data = np.load(str(path))
        turns = data["turns"]
        embeddings = data["embeddings"]
        return {int(t): embeddings[i].copy() for i, t in enumerate(turns)}

    # --- Markdown memory accessors ---

    def read_semantic(self, name: Optional[str] = None, file: str = "entities.md") -> str:
        """Read a semantic memory file."""
        name = name or self.active_profile
        path = self.profile_path(name) / "semantic" / file
        if path.exists():
            return path.read_text()
        return ""

    def write_semantic(self, content: str, name: Optional[str] = None, file: str = "entities.md"):
        """Write to a semantic memory file."""
        name = name or self.active_profile
        path = self.profile_path(name) / "semantic" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def append_semantic(self, content: str, name: Optional[str] = None, file: str = "entities.md"):
        """Append to a semantic memory file."""
        name = name or self.active_profile
        path = self.profile_path(name) / "semantic" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text() if path.exists() else ""
        if not existing.endswith("\n"):
            existing += "\n"
        path.write_text(existing + content)

    def read_procedural(self, name: Optional[str] = None, file: str = "workflows.md") -> str:
        """Read a procedural memory file."""
        name = name or self.active_profile
        path = self.profile_path(name) / "procedural" / file
        if path.exists():
            return path.read_text()
        return ""

    def write_procedural(self, content: str, name: Optional[str] = None, file: str = "workflows.md"):
        """Write to a procedural memory file."""
        name = name or self.active_profile
        path = self.profile_path(name) / "procedural" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_episode(self, name: Optional[str] = None, date: Optional[str] = None) -> str:
        """Read an episode file. If no date, reads the latest."""
        name = name or self.active_profile
        episodes_dir = self.profile_path(name) / "episodes"
        if not episodes_dir.exists():
            return ""
        if date:
            path = episodes_dir / f"{date}.md"
            return path.read_text() if path.exists() else ""
        # Latest
        files = sorted(episodes_dir.glob("*.md"), reverse=True)
        return files[0].read_text() if files else ""

    def write_episode(self, content: str, date: str, name: Optional[str] = None):
        """Write an episode file for a given date."""
        name = name or self.active_profile
        path = self.profile_path(name) / "episodes" / f"{date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def list_episodes(self, name: Optional[str] = None) -> list[str]:
        """List all episode dates for a profile."""
        name = name or self.active_profile
        episodes_dir = self.profile_path(name) / "episodes"
        if not episodes_dir.exists():
            return []
        return sorted([f.stem for f in episodes_dir.glob("*.md")])

    # --- Validation ---

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate and normalize a profile name."""
        name = name.strip().lower()
        if not name:
            raise ValueError("Profile name cannot be empty")
        if not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(f"Profile name '{name}' contains invalid characters. Use alphanumeric, hyphens, and underscores only.")
        if len(name) > 64:
            raise ValueError(f"Profile name too long (max 64 chars)")
        return name
