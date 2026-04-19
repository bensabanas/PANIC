/**
 * PANIC Context Engine
 *
 * Implements OpenClaw's ContextEngine interface by delegating to the Python
 * sidecar. This is the bridge between OpenClaw's turn lifecycle and PANIC's
 * retrieval/extraction pipeline.
 *
 * Lifecycle per turn:
 *   1. ingest()    — OpenClaw sends each message (user + assistant)
 *                     → forwarded to sidecar as graph extraction input
 *   2. assemble()  — OpenClaw asks for context to send to the LLM
 *                     → PANIC queries all 4 memory layers, builds layered prompt,
 *                       returns messages with injected context
 *   3. afterTurn() — called after the LLM responds
 *                     → triggers intermediate extraction every 50 turns
 *   4. compact()   — called when context window is full
 *                     → delegates to OpenClaw's built-in compaction (PANIC doesn't
 *                       own compaction — it manages memory separately)
 */

import type {
  ContextEngine,
  ContextEngineInfo,
  IngestResult,
  AssembleResult,
  CompactResult,
} from "openclaw/plugin-sdk";
import type { PanicSidecar } from "./sidecar.js";

/** Configuration for creating the PANIC context engine */
interface PanicEngineConfig {
  sidecar: PanicSidecar;
  profile: string;
  logger: {
    info: (msg: string) => void;
    warn: (msg: string) => void;
    error: (msg: string) => void;
  };
}

/**
 * Create a PANIC context engine instance.
 *
 * This is called by the factory registered in index.ts. OpenClaw calls it
 * once when resolving the active context engine for a session.
 */
export function createPanicContextEngine(
  config: PanicEngineConfig
): ContextEngine {
  const { sidecar, logger } = config;

  // Track turn count for intermediate extraction timing
  let turnCount = 0;

  const info: ContextEngineInfo = {
    id: "panic",
    name: "PANIC — Persistent Memory Engine",
    version: "0.1.0",
    // false = we delegate compaction to OpenClaw's built-in compaction.
    // PANIC manages its own persistent memory (profiles, episodes, etc.)
    // but doesn't replace the session transcript compaction logic.
    ownsCompaction: false,
  };

  return {
    info,

    /**
     * Ingest a single message into PANIC's graph engine.
     *
     * Called for every message in the session (user, assistant, system, tool).
     * We only process user and assistant messages — system/tool messages don't
     * contain conversational content worth extracting.
     */
    async ingest(params): Promise<IngestResult> {
      // Skip heartbeat messages — they're internal OpenClaw polling, not real conversation
      if (params.isHeartbeat) return { ingested: false };

      // Only extract from user and assistant messages
      const role = params.message.role;
      if (role !== "user" && role !== "assistant") return { ingested: false };

      const content =
        typeof params.message.content === "string"
          ? params.message.content
          : "";
      if (!content.trim()) return { ingested: false };

      // Forward to sidecar for graph extraction.
      // The sidecar's rule extractor processes immediately;
      // the LLM extractor batches and flushes periodically.
      try {
        await sidecar.request("POST", "/api/chat/ingest", {
          message: content,
          role,
          turn: turnCount,
        });
      } catch (e) {
        // Non-fatal: if sidecar is temporarily unavailable, we lose this
        // turn's graph extraction but the session continues
        logger.warn(`PANIC ingest failed: ${e}`);
      }

      return { ingested: true };
    },

    /**
     * Assemble model context with PANIC's layered memory injection.
     *
     * This is the core integration point. OpenClaw gives us all session messages
     * and a token budget. We:
     *   1. Take the most recent user message as the retrieval query
     *   2. Call the sidecar to query all 4 memory layers
     *   3. Build a system prompt addition with labeled memory sections
     *   4. Return the original messages (for conversation history) plus
     *      the memory context as a system prompt addition
     *
     * We don't filter/reorder the messages themselves — that's OpenClaw's job.
     * We just add memory context that the LLM wouldn't otherwise see.
     */
    async assemble(params): Promise<AssembleResult> {
      const messages = params.messages;

      // Find the latest user message to use as retrieval query
      const query = params.prompt || findLastUserMessage(messages);

      let systemPromptAddition = "";

      if (query && sidecar.isReady) {
        try {
          // Ask the sidecar to build layered context for this query.
          // This calls _build_memory_layers() → _retrieve() → construct_layered_prompt()
          // on the Python side and returns the assembled context sections.
          const result = await sidecar.request("POST", "/api/assemble", {
            query,
            token_budget: params.tokenBudget ?? 200_000,
          });

          if (result.context_section) {
            systemPromptAddition = result.context_section;
          }
        } catch (e) {
          // Non-fatal: if sidecar fails, the LLM just won't have memory context
          logger.warn(`PANIC assemble failed: ${e}`);
        }
      }

      // Estimate tokens (rough: 4 chars per token)
      const estimatedTokens = Math.ceil(
        messages.reduce(
          (sum: number, m: any) =>
            sum +
            (typeof m.content === "string" ? m.content.length / 4 : 10),
          0
        ) + systemPromptAddition.length / 4
      );

      return {
        messages,
        estimatedTokens,
        // This gets prepended to the system prompt, so the LLM sees
        // PANIC's memory context at the top of its instructions
        systemPromptAddition: systemPromptAddition || undefined,
      };
    },

    /**
     * Post-turn lifecycle work.
     *
     * Called after each model run completes. We use this to:
     *   - Increment our turn counter
     *   - Trigger intermediate extraction every 50 turns (crash protection)
     *   - Save profile state periodically
     */
    async afterTurn(params) {
      if (params.isHeartbeat) return;

      turnCount++;

      // Every 50 turns, trigger intermediate extraction and save
      if (turnCount > 0 && turnCount % 50 === 0) {
        try {
          await sidecar.request("POST", "/api/session/save", {});
          logger.info(`PANIC: saved state at turn ${turnCount}`);
        } catch (e) {
          logger.warn(`PANIC: periodic save failed: ${e}`);
        }
      }
    },

    /**
     * Compact context when the window is full.
     *
     * PANIC doesn't own compaction — it lets OpenClaw handle transcript
     * summarization. PANIC's persistent memory (profiles, episodes, semantic,
     * procedural) lives outside the session transcript entirely.
     *
     * We do trigger a session save here though, since compaction is a good
     * signal that significant conversation has happened.
     */
    async compact(params): Promise<CompactResult> {
      // Save current state to profile before compaction
      try {
        await sidecar.request("POST", "/api/session/save", {});
      } catch (e) {
        logger.warn(`PANIC: save before compaction failed: ${e}`);
      }

      // Return not-compacted — OpenClaw will run its own compaction
      // since ownsCompaction is false
      return { ok: true, compacted: false, reason: "delegated to runtime" };
    },

    /**
     * Clean up when the engine is disposed (gateway shutdown).
     * Triggers session-end extraction to persist accumulated knowledge,
     * then the sidecar service handles process shutdown separately.
     */
    async dispose() {
      // Guard: sidecar may already be stopped if the service shutdown ran first
      if (!sidecar.isReady) {
        logger.info("PANIC: sidecar already stopped, skipping dispose extraction");
        return;
      }
      try {
        // Run session-end extraction to save episode + semantic + procedural updates
        await sidecar.request("POST", "/api/session/end", {});
        logger.info("PANIC: session-end extraction completed on dispose");
      } catch (e) {
        logger.warn(`PANIC: session-end extraction failed on dispose: ${e}`);
      }
    },
  };
}

/**
 * Find the last user message in the conversation history.
 * Used as the retrieval query when params.prompt isn't available.
 */
function findLastUserMessage(messages: any[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (
      messages[i].role === "user" &&
      typeof messages[i].content === "string"
    ) {
      return messages[i].content as string;
    }
  }
  return "";
}
