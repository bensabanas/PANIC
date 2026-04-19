/**
 * PANIC OpenClaw Plugin — Entry Point
 *
 * Registers PANIC as a context engine that replaces OpenClaw's default
 * context assembly pipeline. PANIC controls what the LLM sees by querying
 * 4 memory layers (working, semantic, episodic, procedural) and injecting
 * them as labeled sections with per-layer token budgets.
 *
 * Architecture:
 *   - This TypeScript plugin runs inside the OpenClaw gateway process
 *   - It manages a Python sidecar (the PANIC FastAPI backend) via HTTP
 *   - The sidecar handles all retrieval, extraction, and graph operations
 *   - The plugin translates between OpenClaw's ContextEngine interface and
 *     PANIC's REST API
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { Type } from "@sinclair/typebox";
import { createPanicContextEngine } from "./engine.js";
import { PanicSidecar } from "./sidecar.js";

export default definePluginEntry({
  id: "panic",
  name: "PANIC",
  description: "Persistent memory for AI assistants — 4-layer context engine",

  // Mark this as a context-engine plugin so OpenClaw knows it occupies
  // the exclusive context engine slot
  kind: "context-engine",

  register(api) {
    // Read plugin config from openclaw.json → plugins.entries.panic.config
    const config = api.pluginConfig as {
      profile?: string;
      sidecarPort?: number;
      pythonPath?: string;
      panicRoot?: string;
    };

    const port = config.sidecarPort ?? 7420;
    const panicRoot = config.panicRoot ?? findPanicRoot(api.rootDir);
    const pythonPath = config.pythonPath ?? `${panicRoot}/.venv/bin/python`;

    // Create the sidecar manager — handles starting/stopping the Python process
    const sidecar = new PanicSidecar({
      port,
      pythonPath,
      panicRoot,
      logger: api.logger,
    });

    // Register PANIC as the active context engine.
    // OpenClaw will call ingest/assemble/afterTurn/compact on every turn.
    api.registerContextEngine("panic", () =>
      createPanicContextEngine({
        sidecar,
        profile: config.profile ?? "default",
        logger: api.logger,
      })
    );

    // Register a background service that starts the sidecar on gateway boot
    // and stops it on shutdown
    api.registerService({
      id: "panic-sidecar",
      async start() {
        api.logger.info(`Starting PANIC sidecar on port ${port}...`);
        await sidecar.start();
        api.logger.info("PANIC sidecar ready");

        // Connect the LLM extractor so session-end extraction can fire.
        // litellm reads ANTHROPIC_API_KEY from env, so we don't need to pass
        // the actual key — we just need engine.connected = true.
        try {
          await sidecar.request("POST", "/api/connect", {
            provider: "anthropic",
            model: "claude-haiku-4-5-20251001",
            extraction_model: "claude-haiku-4-5-20251001",
          });
          api.logger.info("PANIC sidecar LLM connected");
        } catch (e) {
          api.logger.warn(`Failed to connect PANIC LLM: ${e}`);
        }

        // Switch to configured profile if specified
        if (config.profile && config.profile !== "default") {
          try {
            await sidecar.request("POST", "/api/profiles/switch", {
              name: config.profile,
            });
            api.logger.info(`Switched to profile: ${config.profile}`);
          } catch (e) {
            api.logger.warn(
              `Failed to switch to profile '${config.profile}': ${e}`
            );
          }
        }
      },
      async stop() {
        api.logger.info("Stopping PANIC sidecar...");
        await sidecar.stop();
      },
    });

    // Register a tool so users can check PANIC status and manage profiles
    api.registerTool({
      name: "panic_status",
      label: "PANIC Status",
      description:
        "Check PANIC memory engine status: active profile, turn count, graph stats, memory layer info",
      parameters: Type.Object({}),
      async execute(
        _toolCallId: string,
        _params: unknown,
      ) {
        try {
          const status = await sidecar.request("GET", "/api/status");
          return {
            content: [
              { type: "text" as const, text: JSON.stringify(status, null, 2) },
            ],
            details: status,
          };
        } catch (e) {
          return {
            content: [
              {
                type: "text" as const,
                text: `PANIC sidecar error: ${e}`,
              },
            ],
            details: { error: String(e) },
          };
        }
      },
    });

    api.logger.info("PANIC plugin registered");
  },
});

/**
 * Try to find the PANIC project root by looking relative to the plugin directory.
 * Falls back to a sensible default.
 */
function findPanicRoot(pluginDir?: string | null): string {
  if (pluginDir) {
    // Plugin is inside the panic project: /Users/ben/Desktop/panic/plugin/
    // So panic root is one level up
    const parentDir = pluginDir.replace(/\/plugin\/?$/, "");
    return parentDir;
  }
  // Fallback: assume standard location
  return `${process.env.HOME}/Desktop/panic`;
}
