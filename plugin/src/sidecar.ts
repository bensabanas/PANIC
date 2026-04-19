/**
 * PANIC Sidecar Manager
 *
 * Manages the Python FastAPI process that runs the PANIC retrieval engine.
 * The sidecar is started as a child process on gateway boot and stopped
 * on shutdown. All communication is via HTTP on localhost.
 *
 * Why a sidecar instead of WASM/native?
 *   - PANIC's retrieval depends on numpy, sentence-transformers, and SQLite
 *     with Python bindings — none of which run natively in Node
 *   - The Python codebase is already complete and tested
 *   - HTTP is simple, debuggable, and lets us run the API standalone too
 *   - Latency is <1ms for localhost HTTP (negligible vs 8-11ms retrieval)
 */

import { spawn, type ChildProcess } from "node:child_process";

/** Configuration for the PANIC sidecar process */
export interface SidecarConfig {
  /** Port the FastAPI server listens on */
  port: number;
  /** Path to the Python binary (usually the venv python) */
  pythonPath: string;
  /** Path to the PANIC project root (where panic/ package lives) */
  panicRoot: string;
  /** Logger from the plugin API */
  logger: { info: (msg: string) => void; warn: (msg: string) => void; error: (msg: string) => void };
}

export class PanicSidecar {
  private process: ChildProcess | null = null;
  private baseUrl: string;
  private config: SidecarConfig;
  private ready = false;

  constructor(config: SidecarConfig) {
    this.config = config;
    this.baseUrl = `http://127.0.0.1:${config.port}`;
  }

  /**
   * Start the Python sidecar process.
   * Runs uvicorn serving panic.api:app on the configured port.
   * Waits until the /api/status endpoint responds before resolving.
   */
  async start(): Promise<void> {
    if (this.process) {
      this.config.logger.warn("Sidecar already running, skipping start");
      return;
    }

    // Start uvicorn as a child process.
    // DYLD_LIBRARY_PATH is needed on macOS for the expat library (xml parsing in Python).
    // PYTHONPATH ensures the panic package is importable.
    this.process = spawn(
      this.config.pythonPath,
      [
        "-m",
        "uvicorn",
        "panic.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(this.config.port),
        "--log-level",
        "warning",
      ],
      {
        cwd: this.config.panicRoot,
        env: {
          ...process.env,
          DYLD_LIBRARY_PATH: "/opt/homebrew/opt/expat/lib",
          PYTHONPATH: this.config.panicRoot,
        },
        stdio: ["ignore", "pipe", "pipe"],
      }
    );

    // Log stdout/stderr from the sidecar for debugging
    this.process.stdout?.on("data", (data: Buffer) => {
      const msg = data.toString().trim();
      if (msg) this.config.logger.info(`[panic-sidecar] ${msg}`);
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      const msg = data.toString().trim();
      if (msg) this.config.logger.info(`[panic-sidecar] ${msg}`);
    });

    // Handle unexpected exits
    this.process.on("exit", (code) => {
      this.ready = false;
      if (code !== 0 && code !== null) {
        this.config.logger.error(`PANIC sidecar exited with code ${code}`);
      }
      this.process = null;
    });

    // Wait for the sidecar to be ready (poll /api/status)
    await this.waitForReady(30_000);
    this.ready = true;
  }

  /**
   * Stop the sidecar process gracefully.
   * Sends SIGTERM and waits up to 5 seconds before force-killing.
   */
  async stop(): Promise<void> {
    if (!this.process) return;

    this.ready = false;
    const proc = this.process;
    this.process = null;

    // Try graceful shutdown
    proc.kill("SIGTERM");

    // Wait up to 5 seconds for exit
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        proc.kill("SIGKILL"); // force kill if still alive
        resolve();
      }, 5000);

      proc.on("exit", () => {
        clearTimeout(timeout);
        resolve();
      });
    });
  }

  /**
   * Make an HTTP request to the sidecar API.
   *
   * @param method - HTTP method (GET, POST, PATCH, DELETE)
   * @param path - API path (e.g. "/api/status")
   * @param body - Optional JSON body for POST/PATCH requests
   * @returns Parsed JSON response
   * @throws Error if the sidecar is not ready or the request fails
   */
  async request(
    method: string,
    path: string,
    body?: Record<string, unknown>
  ): Promise<any> {
    if (!this.ready) {
      throw new Error("PANIC sidecar is not ready");
    }

    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
    };

    if (body && (method === "POST" || method === "PATCH")) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`PANIC API error ${response.status}: ${text}`);
    }

    return response.json();
  }

  /** Check if the sidecar is currently running and responsive */
  get isReady(): boolean {
    return this.ready;
  }

  /**
   * Poll the sidecar's /api/status endpoint until it responds.
   * Used during startup to wait for uvicorn to be ready.
   *
   * @param timeoutMs - Maximum time to wait before giving up
   * @throws Error if the sidecar doesn't respond within the timeout
   */
  private async waitForReady(timeoutMs: number): Promise<void> {
    const start = Date.now();
    const pollInterval = 500; // check every 500ms

    while (Date.now() - start < timeoutMs) {
      try {
        const response = await fetch(`${this.baseUrl}/api/status`);
        if (response.ok) return; // sidecar is ready
      } catch {
        // Connection refused — sidecar not yet listening, keep polling
      }
      await new Promise((r) => setTimeout(r, pollInterval));
    }

    throw new Error(
      `PANIC sidecar did not start within ${timeoutMs / 1000}s`
    );
  }
}
