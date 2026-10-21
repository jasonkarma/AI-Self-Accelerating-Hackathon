/**
 * Typed fetch layer for the FastAPI backend.
 *
 * Design rules:
 * * The base URL is relative by default (`/api/...`) so Vite's dev proxy, a
 *   reverse proxy, or `vite preview` all work without a rebuild. `VITE_API_BASE`
 *   overrides it for a cross-origin static deploy (the backend's CORS is `*`).
 * * Every response is parsed as `unknown` and narrowed by the caller's declared
 *   type. There is deliberately no `any` anywhere: an unexpected payload surfaces
 *   as a `ApiError` with the endpoint and status, not as a silent `undefined`
 *   that a chart renders as zero.
 *
 * The override is read through `import.meta.env`, which Vite substitutes at build
 * time. It is read defensively because the offline render harness imports this
 * module in plain Node, where `import.meta.env` does not exist — a module-level
 * throw there would fail every smoke test for a reason unrelated to the code.
 */
const env = (import.meta as { env?: { VITE_API_BASE?: string } }).env;
export const API_BASE: string = (env?.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  readonly url: string;
  readonly status: number | null;

  constructor(message: string, url: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.url = url;
    this.status = status;
  }
}

function url(path: string): string {
  return `${API_BASE}${path}`;
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  const target = url(path);
  let response: Response;
  try {
    response = await fetch(target, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the backend at ${target}. Start it with \`uvicorn app.server:app --port 8000\`.`,
      target,
      null,
    );
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        const raw = (body as { detail: unknown }).detail;
        detail = typeof raw === "string" ? raw : JSON.stringify(raw);
      }
    } catch {
      // Non-JSON error body: the status line is already the best available message.
    }
    throw new ApiError(detail, target, response.status);
  }
  return (await response.json()) as unknown;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = signal ? { signal } : {};
  return (await request(path, init)) as T;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  };
  return (await request(path, init)) as T;
}

// The four showcase endpoints use separate import sites so a missing field is
// attributed to the right payload at the type level.

export const endpoints = {
  state: (round?: number, signal?: AbortSignal) =>
    getJson<import("../types/state").StatePayload>(
      round === undefined ? "/api/state" : `/api/state?round=${round}`,
      signal,
    ),
  benchmark: (signal?: AbortSignal) =>
    getJson<import("../types/report").BenchmarkPayload>("/api/benchmark", signal),
  crossTarget: (signal?: AbortSignal) =>
    getJson<import("../types/report").CrossTargetPayload>("/api/cross-target", signal),
  attribution: (signal?: AbortSignal) =>
    getJson<import("../types/report").AttributionPayload>("/api/attribution", signal),

  // Control routes on the same server (app/server.py).
  rounds: (signal?: AbortSignal) =>
    getJson<import("../types/state").RoundsPayload>("/api/rounds", signal),
  config: (signal?: AbortSignal) =>
    getJson<import("../types/state").ConfigPayload>("/api/config", signal),
  ledger: (signal?: AbortSignal) =>
    getJson<import("../types/state").LedgerPayload>("/api/ledger", signal),
  health: (signal?: AbortSignal) =>
    getJson<import("../types/state").HealthPayload>("/api/health", signal),

  rewind: (round: number) =>
    postJson<import("../types/state").RewindResult>("/api/rewind", { round }),
  restoreGolden: () =>
    postJson<import("../types/state").RewindResult>("/api/restore-golden"),
  runRound: () => postJson<{ accepted: boolean; stream: string }>("/api/round/run"),
};

/**
 * The SSE URL for one round. `EventSource` cannot send a POST body, and the
 * backend's stream endpoint is a GET that triggers the round itself, so the
 * client connects and lets the server do the work.
 */
export function streamRoundUrl(): string {
  return url("/api/stream-round");
}

/** One parsed SSE message from `/api/stream-round`. */
export interface StreamEvent {
  event: string;
  data: Record<string, unknown>;
}
