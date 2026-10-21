/**
 * Provenance constants that **no endpoint serves**.
 *
 * Everything else in this app is computed from `/api/state`, `/api/benchmark`,
 * `/api/cross-target`, `/api/attribution` (plus `/api/ledger`, `/api/rounds` and
 * `/api/config`, which are control routes on the same server). These three
 * values are different in kind: they are claims about the *repository*, not
 * measurements of a run, so there is nowhere to fetch them from.
 *
 * They are declared once, here, and rendered with a tooltip that says where they
 * came from. The live half of the provenance pill — the ledger hash-chain head
 * and its verification — is fetched from `GET /api/ledger` and is never faked.
 */
export const PINNED = {
  /** SHA-256 of `results/golden_seed21.sqlite3` at the locked commit `28b2520`. */
  goldenLedgerSha256: "49fea618572b3836d06cc97c4be9ae9efd93bc9988a6ba824968dec7b9a3aad9",
  /** Backend suite result recorded for the same commit. */
  backendTests: "109 / 109",
  /** The production objective law shipped at `e567cd4` and still in force. */
  objectiveLaw: "a_null := a_base",
  lockedCommit: "28b2520",
} as const;

/** First 8 hex characters — the form the pill shows before the tooltip. */
export function shortSha(sha: string, chars = 8): string {
  return sha.slice(0, chars);
}
