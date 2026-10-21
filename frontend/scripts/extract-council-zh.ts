/**
 * Extractor: pulls every CJK-bearing council string out of the frozen ledger and
 * writes `src/utils/council-zh-source.json`.
 *
 * The seven strings in that file are **verbatim ledger content** — they are what
 * two frontier models actually returned during the recorded Seed-21 run, in
 * Chinese, with the last two truncated mid-sentence by a completion ceiling. They
 * are kept because the console must be able to show the raw evidence behind a
 * translation (the "Raw ledger" toggle in each council card). They are never
 * rendered by default.
 *
 * Reads `results/golden_seed21.sqlite3` **read-only** via `node:sqlite`, plus
 * `api_contract.json`, and asserts that the two sources agree: every CJK string in
 * the contract must also be in the ledger. Run with `npm run i18n:extract`.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import { councilHash } from "../src/utils/councilHash";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const ledgerPath = resolve(root, "../results/golden_seed21.sqlite3");
const contractPath = resolve(root, "../api_contract.json");
const outPath = resolve(root, "src/utils/council-zh-source.json");

const CJK = /[一-鿿]/;

interface Memo {
  role?: unknown;
  memo?: unknown;
  thinking_trace?: unknown;
}

interface Entry {
  event_type?: unknown;
  round_number?: unknown;
  payload?: unknown;
}

/** `{ hash: raw }` — the raw Chinese, verbatim, for the provenance toggle. */
const source: Record<string, string> = {};
/** `{ hash: [round, role, fields…] }` — where each string was said. */
const provenance: Record<string, { round: number; role: string; fields: string[]; chars: number }> = {};

const db = new DatabaseSync(ledgerPath, { readOnly: true });
const rows = db
  .prepare("SELECT event_type, round_number, payload FROM entries ORDER BY round_number, sequence_no")
  .all() as Entry[];
db.close();

function visit(text: unknown, round: number, role: string, field: string): void {
  if (typeof text !== "string" || !CJK.test(text)) return;
  const hash = councilHash(text);
  source[hash] = text;
  const existing = provenance[hash];
  if (existing) {
    if (!existing.fields.includes(field)) existing.fields.push(field);
    return;
  }
  provenance[hash] = { round, role, fields: [field], chars: text.length };
}

for (const row of rows) {
  if (row.event_type !== "round_council") continue;
  const payload = row.payload as string | { memos?: Memo[] } | null;
  const parsed =
    typeof payload === "string" ? (JSON.parse(payload) as { memos?: Memo[] }) : payload ?? {};
  const round = typeof row.round_number === "number" ? row.round_number : 0;
  for (const memo of parsed.memos ?? []) {
    const role = typeof memo.role === "string" ? memo.role : "unknown";
    visit(memo.memo, round, role, "memo");
    visit(memo.thinking_trace, round, role, "thinking_trace");
  }
}

// Cross-check against the contract: `/api/state` serves the latest council only,
// so it must be a subset of the ledger. A string in the contract but not the
// ledger would mean the two artifacts disagree.
const contract = JSON.parse(readFileSync(contractPath, "utf8")) as {
  state?: { agent_council?: Memo[] };
};
const contractStrings: string[] = [];
for (const seat of contract.state?.agent_council ?? []) {
  for (const field of ["memo", "thinking_trace"] as const) {
    const text = seat[field];
    if (typeof text === "string" && CJK.test(text)) contractStrings.push(text);
  }
}
const missing = contractStrings.filter((text) => !(councilHash(text) in source));
if (missing.length) {
  console.error(`contract strings absent from the ledger: ${missing.length}`);
  process.exit(1);
}

const sorted = Object.fromEntries(Object.entries(source).sort(([a], [b]) => a.localeCompare(b)));
writeFileSync(outPath, `${JSON.stringify(sorted, null, 2)}\n`);

console.log(`${Object.keys(sorted).length} CJK council strings extracted → ${outPath}`);
for (const [hash, meta] of Object.entries(provenance).sort(([, a], [, b]) => a.round - b.round)) {
  const truncated = !/[.!?)\]。！？]$/.test(sorted[hash] ?? "");
  console.log(
    `  ${hash}  R${meta.round}  ${meta.role.padEnd(24)} ${meta.fields.join("+").padEnd(26)} ${String(meta.chars).padStart(5)} chars${truncated ? "  [truncated in ledger]" : ""}`,
  );
}
