/**
 * Council i18n audit: every council string the backend can serve renders in
 * English, and every translation preserves the science.
 *
 * `npm run audit:i18n`. Two assertions, both against `results/golden_seed21.sqlite3`
 * (read-only) and `api_contract.json`:
 *
 * 1. **Coverage** — each CJK string in the ledger has a translation-table entry
 *    (or is handled by the segment fallback without leaving CJK behind). A miss
 *    here means a memo would render in Chinese, or in `[untranslated …]` markers.
 * 2. **Fidelity** — every *structural token* in the raw string survives into the
 *    English rendering: candidate ids, hypothesis ids, position indices of the
 *    form `W3`/`G6`/`Y10`, 13-mer sequences, numeric readings, and the contract
 *    field names the memo quotes. This is the check that stops a fluent
 *    translation from quietly moving or dropping a number.
 *
 * It also prints, per memo, how many such tokens were verified, so the coverage
 * of the audit is visible rather than asserted.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

import { normalizeCouncilText, CJK_PATTERN } from "../src/utils/normalizeCouncilText";
import { COUNCIL_TRANSLATIONS } from "../src/utils/councilTranslations";
import { councilHash } from "../src/utils/councilHash";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const ledgerPath = resolve(root, "../results/golden_seed21.sqlite3");
const contractPath = resolve(root, "../api_contract.json");

const failures: string[] = [];
let checks = 0;

function expect(label: string, condition: boolean, detail: string): void {
  checks += 1;
  if (!condition) failures.push(`${label}: ${detail}`);
}

// ── Gather every council string the backend can serve ─────────────────────
interface Memo {
  role?: unknown;
  memo?: unknown;
  thinking_trace?: unknown;
}

const strings: Array<{ text: string; round: number; role: string; field: string; origin: string }> = [];

const db = new DatabaseSync(ledgerPath, { readOnly: true });
const rows = db
  .prepare("SELECT event_type, round_number, payload FROM entries ORDER BY round_number, sequence_no")
  .all() as Array<{ event_type?: unknown; round_number?: unknown; payload?: unknown }>;
db.close();

for (const row of rows) {
  if (row.event_type !== "round_council") continue;
  const parsed =
    typeof row.payload === "string"
      ? (JSON.parse(row.payload) as { memos?: Memo[] })
      : ((row.payload ?? {}) as { memos?: Memo[] });
  const round = typeof row.round_number === "number" ? row.round_number : 0;
  for (const memo of parsed.memos ?? []) {
    const role = typeof memo.role === "string" ? memo.role : "unknown";
    for (const field of ["memo", "thinking_trace"] as const) {
      const value = memo[field];
      if (typeof value === "string" && value.trim()) {
        strings.push({ text: value, round, role, field, origin: "ledger" });
      }
    }
  }
}

const contract = JSON.parse(readFileSync(contractPath, "utf8")) as {
  state?: { agent_council?: Memo[] };
};
for (const seat of contract.state?.agent_council ?? []) {
  const role = typeof seat.role === "string" ? seat.role : "unknown";
  for (const field of ["memo", "thinking_trace"] as const) {
    const value = seat[field];
    if (typeof value === "string" && value.trim()) {
      strings.push({ text: value, round: -1, role, field, origin: "contract" });
    }
  }
}

// ── 1. Coverage ───────────────────────────────────────────────────────────
const cjkStrings = strings.filter((entry) => CJK_PATTERN.test(entry.text));
const uniqueCjk = new Map<string, (typeof strings)[number]>();
for (const entry of cjkStrings) uniqueCjk.set(entry.text, entry);

console.log(
  `${strings.length} council strings served (${uniqueCjk.size} distinct CJK) across ${
    new Set(strings.map((entry) => entry.round)).size
  } rounds`,
);

for (const [text, meta] of uniqueCjk) {
  const normalized = normalizeCouncilText(text);
  const hash = councilHash(text);
  const label = `R${meta.round} ${meta.role}/${meta.field} [${hash}]`;

  expect(`${label} has a translation entry`, hash in COUNCIL_TRANSLATIONS, "missing from councilTranslations.ts");
  expect(
    `${label} renders without CJK`,
    !CJK_PATTERN.test(normalized.text),
    `still contains CJK: ${normalized.text.match(/[一-鿿]+/)?.[0] ?? ""}`,
  );
  expect(`${label} is marked translated`, normalized.source === "translated", normalized.source);
  expect(`${label} keeps the raw string`, typeof normalized.raw === "string" && normalized.raw === text, "raw lost");

  // ── 2. Fidelity ─────────────────────────────────────────────────────────
  // Structural tokens that must survive. Each pattern targets something a
  // translation could plausibly mangle: an id, a position index, a sequence, a
  // number, or a contract field name.
  const patterns: Array<{ name: string; regex: RegExp }> = [
    { name: "candidate id", regex: /\bCAND-\d{3}\b/g },
    { name: "hypothesis id", regex: /\bHYP-R\d+-\d+\b/g },
    { name: "position index", regex: /\b[A-Z]\d{1,2}\b/g },
    { name: "contract field", regex: /\b[a-z]+(?:_[a-z0-9]+)+\b/g },
    { name: "number", regex: /-?\d+(?:\.\d+)?(?:e-?\d+)?/g },
    { name: "sequence", regex: /\b[A-Z]{5,13}\b/g },
  ];

  let verified = 0;
  for (const pattern of patterns) {
    const rawMatches = new Set((text.match(pattern.regex) ?? []).map((token) => token));
    for (const token of rawMatches) {
      // A token that already appears as an ASCII run must appear in the English
      // rendering verbatim. Anything else means the translator paraphrased a
      // value instead of carrying it across.
      expect(
        `${label} preserves ${pattern.name} "${token}"`,
        normalized.text.includes(token),
        "token dropped or altered by the translation",
      );
      verified += 1;
    }
  }
  console.log(`  ✓ ${label.padEnd(58)} ${String(verified).padStart(4)} tokens preserved`);
}

// ── 3. Every table row survives ───────────────────────────────────────────
// Table structure is what makes a memo scannable; a translation that flattens a
// table into prose would still pass the token check. Compare row counts instead.
for (const [text, meta] of uniqueCjk) {
  const normalized = normalizeCouncilText(text);
  const rawRows = text.split("\n").filter((line) => line.trim().startsWith("|")).length;
  const englishRows = normalized.text.split("\n").filter((line) => line.trim().startsWith("|")).length;
  if (rawRows === 0) continue;
  expect(
    `R${meta.round} ${meta.role}/${meta.field} keeps its table shape`,
    rawRows <= englishRows,
    `${rawRows} source pipe rows vs ${englishRows} translated`,
  );
}

// ── 4. The fallback leaves no CJK behind either ───────────────────────────
// Simulate a future memo the table does not know: the glossary path must still
// produce English (possibly with an explicit marker), never raw Chinese.
const probe = "## 结论先说 CAND-999 的 max_5mer_hydropathy = 1.23 属于高风险。";
const fallback = normalizeCouncilText(probe);
expect("fallback produces English", !CJK_PATTERN.test(fallback.text), `left CJK: ${fallback.text}`);
expect("fallback preserves the candidate id", fallback.text.includes("CAND-999"), fallback.text);
expect("fallback preserves the number", fallback.text.includes("1.23"), fallback.text);
expect("fallback preserves the field name", fallback.text.includes("max_5mer_hydropathy"), fallback.text);
expect("fallback flags itself", fallback.source === "untranslated", fallback.source);

if (failures.length) {
  console.error(`\ncouncil i18n audit FAILED: ${failures.length} of ${checks} assertions`);
  for (const failure of failures.slice(0, 25)) console.error(`  ✗ ${failure}`);
  if (failures.length > 25) console.error(`  … and ${failures.length - 25} more`);
  process.exit(1);
}

console.log(`\ncouncil i18n audit ok: ${checks} assertions over ${uniqueCjk.size} CJK strings`);
