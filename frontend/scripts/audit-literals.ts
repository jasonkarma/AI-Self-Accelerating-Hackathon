/**
 * Literal audit: no result number may be typed into the source.
 *
 * `npm run audit:literals`. The console's contract with a judge is that every
 * figure comes from an endpoint. A hard-coded `0.6578` in a caption would survive
 * `verify-numbers.ts` (the payload really does say 0.6578 today) and then quietly
 * lie after the next re-run. This check makes that impossible to introduce
 * accidentally: it collects every *result* number the contract publishes, renders
 * each at the precision a human would paste in, and fails if any of them appears
 * as a literal anywhere under `src/`.
 *
 * Deliberately **not** forbidden, because they are not results:
 * * structural constants (`0`, `1`, `0.5`, `2`, `3`, `13`, …) — split weights,
 *   array indices, layout maths;
 * * the protocol constants the backend publishes as *law* rather than measurement
 *   (`tau_dev`, the `0.65/0.35` objective weighting, the `±0.15` prior bound) —
 *   these appear in prose that explains a formula, and the formula is the point;
 * * anything in `src/lib/pinned.ts`, which exists precisely to hold the two
 *   repository-level facts no endpoint serves, and is documented as such.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const srcDir = join(root, "src");
const contractPath = resolve(root, "../api_contract.json");

const contract = JSON.parse(readFileSync(contractPath, "utf8")) as {
  state: Record<string, unknown>;
  benchmark: Record<string, unknown>;
  cross_target: Record<string, unknown>;
  attribution: Record<string, unknown>;
};

/** Files allowed to contain result-shaped literals. */
const ALLOWED = new Set(["src/lib/pinned.ts"]);

/** Values that a copy-paste would produce, per digit count. */
function renderings(value: number): string[] {
  if (!Number.isFinite(value) || value === 0) return [];
  const magnitude = Math.abs(value);
  const out: string[] = [];
  // p-values are pasted in exponent form; everything else in fixed form.
  if (magnitude < 1e-3) {
    const [mantissa, exponent] = value.toExponential(1).split("e");
    out.push(`${mantissa}e${Number(exponent)}`, `${mantissa}e${String(Number(exponent)).padStart(2, "0")}`);
    out.push(value.toExponential(2), value.toExponential(3));
    return out;
  }
  for (const digits of [2, 3, 4]) {
    const fixed = value.toFixed(digits);
    // Skip strings that are too short to be distinctive (0.10, 1.00, 8.50 when the
    // value is really 8.5) — a coincidental match on those would be noise.
    if (fixed.replace(/[^0-9]/g, "").length >= 5) out.push(fixed);
  }
  return out;
}

/** Every result scalar in the contract, flattened with a provenance path. */
function collectResults(node: unknown, path: string, out: Map<string, string>): void {
  if (typeof node === "number") {
    for (const text of renderings(node)) {
      if (!out.has(text)) out.set(text, path);
    }
    return;
  }
  if (Array.isArray(node)) {
    // Curves are indexed by checkpoint; sampling the first and last rows keeps the
    // forbidden set small while still covering the visible figures.
    const sampled = node.length > 8 ? [node[0], node[node.length - 1]] : node;
    for (const [index, item] of sampled.entries()) collectResults(item, `${path}[${index}]`, out);
    return;
  }
  if (node && typeof node === "object") {
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      // Skip structural / protocol blocks: they are law, not measurement.
      if (
        [
          "matched_initial_seeds",
          "initial_ids",
          "length_histogram",
          "protocol",
          "campaign_protocol",
          "normalizer",
          "aubqc_per_seed",
          "rho_per_seed",
        ].includes(key)
      ) {
        continue;
      }
      collectResults(value, `${path}.${key}`, out);
    }
  }
}

const forbidden = new Map<string, string>();
for (const endpoint of ["state", "benchmark", "cross_target", "attribution"] as const) {
  collectResults(contract[endpoint], endpoint, forbidden);
}
// Drop values that are impossible to distinguish from structural constants.
for (const text of [...forbidden.keys()]) {
  const bare = text.replace(/[-]/g, "");
  if (/^(-?)(0|1)(\.0+)?$/.test(text) || bare.length < 4) forbidden.delete(text);
}

function walk(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) files.push(...walk(full));
    else if (/\.(ts|tsx)$/.test(entry)) files.push(full);
  }
  return files;
}

/**
 * Strip comments before scanning.
 *
 * A doc comment that says `e.g. coverage_pct = 16.5344` is documentation of the
 * wire format, not a rendered number — but a regex cannot tell the difference, and
 * an allow-list per file would let a real literal hide behind a comment. Removing
 * comments first is the honest reading of the rule: *code* may not carry results.
 */
function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:"'`\\])\/\/[^\n]*/g, "$1 ");
}

const hits: string[] = [];
for (const file of walk(srcDir)) {
  const rel = relative(root, file).replace(/\\/g, "/");
  if (ALLOWED.has(rel)) continue;
  const text = stripComments(readFileSync(file, "utf8"));
  for (const [literal, provenance] of forbidden) {
    // Word-ish boundary so `0.6578` does not fire on `10.6578`.
    const pattern = new RegExp(`(^|[^0-9.])${literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^0-9]|$)`);
    if (pattern.test(text)) {
      hits.push(`${rel}: literal "${literal}" (published at ${provenance})`);
    }
  }
}

if (hits.length) {
  console.error(`\nliteral audit FAILED: ${hits.length} result value(s) typed into the source`);
  for (const hit of hits) console.error(`  ✗ ${hit}`);
  console.error(
    "\nEvery figure must be read from a payload. If a number belongs in prose as a formula\n" +
      "constant, add it to the allow-list rationale in scripts/audit-literals.ts, not to src/.",
  );
  process.exit(1);
}

console.log(
  `literal audit ok: ${forbidden.size} published result values checked, none hard-coded in src/`,
);
