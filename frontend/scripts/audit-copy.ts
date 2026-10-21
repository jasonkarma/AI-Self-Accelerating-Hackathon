/**
 * Copy audit: the raw-statistics vocabulary may not come back onto the surface.
 *
 * `npm run audit:copy`. This is the companion to `audit-literals.ts`. That one
 * stops a *number* being typed into the source; this one stops a *term* being
 * rendered where a plain-language label belongs.
 *
 * The redesign's rule is narrow and mechanical:
 *
 * > A statistic's internal name may appear in `src/lib/plain.ts` (where it is
 * > declared), in a `TokenChip`/`InfoTip` (where it is defined), in an attribute
 * > (`title=`, `aria-label=`, `token=`, `formula=`), or as a **code identifier**
 * > (`slot.a_h1`). It may not appear as visible body copy.
 *
 * A regex cannot see the DOM, so this check reads the source and flags the two
 * shapes that actually caused the problem:
 *
 * 1. **A jargon token in a JSX text node or a rendered string** — `Spearman ρ`,
 *    `AUBQC`, `Δ / SE`, `η_SAR`, `N_qual` … written straight into the markup.
 *    Property accesses and string-literal *keys* are stripped first, so
 *    `slot.a_h1` and `diagnostics.steers_null_objective` do not trip it.
 * 2. **A headline statistic formatted by hand** — `rho.toFixed(..)` in a view.
 *    The views must format through `lib/format.ts` / `lib/plain.ts`, which is
 *    what stopped the same figure appearing at three precisions across four
 *    tabs.
 *
 * Deliberately allowed, because they are not surface copy:
 * * definitions and provenance (`InfoTip`, `TokenChip`, `title=`, `formula=`);
 * * protocol constants and axis formatters, where the precision *is* the design;
 * * comments, so a doc comment may name the term it explains.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const srcDir = join(root, "src");

/**
 * The jargon the redesign removed from the surface, with the plain label that
 * replaced it. The message names the replacement so a failure is a fix
 * instruction rather than a scolding.
 */
const BANNED: Array<{ pattern: RegExp; replacement: string }> = [
    { pattern: /\bAUBQC\b/, replacement: "“best binder found” (PLAIN.aubqc.label)" },
    { pattern: /\bSpearman\s*(?:ρ|rho)\b/i, replacement: "“ranking accuracy” (PLAIN.spearman_rho.label)" },
    { pattern: /Δ\s*\/\s*SE/, replacement: "“gap vs. run-to-run noise” (noiseText)" },
    { pattern: /\bN_qual\b/, replacement: "“usable candidates”" },
    { pattern: /η_SAR/, replacement: "“how much the prior is trusted”" },
    { pattern: /\bVoI\b/, replacement: "“how much a test is worth”" },
    { pattern: /−log₁₀\s*K/, replacement: "“binding strength”" },
    { pattern: /\bH[01]\s*(?:vs\.?|\/)\s*H[01]\b/, replacement: "“hypothesis vs. control”" },
    { pattern: /\bzero-shot\b/i, replacement: "“off-the-shelf predictors”" },
    { pattern: /\blockbox\b/i, replacement: "“the holdout”" },
    // The baseline's old product name. Renamed to "Standard UCB" so screen and
    // speech agree, and enforced here because a rename that is not checked drifts
    // back the first time someone writes a new caption. "UCB" also makes Beat 3's
    // "a UCB loop is copyable in a weekend" a callback rather than a new term.
    { pattern: /\bstandard method\b/i, replacement: "“Standard UCB” (methodName('m3_ucb'))" },
];

/**
 * A metric name formatted by hand in a view. Precision drift is the failure this
 * catches: `0.6578` in one panel, `0.657` in another, `0.66` on an axis.
 */
const HAND_FORMATTED = /\.(?:rho|aubqc|mean_diff|se|p_value)\w*\.toFixed\(/;

/**
 * Files where the tokens are the *subject*: the vocabulary table, the number
 * formatter's method metadata, and the type definitions transcribed from the
 * API contract. Banning the tokens there would delete the mapping itself.
 */
const ALLOWED_FILES = new Set([
    "src/lib/plain.ts",
    "src/lib/format.ts",
    "src/types/report.ts",
    "src/types/state.ts",
]);

/**
 * Strip every surface where a token is legitimate.
 *
 * Order matters: attributes and component bodies are removed before the
 * property-access rule, because `token={PLAIN.x.token}` must not be mistaken for
 * a property access that survived.
 */
function stripDefinitionSurfaces(source: string): string {
    return (
        source
            // JSX attributes that carry a definition rather than a label.
            .replace(
                /\b(title|aria-label|aria-description|token|formula|hint|meaning)\s*=\s*("(?:[^"\\]|\\.)*"|\{(?:[^{}]|\{[^{}]*\})*\})/g,
                " ",
            )
            // <InfoTip …>…</InfoTip> and <TokenChip …>…</TokenChip> bodies.
            .replace(/<(InfoTip|TokenChip)\b[\s\S]*?<\/\1>/g, " ")
            // Comments, so a doc comment may name the term it explains.
            .replace(/\/\*[\s\S]*?\*\//g, " ")
            .replace(/(^|[^:"'`\\])\/\/[^\n]*/g, "$1 ")
            // Property accesses and object keys: `slot.a_h1`, `row.N_qual`,
            // `{ a_null: … }`. These are identifiers, never rendered text.
            .replace(/\.\s*(?:a_h1|a_null|delta_agent|voi|N_qual|slot_graduation|steers_null_objective)\b/g, " ")
            .replace(/\b(?:slot_graduation|steers_null_objective|delta_agent)\s*:/g, " ")
            // A local variable named for the concept is an identifier too. The copy
            // audit checks what is *rendered*, not what an internal binding is called.
            .replace(/\blockbox\b(?=\s*[?:=,)\]])/g, " ")
            .replace(/\b(?:const|let|var)\s+lockbox\b/g, " ")
    );
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

const hits: string[] = [];

for (const file of walk(srcDir)) {
    const rel = relative(root, file).replace(/\\/g, "/");
    if (ALLOWED_FILES.has(rel)) continue;

    const raw = readFileSync(file, "utf8");
    const text = stripDefinitionSurfaces(raw);
    const lines = text.split("\n");

    for (const [index, line] of lines.entries()) {
        for (const { pattern, replacement } of BANNED) {
            if (pattern.test(line)) {
                hits.push(`${rel}:${index + 1}  ${line.trim().slice(0, 100)}\n      → use ${replacement}`);
            }
        }
        if (HAND_FORMATTED.test(line)) {
            hits.push(
                `${rel}:${index + 1}  ${line.trim().slice(0, 100)}\n      → format through lib/format.ts or lib/plain.ts`,
            );
        }
    }
}

if (hits.length) {
    console.error(`\ncopy audit FAILED: ${hits.length} jargon/precision drift(s) on the surface`);
    for (const hit of hits) console.error(`  ✗ ${hit}`);
    console.error(
        "\nA statistic's internal name belongs in lib/plain.ts, a TokenChip or a tooltip —\n" +
        "never in visible body copy. The plain label is what a judge reads in the demo.",
    );
    process.exit(1);
}

console.log(
    `copy audit ok: ${BANNED.length} jargon patterns and hand-formatted statistics checked, none on the surface`,
);
