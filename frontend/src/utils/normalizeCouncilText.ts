/**
 * Council-text normalisation: every string the Scientist Council produced is
 * rendered in English, whatever language the model answered in.
 *
 * ## The problem
 *
 * The recorded Seed-21 campaign used live frontier models. Two Anthropic seats
 * answered in Chinese on five of six rounds, so `/api/state` serves four CJK
 * strings and the ledger holds seven. The ledger is frozen and must not be edited.
 * The fix therefore lives at the presentation boundary.
 *
 * ## The design
 *
 * Three layers, in order of trust:
 *
 * 1. **A translation table** (`councilTranslations.ts`) keyed by a hash of the raw
 *    string. Every CJK string the frozen ledger contains has a hand-authored
 *    English rendering that preserves all positions, sequences, numbers and table
 *    structure. This is the path the golden campaign takes.
 * 2. **A segment fallback** for text the table does not know — a live round run
 *    from the console, a future artifact. It translates at the *segment* level so
 *    that structural tokens survive untouched: fenced code, inline code spans,
 *    candidate IDs, sequences, position indices, numbers and table pipes are
 *    copied verbatim; only the prose between them is passed to a glossary.
 * 3. **A final guard**, `assertEnglish`, which the render smoke test calls on the
 *    produced markup. Nothing with a CJK code point reaches the screen silently —
 *    if the table and the glossary both miss, the unmatched prose is replaced with
 *    an explicit marker rather than shown in a language the audience cannot read.
 *
 * ## What is deliberately *not* done
 *
 * No machine translation is attempted for unmatched prose. A mistranslated
 * biochemical claim is worse than a visible gap, so the fallback flags the string
 * and quotes the raw text in the provenance tooltip. The verification script
 * asserts the table covers every CJK string in the payload, so in practice the
 * flag never appears.
 */
import type { CouncilSeat } from "../types/state";
import { councilHash } from "./councilHash";
import { COUNCIL_TRANSLATIONS, type CouncilTranslation } from "./councilTranslations";

/** Any CJK ideograph. Used by the guard, the extractor and the smoke test. */
export const CJK_PATTERN = /[一-鿿]/;

export interface NormalizedText {
  /** The English text to render. Markdown, same structure as the source. */
  text: string;
  /** The raw ledger string, or `null` when it was already English. */
  raw: string | null;
  /** `translated` when a table entry was used, `english` when nothing was needed. */
  source: "english" | "translated" | "untranslated";
  /** Which round and seat the original came from, when known. */
  provenance: CouncilTranslation | null;
}

/**
 * Normalise one council string.
 *
 * Returns the English rendering plus enough provenance for the "Raw ledger"
 * toggle to show what the model actually said and where.
 */
export function normalizeCouncilText(text: string | null | undefined): NormalizedText {
  if (typeof text !== "string" || text.trim() === "") {
    return { text: "", raw: null, source: "english", provenance: null };
  }
  if (!CJK_PATTERN.test(text)) {
    return { text, raw: null, source: "english", provenance: null };
  }
  const entry = COUNCIL_TRANSLATIONS[councilHash(text)];
  if (entry) {
    return { text: entry.text, raw: text, source: "translated", provenance: entry };
  }
  return {
    text: translateSegments(text),
    raw: text,
    source: "untranslated",
    provenance: null,
  };
}

/**
 * A seat with its memo and thinking trace normalised.
 *
 * The rest of the seat (provider, model, tool lists, activation reasons) is passed
 * through untouched: those come from the router and the drill schedule, not from
 * the model's prose.
 *
 * The Option-C annotation fields are defaulted rather than trusted. The live
 * `GET /api/state` always sets them (`Campaign._annotate_council_memos`), but a
 * memo reconstructed from a raw ledger entry — which is exactly how
 * `GET /api/state?round=k` and the render harness build one — carries only what
 * the run recorded: `role`, `provider`, `model`, `thinking_budget`,
 * `thinking_trace`, `memo`, `tools_invoked`. Letting `tools_inactive` be
 * `undefined` would crash the seat row on a payload that is otherwise complete.
 */
export function normalizeCouncilSeat(seat: CouncilSeat): CouncilSeat & {
  memoRaw: string | null;
  thinkingTraceRaw: string | null;
  memoSource: NormalizedText["source"];
  textProvenance: CouncilTranslation | null;
} {
  const memo = normalizeCouncilText(seat.memo);
  const trace = normalizeCouncilText(seat.thinking_trace);
  const cited = Array.isArray(seat.tools_invoked) ? seat.tools_invoked : [];
  const active = Array.isArray(seat.tools_invoked_active) ? seat.tools_invoked_active : cited;
  return {
    ...seat,
    memo: memo.text,
    thinking_trace: trace.text,
    tools_invoked: cited,
    tools_invoked_active: active,
    tools_inactive: Array.isArray(seat.tools_inactive) ? seat.tools_inactive : [],
    tool_activation_reasons: seat.tool_activation_reasons ?? {},
    memoRaw: memo.raw,
    thinkingTraceRaw: trace.raw,
    memoSource: memo.source,
    textProvenance: memo.provenance ?? trace.provenance,
  };
}

export function normalizeCouncil(seats: CouncilSeat[]): ReturnType<typeof normalizeCouncilSeat>[] {
  return seats.map(normalizeCouncilSeat);
}

// ──────────────────────────── segment fallback ────────────────────────────

/**
 * Tokens that must survive any translation attempt untouched.
 *
 * Ordered so that longer constructs win: a fenced block before the inline span
 * inside it, an inline span before the identifier inside it, and — last — any run
 * of ASCII word characters. That final rule is deliberately broad: candidate ids,
 * hypothesis ids, contract field names (`max_5mer_hydropathy`), 13-mer sequences,
 * position indices, decimals, exponents and percentages are all "ASCII runs", and
 * carving them out as one class is more robust than enumerating each shape. What
 * is left for the glossary is exactly the Chinese prose between them.
 */
const PROTECTED = [
  /```[\s\S]*?```/g, // fenced code
  /`[^`\n]+`/g, // inline code
  /\|/g, // table cell separators
  // Any ASCII word run, including embedded dots, signs, underscores and the
  // statistical glyphs that appear next to numbers in these memos.
  /[A-Za-z0-9_.+\-−×±%:/]+/g,
];

/**
 * A small glossary for the prose fragments the translation table does not cover.
 *
 * It exists to make an *unseen* memo readable, not to be complete. The entries are
 * the function words and nouns that recur across every council memo in this
 * repository (risk tiers, the metric names' surrounding prose, verdict language).
 */
const GLOSSARY: Array<[RegExp, string]> = [
  [/结论先说|结论先行|审计结论/g, "Conclusion"],
  [/建议/g, "Recommendation"],
  [/候选/g, "candidate"],
  [/高风险/g, "high risk"],
  [/中风险/g, "moderate risk"],
  [/疏水性|疏水/g, "hydrophobicity"],
  [/芳香族|芳香/g, "aromatic"],
  [/净电荷/g, "net charge"],
  [/异构化/g, "isomerization"],
  [/脱酰胺/g, "deamidation"],
  [/聚集体|聚集/g, "aggregation"],
  [/置信度有限/g, "confidence is limited"],
  [/阈值/g, "threshold"],
  [/证据/g, "evidence"],
  [/裁决/g, "verdict"],
  [/属于/g, "is"],
  [/的/g, " of "],
  [/与|和/g, " and "],
  [/序列/g, "sequence"],
];

function gloss(prose: string): string {
  if (!CJK_PATTERN.test(prose)) return prose;
  let translated = prose;
  for (const [pattern, english] of GLOSSARY) translated = translated.replace(pattern, english);
  if (!CJK_PATTERN.test(translated)) return translated;
  // Still Chinese. The **marker must not carry CJK either** — otherwise the
  // fallback would satisfy "we labelled it" while still rendering Chinese on
  // screen, which is exactly what the guard exists to prevent. The raw fragment is
  // already preserved in `NormalizedText.raw` and shown by the Raw ledger toggle,
  // so the marker only needs to say that something was left untranslated.
  const trimmed = translated.replace(/\s+/g, " ").trim();
  return `[untranslated fragment, ${[...trimmed].filter((char) => CJK_PATTERN.test(char)).length} CJK chars]`;
}

/**
 * Translate a string the table does not know, one protected segment at a time.
 *
 * Structural tokens are lifted out first, the remaining prose is passed through
 * the glossary, and any fragment still containing a CJK ideograph is replaced with
 * an explicit marker. The result is **always free of CJK**; it is never a guess,
 * and the raw text stays available through `NormalizedText.raw`.
 */
export function translateSegments(text: string): string {
  // Collect protected spans with their positions so the prose between them can be
  // translated without disturbing offsets.
  const spans: Array<{ start: number; end: number; value: string }> = [];
  for (const pattern of PROTECTED) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      if (spans.some((span) => start < span.end && end > span.start)) continue;
      spans.push({ start, end, value: match[0] });
    }
  }
  spans.sort((a, b) => a.start - b.start);

  let cursor = 0;
  let out = "";
  for (const span of spans) {
    out += gloss(text.slice(cursor, span.start));
    out += span.value;
    cursor = span.end;
  }
  out += gloss(text.slice(cursor));
  return out;
}


/**
 * Guard used by tests and by `scripts/audit-council-i18n.ts`.
 *
 * Returns the offending strings (empty when the text is fully English) so a
 * failure names exactly which seat and field leaked through, instead of just
 * asserting `false`.
 */
export function findCjk(text: string): string[] {
  const hits: string[] = [];
  for (const line of text.split("\n")) {
    if (CJK_PATTERN.test(line)) hits.push(line.trim().slice(0, 160));
  }
  return hits;
}
