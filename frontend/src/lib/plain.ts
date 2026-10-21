/**
 * The plain-language layer.
 *
 * Everything a non-specialist reads on the surface is produced here: the words
 * that name a statistic, and the derivations that turn a published number into
 * something a judge can feel in one second ("+91%", "17× bigger than noise",
 * "less than 1 in a trillion").
 *
 * Three rules this module exists to enforce:
 *
 * 1. **A statistic is never shown without its scale.** `PLAIN[key].scale` travels
 *    with the label, so "0.6578" cannot appear on its own.
 * 2. **The technical token is not the label.** `spearman_rho_at_q35` is a
 *    tooltip, not a heading — `PLAIN[key].token` is where it lives.
 * 3. **A translation never overstates.** `flukeOdds` only ever *understates*
 *    improbability (see the ladder below), and every function here returns
 *    `null` rather than inventing a value for an undefined payload.
 *
 * No result number may be typed here — `scripts/audit-literals.ts` fails the
 * build if one is. Every figure below is computed from an argument.
 */

import { percent } from "./format";

const EM_DASH = "—";

function finite(value: number | null | undefined): value is number {
    return typeof value === "number" && Number.isFinite(value);
}

// ─────────────────────────── the vocabulary ───────────────────────────────

export interface PlainStat {
    /** What the surface calls it. */
    label: string;
    /** The one line that makes the number readable. Always shown next to it. */
    scale: string;
    /** The internal name. Tooltips and provenance only. */
    token: string;
    /** What it measures, in one sentence, for a tooltip. */
    meaning: string;
}

/**
 * Every statistic the console publishes, named for a reader who has never seen
 * a benchmark. Adding a metric to a view means adding an entry here first.
 */
export const PLAIN: Record<string, PlainStat> = {
    spearman_rho: {
        label: "Ranking accuracy",
        scale: "0 = guessing · 1 = perfect",
        token: "spearman_rho_at_q35",
        meaning:
            "How well a method's ranking of the candidate library agrees with the measured binding. 0.5 means it gets the order half-right.",
    },
    aubqc: {
        label: "Best binder found",
        scale: "running average of the best hit so far · higher is better",
        token: "aubqc",
        meaning:
            "Averaged over the whole run, how good the single best candidate found so far was. Once two methods reach the same ceiling this stops separating them.",
    },
    normalized_aubqc: {
        label: "Share of the possible gain",
        scale: "0 = no improvement · 1 = the best candidate in the library",
        token: "normalized_aubqc",
        meaning:
            "The same running average, rescaled between the score you start with and the best candidate that exists.",
    },
    top5_panel_aubqc: {
        label: "Depth of the top 5 leads",
        scale: "average of the five best · higher is better",
        token: "top5_panel_aubqc",
        meaning:
            "How good your five best candidates are on average — a portfolio measure, not a single-hit measure.",
    },
    top_decile_hit_rate: {
        label: "Hit rate in the top 10%",
        scale: "share of your picks that land in the true best 10% · higher is better",
        token: "top_decile_hit_rate",
        meaning:
            "What fraction of the candidates a method picked actually belong to the genuinely best tenth of the library.",
    },
    delta_over_se: {
        label: "Gap vs. run-to-run noise",
        scale: "1× = indistinguishable from luck",
        token: "mean_diff / se",
        meaning:
            "The observed gap divided by how much the result moves between identical reruns. Under 2× is noise; above 5× is a real effect.",
    },
    fold_over_reference: {
        label: "Tighter binding",
        scale: "× = how many times tighter than the starting antibody",
        token: "10 ^ delta_vs_wt",
        meaning:
            "Binding is measured on a log scale, so a +1.0 change means ten times tighter binding. This converts the log difference into a multiplier.",
    },
    library_skipped: {
        label: "Library skipped",
        scale: "share of candidates never tested in the lab",
        token: "1 − query_budget_spent / qualified_count",
        meaning:
            "The lab tests only the candidates the loop asks for. This is the fraction of the qualified library it never had to assay.",
    },
};

/** The label for a statistic key, falling back to the key itself. */
export function plainLabel(key: string): string {
    return PLAIN[key]?.label ?? key;
}

/** The scale line for a statistic key. */
export function plainScale(key: string): string {
    return PLAIN[key]?.scale ?? "";
}

/** The internal name, for a tooltip. */
export function plainToken(key: string): string {
    return PLAIN[key]?.token ?? key;
}

// ─────────────────────────── method names ─────────────────────────────────

/**
 * Method names in words.
 *
 * `M1…M5` is a lookup table the audience will never build, so the surface uses
 * these and the token stays available for tables and provenance.
 */
export const METHOD_WORDS: Record<string, { name: string; full: string; role: string }> = {
    m1_random: { name: "Random", full: "Random picking", role: "The floor: no model at all." },
    m2_greedy: { name: "Greedy", full: "Greedy (always take the top prediction)", role: "Uses the model but never explores." },
    // "Standard UCB" is the name spoken in the pitch and printed on every card,
    // so the surface and the script cannot disagree during a demo.
    m3_ucb: {
        name: "Standard UCB",
        full: "Standard UCB baseline",
        role: "The active-learning baseline every paper in this field compares against.",
    },
    m4_fixed_prior: { name: "Fixed-prior", full: "Fixed-prior variant", role: "The standard baseline plus a hand-written bias." },
    m5_hybrid: { name: "Virtual Lab", full: "Virtual Lab (ours)", role: "This work." },
    m5_warm_start: { name: "Virtual Lab, guided start", full: "Virtual Lab with a guided first batch", role: "Same loop, smarter first five candidates." },
    live_m5: { name: "The recorded run", full: "The recorded run", role: "The single campaign the demo replays." },
};

export function methodName(key: string): string {
    return METHOD_WORDS[key]?.name ?? key;
}

export function methodFull(key: string): string {
    return METHOD_WORDS[key]?.full ?? key;
}

// ─────────────────────────── derivations ──────────────────────────────────

/**
 * Relative improvement of `performer` over `baseline`, as a fraction.
 *
 * `0.6578` against `0.3442` gives `0.9111`, i.e. "+91.1%". The denominator is
 * the baseline, because that is the question a judge asks: *how much better
 * than the thing you are compared against?*
 */
export function relativeGain(
    performer: number | null | undefined,
    baseline: number | null | undefined,
): number | null {
    if (!finite(performer) || !finite(baseline) || baseline === 0) return null;
    return (performer - baseline) / Math.abs(baseline);
}

/** `performer / baseline`, e.g. `1.91`. */
export function ratio(
    performer: number | null | undefined,
    baseline: number | null | undefined,
): number | null {
    if (!finite(performer) || !finite(baseline) || baseline === 0) return null;
    return performer / baseline;
}

/** A signed percentage from a fraction: `0.9111` → `"+91.1%"`. */
export function gainText(gain: number | null | undefined, digits = 1): string {
    if (!finite(gain)) return EM_DASH;
    const magnitude = Math.abs(gain * 100).toFixed(digits);
    const sign = gain > 0 ? "+" : gain < 0 ? "−" : "±";
    return `${sign}${magnitude}%`;
}

/** The same percentage rounded to whole digits, for the hero: `"+91%"`. */
export function gainWhole(gain: number | null | undefined): string {
    return gainText(gain, 0);
}

/** `1.9111` → `"1.91×"`. */
export function ratioText(value: number | null | undefined, digits = 2): string {
    if (!finite(value)) return EM_DASH;
    return `${value.toFixed(digits)}×`;
}

/** A fraction as a whole percentage: `0.8333` → `"83%"`. */
export function shareWhole(fraction: number | null | undefined): string {
    if (!finite(fraction)) return EM_DASH;
    return percent(fraction, 0);
}

/** `0.8333` → `"83.3%"`. */
export function shareText(fraction: number | null | undefined, digits = 1): string {
    if (!finite(fraction)) return EM_DASH;
    return percent(fraction, digits);
}

/**
 * The gap expressed in rerun-noise units.
 *
 * This is the honest replacement for `Δ / SE`: `17` means the gap is seventeen
 * times larger than the spread between two identical runs of the same method.
 */
export function noiseMultiple(
    meanDiff: number | null | undefined,
    se: number | null | undefined,
): number | null {
    if (!finite(meanDiff) || !finite(se) || se === 0) return null;
    return Math.abs(meanDiff) / se;
}

/** `17.36` → `"17×"`. */
export function noiseText(value: number | null | undefined): string {
    if (!finite(value)) return EM_DASH;
    return `${Math.round(value)}×`;
}

/**
 * The p-value as a sentence, on a deliberately **conservative** ladder.
 *
 * Each phrase states an improbability the p-value guarantees and never a
 * stronger one: `p = 8.4e-32` is quoted as "less than 1 in a trillion"
 * although the true odds are nearer 1 in 10³¹. Understating is the only
 * direction that cannot embarrass the project, and the exact value stays in the
 * tooltip via `pExact`.
 */
export function flukeOdds(p: number | null | undefined): string {
    if (!finite(p)) return EM_DASH;
    if (p >= 0.05) return "could easily be chance";
    if (p >= 1e-3) return "unlikely to be chance";
    if (p >= 1e-6) return "less than 1 in 1,000";
    if (p >= 1e-12) return "less than 1 in a million";
    return "less than 1 in a trillion";
}

/** True when the p-value clears the conventional 0.05 bar. */
export function isClear(p: number | null | undefined): boolean {
    return finite(p) && p < 0.05;
}

/**
 * A log-scale binding difference as a fold-change.
 *
 * Binding is reported as `−log₁₀ K_D`, so a difference of `d` means `10^d`
 * times tighter. `0.872` → `7.45`, rendered "7.4×".
 */
export function foldOverReference(delta: number | null | undefined): number | null {
    if (!finite(delta)) return null;
    return Math.pow(10, delta);
}

export function foldText(delta: number | null | undefined, digits = 1): string {
    const fold = foldOverReference(delta);
    if (!finite(fold)) return EM_DASH;
    return `${fold.toFixed(digits)}×`;
}

/**
 * The share of the qualified library the lab never had to assay.
 *
 * `spent` assays against a `qualified` pool: `1 − 35/210` → `0.833`, "83%".
 * Returns `null` rather than a negative number when the budget exceeds the pool
 * (which would mean the payloads disagree, not that the lab skipped −5%).
 */
export function librarySkipped(
    spent: number | null | undefined,
    qualified: number | null | undefined,
): number | null {
    if (!finite(spent) || !finite(qualified) || qualified <= 0) return null;
    const fraction = 1 - spent / qualified;
    if (fraction < 0 || fraction > 1) return null;
    return fraction;
}

/** `3` of `3` as a sentence fragment: `"3 of 3"`; `"—"` when either is missing. */
export function countOf(done: number | null | undefined, total: number | null | undefined): string {
    if (!finite(done) || !finite(total)) return EM_DASH;
    return `${Math.round(done)} of ${Math.round(total)}`;
}
