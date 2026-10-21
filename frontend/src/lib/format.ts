/**
 * Number formatting.
 *
 * One rule: **formatting never invents precision.** A value is shown at the
 * precision the backend published it at, and a value of `null` renders as an
 * em dash — never as `0`, which would be a fabricated measurement. Where a
 * derived quantity is computed in the UI (a Δ, a ratio), the derivation is
 * named in the call site so a reader can check it against the two inputs.
 */

const EM_DASH = "—";

/** `8.58` -> `"8.580"`; `null` -> `"—"`. Trailing zeros are not trimmed, so a
 * column of binding values stays optically aligned. */
export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    useGrouping: false,
  });
}

/** Fixed precision without a thousands separator, for axis ticks. */
export function tick(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : EM_DASH;
}

/** Always shows the sign — for Δ columns and paired deltas. */
export function signed(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  const text = Math.abs(value).toFixed(digits);
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${text}`;
}

/** Integers with a thousands separator. */
export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  return Math.round(value).toLocaleString("en-US");
}

/**
 * A p-value: `0.77`, `0.086`, `2.4e-10`.
 *
 * The switch to exponential at `< 1e-4` is deliberate. `p = 0.0000` would read
 * as "exactly zero", which a finite-sample t-test never produces; the exponent
 * form keeps the magnitude and stops the reader from rounding it away.
 */
export function pvalue(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return EM_DASH;
  if (p === 0) return "0";
  if (p >= 1e-4) return p >= 0.1 ? p.toFixed(2) : p.toFixed(3);
  const [mantissa, exponent] = p.toExponential(1).split("e");
  const exp = Number(exponent);
  return `${mantissa}×10${superscript(exp)}`;
}

/** `2.4e-10` -> `2.4×10⁻¹⁰` for prose; ASCII exponent for `mono` tooltips. */
function superscript(exponent: number): string {
  const glyphs: Record<string, string> = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
  };
  return String(exponent)
    .split("")
    .map((char) => glyphs[char] ?? char)
    .join("");
}

/** Machine-readable p-value for `title=` tooltips and copy/paste. */
export function pExact(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return EM_DASH;
  return p.toExponential(3);
}

/** A fraction in `[0, 1]` as a percentage, e.g. coverage. */
export function percent(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return EM_DASH;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** A number already expressed in percent units, e.g. `coverage_pct = 16.5344`. */
export function percentOf100(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  return `${value.toFixed(digits)}%`;
}

/** US dollars, rounded to whole dollars — used for wet-lab cost claims. */
export function usd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  return Math.round(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

/** `m5_hybrid` -> `M5 · Virtual Lab Hybrid`; `m3_ucb` -> `M3 · Standard UCB`. */
export const METHOD_META: Record<
  string,
  { short: string; label: string; color: string; description: string }
> = {
  m1_random: {
    short: "M1",
    label: "Random Search",
    color: "#64748B",
    description: "Uniform draw from the qualified pool. The no-model floor.",
  },
  m2_greedy: {
    short: "M2",
    label: "Greedy Ensemble",
    color: "#F59E0B",
    description: "Dual-surrogate mean only (β₀ = 0): exploit μ, never σ.",
  },
  m3_ucb: {
    short: "M3",
    label: "Standard UCB",
    color: "#6366F1",
    description: "Joint UCB (A_base). The primary active-learning competitor and the H0 arm.",
  },
  m4_fixed_prior: {
    short: "M4",
    label: "Fixed Structural Prior",
    color: "#A855F7",
    description: "M3 plus a static wild-type anchor bonus on positions 3, 11, 12, 13.",
  },
  m5_hybrid: {
    short: "M5",
    label: "Virtual Lab Hybrid",
    color: "#10B981",
    description:
      "Deterministic tool arsenal + bounded agent priors + the 5-slot falsification quota.",
  },
  m5_warm_start: {
    short: "M5ᵂ",
    label: "M5 warm start",
    color: "#06B6D4",
    description: "M5 with Round 0 chosen by the cold-start prior instead of five random draws.",
  },
  live_m5: {
    short: "LAB",
    label: "Live campaign",
    color: "#06B6D4",
    description: "The recorded seed-21 campaign, shown on its own seed.",
  },
};

export function methodMeta(key: string): { short: string; label: string; color: string; description: string } {
  return (
    METHOD_META[key] ?? {
      short: key.slice(0, 3).toUpperCase(),
      label: key,
      color: "#94A3B8",
      description: "",
    }
  );
}

/**
 * The slot-role tokens, keyed by the contract's `slot_type`.
 *
 * The plate derives its badge from `slot_type` **plus** `slot_graduation` (see
 * `slotRoles` in `derive.ts`), because a slot whose rule was "Max Joint
 * Uncertainty" is a pure exploit pick once the harvest phase begins. This table is
 * the colour/role half of that mapping, kept here so every panel that names a slot
 * rule uses the same three strings.
 */
export const SLOT_META: Record<string, { label: string; color: string; role: string }> = {
  "Dual-Surrogate UCB": { label: "Dual-Surrogate UCB", color: "#10B981", role: "Exploit" },
  "Max Joint Uncertainty": { label: "Max Joint Uncertainty", color: "#0EA5E9", role: "VoI Probe" },
  "Agent Hypothesis Test": { label: "Agent Hypothesis Test", color: "#10B981", role: "Exploit" },
  "Skeptic Falsification Probe": {
    label: "Skeptic Falsification Probe",
    color: "#F59E0B",
    role: "H0 Control",
  },
};

/** The alignment palette, exported so the grid and its legend cannot diverge. */
export const ANCHOR_COLOR = "#22D3EE";
export const MUTATION_COLOR = "#34D399";
export const FALSIFIED_COLOR = "#F59E0B";
