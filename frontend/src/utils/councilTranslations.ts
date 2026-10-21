/**
 * English renderings of the seven Chinese council memos in the Seed-21 ledger.
 *
 * ## Why this file exists
 *
 * The recorded golden campaign used live frontier models with multilingual
 * behaviour. Two seats — the Developability Scientist and the Scientific Skeptic,
 * both Anthropic seats — answered in Chinese on five of the six rounds. The
 * science is sound; only the language is wrong for a publication console. The
 * ledger is frozen and must not be edited, so the translation happens **here, in
 * the presentation layer**, keyed by a hash of the original string.
 *
 * ## The rules these translations follow
 *
 * 1. **Nothing is added.** Every number, candidate id, position index, sequence,
 *    threshold and table row in the English text appears in the Chinese source.
 *    Where a claim needed a hedge the original did not have, the hedge was left
 *    out.
 * 2. **Nothing is dropped**, except where the ledger itself is truncated: three of
 *    the seven memos were cut mid-sentence by the model's completion ceiling.
 *    Those are marked in place rather than completed by inference — inventing the
 *    missing clause would be worse than showing the cut.
 * 3. **Structure is preserved.** Headings, tables, bullet lists and code spans map
 *    one-to-one, so the rendered document has the same reading order as the raw
 *    one. `markdown.ts` parses both through the same paths.
 * 4. **Terminology is the repository's.** `max_5mer_hydropathy`,
 *    `aromatic_fraction`, `net_charge_abs_fraction`, `decisive`, `SUPPORTED` and
 *    the rest are kept verbatim — they are contract field names, not prose.
 *
 * ## Provenance
 *
 * Each key is the FNV-1a hash of the raw string. The raw text lives in
 * `council-zh-source.json` (extracted from `results/golden_seed21.sqlite3`,
 * read-only, by `scripts/extract-council-zh.ts`) and is what the "Raw ledger"
 * toggle shows. `npm run verify` asserts that every CJK string the payload can
 * serve has an entry here.
 */
export interface CouncilTranslation {
  /** Where and by whom the original was said. */
  round: number;
  role: string;
  /** The English rendering. Markdown, same structure as the source. */
  text: string;
}

export const COUNCIL_TRANSLATIONS: Record<string, CouncilTranslation> = {
  // ───────────── R1 · Developability Scientist · 832 chars ─────────────
  "7f112483": {
    round: 1,
    role: "Developability Scientist",
    text: `## Bottom line first

Across the 15 13-mer peptides, discrimination comes almost entirely from the **hydrophobic window** and the **aromatic fraction**. Charge clusters, unpaired cysteines and deamidation sites are zero throughout the set and therefore carry no selection weight in this batch.

Three risk tiers:

| Tier | Candidates | Discriminating evidence |
|---|---|---|
| Recommended to advance | CAND-015, CAND-318, CAND-164 | No flagged positions; max 5-mer hydropathy ≤ 0.56; aromatic fraction low to mid |
| Advance with targeted monitoring | CAND-219, CAND-292, CAND-325, CAND-208, CAND-307, CAND-202, CAND-096, CAND-166 | Exactly one clear liability each: net charge near zero, Met/Trp oxidation, acidic C-terminal instability, or an elevated aromatic fraction |
| High risk, recommend down-weighting | CAND-247, CAND-051, CAND-070, CAND-227 | max 5-mer hydropathy ≥ 1.10 together with aromatic fraction 0.23–0.31; CAND-051 additionally carries a DS isomerization site |

## How the call is made

**1. Hydrophobicity is the primary axis.** The readings separate cleanly into two groups — ≤ 0.56 (015, 318, 164, 219, 292, 208, 202, 096, 166, 307) and ≥ 0.92 (325, 227, 070, 051, 247) — with an obvious gap between them. The five in the upper group also carry elevated aromatic fractions and show the classic aromatic–hydrophobic aggregation motif (several contain \`Y-x-F\` or \`Y-G-x\` patterns); this is not an isolated signal. CAND-247 (1.32, sequence \`AGSVAEGWYHYRD\`) is the worst combination in the set: the widest hydrophobic window, with Trp and both Tyr residues concentrated in the C-terminal half.

**2. Candidates at zero net charge lack the electrostatic repulsion that stabilises a colloid.** The four with \`net_charge_abs_fraction = 0\` (CAND-070, CAND-166, CAND-202, CAND-

*[The ledger entry ends mid-list here: the model hit its completion ceiling. The rendering is truncated at the same point rather than completed by inference.]*`,
  },

  // ──────────── R2 · Scientific Skeptic · 2464 chars, complete ────────────
  "8999cdcd": {
    round: 2,
    role: "Scientific Skeptic",
    text: `## Conclusion first

I do not recommend maintaining \`SUPPORTED\`. On the numbers in the same evidence package, I recommend **downgrading HYP-R1-01 to \`INDETERMINATE\` (provisional)** until the items in §5 are supplied.

The core claim in one sentence: **the 0.68 paired lift is smaller than the uncertainty scale this batch itself reports, and 100% of that lift comes from a single point that deviates from expectation by 2.9× the tolerance and has no replicate.**

---

## 1. The decision threshold contradicts the noise scale the same toolkit reports (blocking)

- \`decisive\` threshold: **0.10**
- \`residual_tolerance\` in the same record: **0.31**
- Observed range of \`sigma_joint\` in \`inspect_joint_uncertainty_hotspots\`: **0.726 – 0.917**

That is: the bar for calling a result "decisive" is set below one third of the noise this pipeline itself admits to (0.31), and far below the observed joint uncertainty. A lift of 0.68 falls inside roughly **1σ**. Using 0.10 as the decisive line means the resulting \`decisive_lift: true\` is a consequence of threshold choice, not a statistical finding.

**Required**: \`sigma_bind / sigma_dev / sigma_joint\` for \`CAND-010\` and \`CAND-094\` individually. Neither ID appears in the hotspot table, which lists only the top-15 most uncertain candidates — a biased sample that cannot be read as the population distribution.

## 2. The H1 target overshoots the residual tolerance by ~2.9×, and it supplies the entire lift (blocking)

| Quantity | Value |
|---|---|
| \`expected_binding\` (CAND-010) | 7.616 |
| \`h1_binding\` (CAND-010) | 8.524 |
| residual | **0.908 ≈ 2.9 × 0.31** |
| \`h0_residual\` | 0.260 (0.84 × tolerance — passes narrowly) |

Back-solving \`h1_arm_mean_binding = 8.063 = mean(8.524, x)\` gives **CAND-154 ≈ 7.602**, and with \`h1_mean_residual = 0.467\` that puts CAND-154's residual at ≈ **0.026** — essentially exactly on its expectation.

In other words: within one H1 arm, two candidates differ by ≈ **0.92**; one sits on expectation and one far above it. If they genuinely share a background, this looks like a single-point outlier rather than a consistent treatment effect. At the same time \`h1_mean_residual 0.467 > residual_tolerance 0.31\` — **the arm fails its own tolerance criterion and was still recorded as SUPPORTED.** "overriding residual doubt" in the reason field is wording, not a resolution.

## 3. The model's expectation points against the hypothesis, so there is no second independent evidence source

\`expected_binding = 7.616\` is below the measured control at \`7.845\` (by 0.23). The pre-query model therefore judged CAND-010 to be **worse**. All evidence in favour of the hypothesis reduces to this one raw measurement.

The model may well be wrong; that is not the problem. The problem is that once this measurement is absorbed by residual or position effects, the hypothesis has **no remaining support**. \`SUPPORTED\` requires an independent source beyond the model, and there is none here.

## 4. The position effect was never actually tested

The design is **slot 4 vs slot 5** — adjacent positions, the textbook position-confounding scenario. Yet the record carries \`position_residuals: {}\` and \`inverted_positions: []\`, both empty, so **"checked and found nothing" cannot be distinguished from "never computed".** Writing "overriding residual doubt" on top of an empty position diagnostic is a conclusion stronger than its evidence.

## 5. Secondary issues

- The control's \`h0_residual = 0.26\` consumes 84% of the tolerance, so the control is not clean either. Both arms deviate in the same positive direction (+0.26 / +0.91), which suggests a systematic uplift (plate, batch or readout offset).
- Everything is **n = 1**: no replicates, no error bars. Supporting a 0.68 difference on a single measurement does not hold under any threshold setting.
- \`h1_arm_ids\` contains \`CAND-154\`, but \`graded_on\` lists only \`CAND-010\` and \`CAND-094\` — the arm statistic and the graded statistic are computed over different objects, so the arm mean of 8.063 must not be used to support a verdict on 8.524.

---

## Minimum evidence package (ordered by priority)

1. **\`sigma_joint\` for CAND-010 and CAND-094** — this alone determines whether 0.68 is meaningful.
2. **At least 2 independent replicates of CAND-010** (different plate or different slot), reporting mean and dispersion; repeat CAND-094 in the same batch as the control.
3. **A non-empty per-position residual table for slot 4 / slot 5 on the same plate**, to exclude a position effect.
4. **The construction of CAND-154**: same background as CAND-010, or an intermediate? Explain the 0.92 divergence within the arm.
5. If the lift survives at ≥ 0.68 with dispersion < 0.31, \`SUPPORTED\` can be restored; if it shrinks to ≤ 0.31 or to the order of σ, it must be recorded as \`FALSIFIED\` or \`INDETERMINATE\`.

## Current verdict

**INDETERMINATE (provisional).** Grounds: the 0.68 lift is under the joint uncertainty scale this batch reports; the target deviates from expectation by 2.9× tolerance; there are no replicates; the position effect is untested. Until items 1–3 are supplied, the word "decisive" should not appear in this record.`,
  },

  // ──────────── R3 · Developability Scientist · 2361 chars ────────────
  a049f551: {
    round: 3,
    role: "Developability Scientist",
    text: `## Developability grading of 15 candidates

Grading below uses deterministic sequence metrics only — no pI, no experimental data, no forced-degradation results — so confidence in the conclusions is limited.

### Decision rules (auditable)
| Metric | Threshold | Meaning |
|---|---|---|
| \`max_5mer_hydropathy\` | > 1.0 / > 1.8 | aggregation risk / high risk |
| \`aromatic_fraction\` | > 0.35 | aromatic residue clustering (strong hydrophobic patch) |
| \`net_charge_abs_fraction\` | < 0.02 | net charge near zero, close to pI, poor solubility at neutral pH |
| \`isomerization_motifs\` | any | DS/DG proceed through a succinimide intermediate to iso-Asp, producing charge heterogeneity |
| \`adjacent_charge_clusters\` | ≥ 1 | charge patch (viscosity, potential immunogenicity) |

Net charge is a **net sum**, not a sum of absolute values — for example CAND-208's R cancels against two D residues to give −1/13. A value of "0.0" therefore means the sequence is near-neutral, not that it carries no charge.

### Grading table
| Tier | Candidate | Sequence | max 5-mer | aromatic | net charge | Key liability |
|---|---|---|---|---|---|---|
| **A — advance directly** | CAND-015 | ANDIYIQGYDLNR | 0.56 | 0.15 | 0.077 | none (best-balanced in the batch) |
| | CAND-208 | ARYPGGGIYDFDP | 0.42 | 0.23 | 0.077 | none |
| | CAND-307 | ASYYAPGSYGFDP | 0.04 | 0.31 | 0.077 | none |
| | CAND-270 | TRYSRSYYYGLDP | −0.10 | 0.31 | 0.077 | none (mild YYY stacking; monitor) |
| **B — one moderate liability** | CAND-219 | AGSSASGWYHYTT | 0.32 | 0.23 | 0.008 | net charge near zero → solubility/pI risk |
| | CAND-166 | SIWERTGQYSYTS | −0.98 | 0.23 | 0.000 | zero net charge → as above |
| | CAND-070 | TPYDVLGSYYFRV | 1.10 | 0.31 | 0.000 | high hydrophobicity + zero net charge (compounding) |
| | CAND-246 | AKFHWGEDWLSDN | −0.68 | 0.23 | 0.146 | one charge cluster; uneven charge distribution, monitor viscosity |
| **C — needs close attention** | CAND-227 | ASVDLGGFYDFRY | 1.10 | 0.31 | 0.077 | aggregation risk |
| | CAND-081 | TRYYYGLYYYMDV | 0.36 | **0.46** | 0.000 | high aromatic density with zero net charge (\`YYY…YYY\` + M; Met oxidation also needs attention) |
| | CAND-172 | TRYYGDGYYDFDY | −0.74 | **0.46** | 0.154 | high aromatic density plus a **DG isomerization site** |
| | CAND-247 | AGSVAEGWYHYRD | **1.32** | 0.23 | 0.069 | high hydrophobicity |
| | CAND-051 | TSIFSGGYYGLDS | 1.14 | 0.23 | 0.077 | high hydrophobicity plus a **DS isomerization site** |
| | CAND-366 | ARDYDFGLSVGAV | **1.92** | 0.15 | 0.077 | strong hydrophobic patch, high aggregation risk |
| | CAND-286 | VLSRIVVSYFYDY | **2.16** | 0.31 | 0.000 | highest hydrophobicity in the batch with zero net charge (compounded risk) |

### Recommendations
- **Advance first**: CAND-015, CAND-208, CAND-307. None of the three quantified metrics is problematic for any of them; take them into expression/purification first.
- **Reserve**: CAND-270. CAND-219 / CAND-166 need pI and low-concentration solubility measured before a keep/drop decision.
- **Down-weight or run targeted assays**:
  - CAND-286, CAND-366: screen buffer conditions by SEC-HPLC and DLS first; if that does not resolve them, drop them.
  - CAND-051, CAND-172: DS/DG are genuine chemical-degradation sites. A sequence score cannot exclude the risk, so forced degradation (40 °C / pH 7–9) combined with IEC and peptide mapping is required to establish the iso-Asp formation rate. If either site sits in the CDR binding interface, the risk is unacceptable.
  - CAND-081: use HIC and DLS to determine whether the aromatic cluster drives self-association.

### Blind spots in this tool (need supplementing)
1. **The conditions used for net charge are unknown** (pH 7? pI?), and every "near-zero net charge" call depends heavily on that assumption.
2. There is no explicit score for **Met/Trp oxidation**, **N-glycosylation sequons (NxS/T)**, or **Asn deamidation variants beyond NG/NS**; CAND-081 carries an M adjacent to Y, an unscored position.
3. pI, actual expression titre, Tm and HIC retention time are all missing, so even tier A cannot be called low-risk.

If useful, I can export this table as CSV, or re-rank it under your own thresholds (for example moving the aggregation line from 1.0 to another value).`,
  },

  // ──────────── R4 · Developability Scientist · 1746 chars ────────────
  ed182dd1: {
    round: 4,
    role: "Developability Scientist",
    text: `## Developability risk assessment (15 candidates)

Scored against deterministic sequence rules; no structural, expression or stability work was done. The thresholds are my working assumptions, not a platform standard.

**Flag rules applied**
- Aggregation risk: \`max_5mer_hydropathy\` ≥ 1.0 high, 0.7–1.0 moderate
- Aromatic fraction ≥ 0.30 → elevated π-stacking / expression burden
- \`deamidation_motifs\` / \`isomerization_motifs\` non-empty → chemical degradation site
- \`adjacent_charge_clusters\` ≥ 1 → charge cluster
- \`unpaired_cys\` > 0 → disulfide mismatch

### Tiering

| Tier | Candidate | Key metrics | Basis |
|---|---|---|---|
| **A — clean** | CAND-302 | hydro −0.24, no motif | most hydrophilic, lowest aggregation risk |
| | CAND-400 | 0.50 / 0.231 / 0.077 | no flag on any axis |
| | CAND-367 | 0.16 / 0.308 / 0.0 | aromatic slightly high but hydrophobic window low |
| | CAND-415 | 0.60 / 0.231 / 0.0 | no flag on any axis, balanced net charge |
| **B — single moderate flag** | CAND-333 | 0.56 / **0.308** | aromatic fraction elevated |
| | CAND-393 | 0.70 / 0.231 | hydro at the boundary |
| | CAND-378 | 0.74 / **NG** | Asn deamidation site |
| | CAND-329 | 0.68 / **0.385** | highest aromatic fraction in the set |
| | CAND-271 | **0.92** / 0.154 | hydrophobic window near the ceiling |
| | CAND-246 | −0.68 / **charge cluster = 1** | charge cluster, otherwise hydrophilic |
| **C — multiple flags** | CAND-355 | **1.08** + **DG** | aggregation + Asp isomerization |
| | CAND-279 | **1.04** + **DS** + **0.308** | three compounding flags |
| | CAND-051 | **1.14** + **DS** | hydrophobicity-dominated |
| **D — high risk** | CAND-366 | **1.92** | second-highest aggregation risk, no compensating term |
| | CAND-196 | **2.30** | highest hydrophobic window in the set |

### Judgement

- Advance tier A first, **CAND-302** and **CAND-415** in particular: the former is the most hydrophilic, and the latter sits comfortably on both net charge and hydrophobicity.
- Tier B can serve as reserve, but each single flag needs a targeted check: CAND-378 needs the actual NG deamidation rate confirmed (sequence context sets its severity), and CAND-246 needs a look at whether its charge cluster affects expression or purification.
- For tiers C and D, resolving aggregation risk through sequence engineering rather than process optimisation is the better trade — the 5-mer hydrophobic windows of CAND-196 and CAND-366 (2.30 / 1.92) are far beyond any commonly accepted range, so hydrophobic residue substitution or outright removal is the recommendation.

### Blind spots in this scoring (need measurement)

The panel does not cover the following positions, so several candidates carry more real risk than the table shows:

- **Met oxidation**: CAND-302, CAND-355 and CAND-271 each contain one Met, and CAND-302's Met sits immediately before the C-terminal region — oxidation there could matter, and it is the top of tier A, so this should be looked at separately.
- **Trp oxidation**: CAND-302, CAND-246 and CAND-355 contain Trp.
- **N-terminal pyroglutamate**: several sequences begin with S/A/T and need confirmation.
- **Asn deamidation**: only NG is reported; NS/NH are not. CAND-378's \`SNGPY\` needs manual review as a potential sequon.
- No expression titre, Tm or aggregation-kinetics data exist at all; \`fallback\` correctly limits confidence.

Suggested next step: take 3–4 tier-A candidates into expression plus SEC monomer content and accelerated stability, and in parallel run forced Met oxidation on CAND-302.`,
  },

  // ────── R4 · Scientific Skeptic · 720 chars, truncated in ledger ──────
  "89646d3c": {
    round: 4,
    role: "Scientific Skeptic",
    text: `## Review outcome: none of the 3 verdicts survives as written

| ID | Tool verdict | My verdict | Core basis |
|---|---|---|---|
| HYP-R1-01 | SUPPORTED (+0.680) | **withdraw → INCONCLUSIVE** | H1 arm residual +0.908 ≈ 2.9× tolerance; the control arm only "passes" under a tolerance 6.2× wider than in the other two rounds |
| HYP-R2-01 | FALSIFIED (−0.540) | **downgrade → INCONCLUSIVE (but weak counter-evidence against R1)** | the target-arm residual of −0.595 is numerically identical to the position-6 residual; candidate effect and position effect are perfectly confounded |
| HYP-R3-01 | SUPPORTED (+0.449) | **withdraw → INCONCLUSIVE** | 74% of the lift comes from a negative deviation in the control arm (−0.334 ≈ 6.7× this round's tolerance), not from an advantage in the target arm |

---

## 1. Structural problem: H0 is effectively the within-round median, not a matched control

| Round | round_median | h0_binding | difference |
|---|---|---|---|
| R1 | 7.843 | 7.845 | +0.002 |
| R2 | 7.921 | 7.921 | 0.000 |
| R3 | 7.509 | 7.509 | 0.000 |

In all three rounds H0 is numerically identical to the within-round median. That makes the control a "regression to the mean" comparison object, while the H1 arm is a selected extreme. Any lift measured this way necessarily contains a selection effect (winner's curse), and cannot separate "the substitution itself works" from "we picked a

*[The ledger entry ends mid-sentence here — the model hit its completion ceiling. The rendering stops at the same point rather than completing the argument by inference.]*`,
  },

  // ──────────── R6 · Scientific Skeptic · 3744 chars, complete ────────────
  "61708950": {
    round: 6,
    role: "Scientific Skeptic",
    text: `## Audit conclusion

**None of the five verdicts meets an acceptable evidentiary standard.** I recommend downgrading all four \`SUPPORTED\` and the single \`FALSIFIED\` to \`INCONCLUSIVE\`. The reason is not that "the evidence is not strong enough" but that **the current design cannot separate the substitution effect from the plate-position effect** — an identifiability problem, not a sample-size problem.

---

## 1. Fatal issue: design and position are perfectly confounded

All five \`design\` fields read \`"claimed pair (planner slot 4 vs slot 5)"\`:

| Round | H1 target position | H0 control position |
|---|---|---|
| R1–R5 | slot 4 (fixed) | slot 5 (fixed) |

The H1 target **never** appears in slot 5 and the H0 control **never** appears in slot 4. "Measured higher" is therefore mathematically equivalent to "slot 4 read higher than slot 5"; the two cannot be separated.

The dataset itself supplies the scale of the position effect: R2's \`position_residuals\` are \`{-0.247, +0.10, -0.595, +0.094, -0.251, -0.134}\`, standard deviation **≈ 0.26**, range **0.69**. The \`paired_lift\` values of the four \`SUPPORTED\` rounds are 0.680 / 0.449 / 0.524 / 0.257 — **every one of them falls inside the plate-position residual range observed within a single round.**

What supports this reading is sign consistency: slot 4 won all four decisive rounds, and the probability of one sign occurring four times is 2 × (1/2)⁴ = 0.125. That looks more like measuring **a systematic advantage of slot 4** than five unrelated substitution combinations each happening to be better.

---

## 2. The "decisive" threshold barely filters anything

The bar is hard-coded at \`>= 0.10\`, while the instrument's own reported noise is σ ≈ 0.26:

\`\`\`
Z = 0.10 / 0.26 ≈ 0.38
P(|lift| > 0.10 | pure noise) ≈ 0.70
\`\`\`

**Under pure noise, roughly 70% of pairwise comparisons would be stamped "decisive".** The expected number of "decisive" outcomes over five comparisons is 3.5 and 4 were observed — a binomial test gives p ≈ 0.53. This threshold has no discriminating power, and using it as the basis for "overriding residual doubt" is circular: it lets a bar smaller than the noise "override" the noise.

---

## 3. In most rounds the residual magnitude exceeds the entire effect size

\`paired_lift\` only means something against the position residuals of **the same round**:

| Round | paired_lift | Key residual | Does the residual swallow the effect? |
|---|---|---|---|
| R1 | +0.680 | h1 +0.467 | No — but \`residual_tolerance=0.31\` > the 0.10 threshold, so the criterion contradicts itself |
| R2 | −0.540 | position 6 \`−0.595\` | **Yes** (residual > effect) |
| R3 | +0.449 | h1 −0.378 / h0 −0.334 | marginal |
| R4 | +0.524 | h0 **−0.771** | **Yes** (the control was pushed down 0.77, exceeding the 0.524 difference) |
| R5 | +0.257 | h0 **−0.372** | **Yes** |

R4 is the clearest: the H1 target (7.534) and the H0 control (7.010) **both** fall below \`expected_binding\` 7.745, with residuals of −0.177 and −0.771 respectively. The "win" comes entirely from the control falling further, not from the target improving.

R1's \`residual_tolerance=0.31\` differs sixfold from the \`0.05\` used in R2–R5, with no explanation — and 0.31 is already larger than the 0.10 significance threshold. **The tolerated noise exceeds the effect threshold**; the criterion is internally inconsistent.

---

## 4. Data-integrity problems (blocking)

**1. HYP-R5-01's target is not in its own arm**

\`\`\`
h1_arm_ids      = ["CAND-086"]
h1_target_id    = "CAND-141"     ← not in h1_arm_ids
h1_binding      = 7.655  <  h1_arm_mean_binding = 7.873
\`\`\`

The graded candidate does not belong to the experimental arm it declares. That record should produce no verdict at all until it is repaired.

**2. The target is systematically worse than its own arm mean**

R2: target 7.381 vs arm mean 7.866 (**the target is the weakest member**, yet was registered as the arm's representative and stamped FALSIFIED)
R4: 7.534 vs 7.592; R5: 7.655 vs 7.873

If arm members disagree in direction with the target's conclusion, the effect is not robust; grading a single target then amounts to picking a sample within the arm.

**3. \`falsified_positions: [3, 6, 10]\` is not independent evidence**

Those three are exactly the largest absolute residuals (−0.595 / −0.251 / −0.247); the rest (+0.10 / +0.094 / −0.134) are small. The field is a rename of "large residual" and cannot serve as a conclusion about position reliability — and it comes from a single round with n = 1.

**4. The residual definition is not reproducible**

Recomputing as \`binding − expected_binding\`: R4 h1 should be +0.211 but is reported as −0.177; R5 h1 should be −0.200 but is reported as −0.074; R1 h0 should be +0.229 but is reported as +0.26. No single residual definition reproduces these numbers across rounds, and R1's sign convention is opposite to R2–R5. **The audit chain cannot be re-verified.**

---

## 5. The uncertainty tool answers a question nobody asked

The 15 high-uncertainty candidates returned by \`inspect_joint_uncertainty_hotspots\` (CAND-333/303/329/…) have an **empty intersection** with all 10 graded candidates (CAND-010/094/213/074/069/177/172/278/141/209). That is: for precisely the comparisons called "decisive", \`sigma_joint\` is completely unknown in this output.

Separately, \`CAND-333\`'s \`sigma_bind\` and \`CAND-313\`'s \`sigma_dev\` are exactly \`1.0\` and sit at the top of the ranking, which looks like a **clipping/normalisation ceiling** rather than a genuine posterior standard deviation. If uncertainties of order 1.0 really exist, then differences of 0.26–0.68 are globally indistinguishable.

---

## 6. Revised verdicts

| hypothesis | Original verdict | Review | Basis |
|---|---|---|---|
| HYP-R1-01 | SUPPORTED | **INCONCLUSIVE** | slot confounding; tolerance 0.31 > threshold 0.10 |
| HYP-R2-01 | FALSIFIED | **INCONCLUSIVE** | a negative result can equally be manufactured by position 6's −0.595 |
| HYP-R3-01 | SUPPORTED | **INCONCLUSIVE** | residual −0.378 is the same order as the lift 0.449 |
| HYP-R4-01 | SUPPORTED | **INCONCLUSIVE** | control residual −0.771 > lift 0.524; both sides below expectation |
| HYP-R5-01 | SUPPORTED | **INVALID** | target not in the arm; control residual −0.372 > lift 0.257 |

Consistency demands: since the plate-position effect can manufacture +0.68, it can equally manufacture −0.54. Keeping R2's FALSIFIED while rejecting the other four is not self-consistent.

---

## 7. Minimum design that would be decidable

1. **Reciprocal pairs**: run the same candidate pair in both (4,5) and (5,4) arrangements and average the two differences. This cancels the slot offset directly; it doubles cost and is the only reliable option.
2. **Position randomisation plus covariate**: enter position as a fixed effect in the model and report \`lift | position\` rather than a raw difference.
3. **Reset the threshold**: move the \`decisive\` bar to ≥ 2σ (≥ 0.52 at the current σ ≈ 0.26), or switch to a paired t-test with a confidence interval. Fix \`residual_tolerance\` as well, and require it to be **below** the significance threshold.
4. **Repair the R5 record**: align \`h1_target_id\` with \`h1_arm_ids\` and re-run, otherwise exclude the round.
5. **Close the uncertainty coverage gap**: compute \`sigma_joint\` for the 10 candidates actually graded, and state whether 1.0 is a ceiling value.

Until items 1 and 3 are done, this audit endorses no \`SUPPORTED\` verdict. The five rounds support exactly one observation — **"slot 4 read higher than slot 5 in four of five rounds"** — and nothing at all about substitution combinations.`,
  },

  // ──────────── R6 · Developability Scientist · 1943 chars ────────────
  "896fdcd8": {
    round: 6,
    role: "Developability Scientist",
    text: `## Developability risk assessment (15 candidates, 13-mer)

The tool supplied metric values only, with no decision thresholds. The tiering below is a rule I set explicitly — please confirm it matches your gate before it is used:

- **High**: \`max_5mer_hydropathy ≥ 0.7\`, or \`hydropathy ≥ 0.6 with aromatic_fraction ≥ 0.3\`, or any motif hit
- **Watch**: \`hydropathy 0.5–0.7\` or \`aromatic_fraction ≥ 0.3\`
- **Low**: everything else

### Tiering result

| Tier | Candidate | Key readings | Principal risk |
|---|---|---|---|
| High | **CAND-355** | hyd 1.08 / iso motif \`DG\` | hydrophobicity compounded with isomerization; highest risk |
| High | **CAND-271** | hyd 0.92 | hydrophobic patch (C-terminal \`DMLF\`) |
| High | **CAND-313** | hyd 0.82 | hydrophobic patch |
| High | **CAND-378** | hyd 0.74 + deam motif \`NG\` | hydrophobicity compounded with deamidation |
| High | **CAND-292** | hyd 0.72 | hydrophobic patch |
| High | **CAND-329** | hyd 0.68 / arom 0.385 | hydrophobicity compounded with high aromaticity |
| Watch | CAND-415 | hyd 0.60 | hydrophobic; aromaticity acceptable |
| Watch | CAND-405 | hyd 0.50 / arom 0.385 | aromaticity elevated |
| Watch | CAND-333 | hyd 0.56 / arom 0.308 | aromaticity with moderate hydrophobicity |
| Watch | CAND-400 | hyd 0.50 | moderate hydrophobicity |
| Watch | CAND-249 | arom 0.308 | aromaticity |
| Watch | CAND-384 | arom 0.385 | aromaticity with low hydrophobicity (0.24) |
| Watch | CAND-367 | arom 0.308 | aromaticity with near-lowest hydrophobicity (0.16) |
| Low | **CAND-303** | hyd 0.32 / arom 0.231 / no motif | all three metrics moderate or low |
| Low | **CAND-302** | hyd −0.24 / arom 0.231 | lowest hydrophobicity |

### Shared findings

- Across all 15 candidates \`adjacent_charge_clusters = 0\`, \`unpaired_cys = 0\` and \`net_charge_abs_fraction ≤ 0.085\`, so **charge clustering and free cysteine contribute no discrimination in this batch**; the decision is driven entirely by hydrophobicity, aromaticity and motifs.
- The five with aromaticity ≥ 0.3 (329/384/367/333/405) all carry consecutive \`YSY\`/\`YY\`-type motifs, an obvious signature of homologous engineering; rising aromaticity here tracks Y substitution in the CDR, the classic affinity-versus-aggregation trade-off.

### Two things I need you to confirm first

1. **Threshold provenance**: the 0.5/0.7/0.3 values I used are industry conventions, not tool output. If you have an internal gate (for example 5-mer hydropathy > 1.0 required for high risk), the tiers move substantially: at 1.0, only CAND-355 is high risk.
2. **Whether the sequence backgrounds are comparable**: the C-termini of CAND-271 (\`DMLF\`), CAND-292 (\`FHDM\`) and CAND-378 (\`NGPY\`) differ from the other 12 (\`FDY\`/\`FAY\`). If these are not variants of one scaffold, ranking \`aromatic_fraction\` across entries confounds the comparison and it must be done within scaffold.

### Recommendations

- Advance first: **CAND-303 / CAND-302** — both primary metrics are clean.
- If high aromaticity must be kept for affinity: **CAND-367 / CAND-384** are the compromise (elevated aromaticity, low hydrophobicity) and need measurement (SEC-HIC, DLS, Tm/Tagg).
- Eliminate or re-engineer outright: **CAND-355** (DG) and **CAND-378** (NG) — motif-class risk can be removed by sequence engineering at the lowest cost and with the clearest benefit.

Per the tool's fallback note: the above is evidence selected by deterministic rules only, with no pI, no 3D hydrophobic patches and no experimental data, so **confidence is limited** and it cannot substitute for measurement. Do you want a ranking, an elimination list, or the slate for the next experimental round?`,
  },
};
