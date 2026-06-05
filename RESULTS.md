# Experimental Results

A 2×2 benchmark comparing Wachter (instance-specific) and GLANCE (global) counterfactual explanation methods across {Adult 1994, ACS 2018 CA} × {Logistic Regression, Random Forest}, evaluated under the FACTS fairness framework at **n = 1000** instances per cell (balanced 50/50 by sex).

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Design](#experimental-design)
3. [Dataset Characteristics](#dataset-characteristics)
4. [Classifier Performance](#classifier-performance)
5. [Overall Method Comparison (4 cells)](#overall-method-comparison-4-cells)
6. [Fairness Analysis](#fairness-analysis)
7. [FACTS Framework Evaluation (4 dimensions)](#facts-framework-evaluation-4-dimensions)
8. [Featured Finding: Wachter on Tree Ensembles](#featured-finding-wachter-on-tree-ensembles)
9. [Statistical Significance](#statistical-significance)
10. [Visualizations](#visualizations)
11. [Key Findings](#key-findings)
12. [Reproducibility](#reproducibility)
13. [References](#references)

---

## Executive Summary

**Two principal findings:**

1. **The Adult sex disparity is a property of the data, not the linear decision boundary.** Under Logistic Regression GLANCE produces a 36.0 pp sex DP gap on Adult (p < 1e-6); when the classifier is swapped to Random Forest the gap remains at 27.2 pp (p < 1e-6). The hypothesis that linearity drives the gap is refuted. ACS shows uniformly small (≤ 3 pp) non-significant gaps in all 4 cells.

2. **Wachter with finite-difference gradients catastrophically fails on tree ensembles.** Validity collapses from 17–52 % on LR to 0–0.2 % on RF, with sparsity rising to 99.89 % (almost no features changed). GLANCE remains functional on RF because its global rule application decouples from local gradients. This is featured as Section 8 below — a methodological contribution beyond the original fairness study.

### Headline numbers (n = 1000 per cell)

| Cell | Classifier acc / AUC | Wachter validity | GLANCE validity | Sex DP gap (best non-degen. method) |
|---|---|---|---|---|
| Adult / LR | 0.819 / 0.848 | 52.20 % | 37.40 % | **36.00 % (p < 1e-6)** — GLANCE |
| Adult / RF | 0.855 / 0.913 | 0.00 % *(degenerate)* | 20.40 % | **27.20 % (p < 1e-6)** — GLANCE |
| ACS / LR | 0.787 / 0.861 | 17.70 % | 10.80 % | 0.37 % (n.s.) — GLANCE |
| ACS / RF | 0.814 / 0.895 | 0.20 % *(degenerate)* | 21.90 % | 2.15 % (n.s.) — GLANCE |

---

## Experimental Design

| Element | Value |
|---|---|
| Datasets | Adult (UCI 1994), ACS Folktables CA 2018 |
| Classifiers | Logistic Regression (lbfgs, max_iter = 1000), Random Forest (n_estimators = 100, max_depth = 10) |
| Counterfactual methods | Wachter (instance-specific, finite-difference gradient), GLANCE (global K-means over change vectors) |
| Sample size per cell | n = 1000 negatively-predicted test instances |
| Sex balance | 50/50 within each cell |
| Random seed | 42 |
| Fairness framework | FACTS — Equal Burden, Equal Effectiveness, Equal Choice, Equal Cost of Effectiveness |
| Statistical tests | t-test (distance), chi-square (validity, DP gap), 95 % CIs via Wilson interval |

The 2×2 design separates **dataset effects** from **classifier effects** on fairness conclusions — addressing two distinct reviewer concerns about transferability.

---

## Dataset Characteristics

### Adult Income (UCI, 1994)

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Original context | 1994 US Census |
| Train / Test (post-cleaning) | 26,048 / 15,060 |
| Features | 13 (6 continuous, 7 categorical) |
| Target | Income > $50K |
| Positive rate | ~24 % |
| Protected attribute | sex (Male / Female) |

### ACS (Folktables CA, 2018)

| Property | Value |
|---|---|
| Source | American Community Survey, California |
| Original context | 2018 Census Data |
| Train / Val / Test | 125,225 / 31,307 / 39,133 |
| Features | 10 (AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P) |
| Target | Income > $50K |
| Positive rate | 41.06 % |
| Protected attribute | SEX (1 = Male, 2 = Female) |

### Key differences

| Factor | Adult (1994) | ACS (2018) |
|---|---|---|
| Training size | 26 K | 125 K (≈ 5×) |
| Temporal distance | 30 years old | Modern |
| Class balance | 24 % positive | 41 % positive |
| Geographic scope | Nationwide | California only |
| Encoding | Mixed types | Census codes |

---

## Classifier Performance

Trained on each dataset's training split, evaluated on the full held-out test split.

| Dataset | Classifier | Accuracy | ROC-AUC | n_test |
|---|---|---|---|---|
| Adult | Logistic Regression | 0.8191 | 0.8482 | 15,060 |
| Adult | Random Forest | **0.8548** | **0.9132** | 15,060 |
| ACS | Logistic Regression | 0.7866 | 0.8613 | 39,133 |
| ACS | Random Forest | **0.8144** | **0.8953** | 39,133 |

RF beats LR by 3–4 percentage points on accuracy and 3–5 points on AUC on both datasets — RF is the stronger underlying classifier.

---

## Overall Method Comparison (4 cells)

All numbers from `results_{adult,adult_rf,acs,acs_rf}/comparison/comparison_report.txt`.

### Adult / Logistic Regression

| Metric | Wachter | GLANCE | Wachter − GLANCE | Test | p-value | Sig |
|---|---|---|---|---|---|---|
| Validity | 52.20 % | 37.40 % | +14.80 pp | χ² = 43.69 | < 1e-6 | *** |
| Avg distance (L2) | 0.3894 | 0.4796 | −0.0902 | t = −7.18 | < 1e-6 | *** |
| Sparsity (% features unchanged) | 4.60 % | 0.00 % | +4.60 pp | t = 20.64 | < 1e-6 | *** |
| Features changed | 12.40 | 13.00 | −0.60 | — | — | — |
| Convergence rate | 100 % | 100 % | 0 | — | — | — |

### Adult / Random Forest

| Metric | Wachter *(degenerate)* | GLANCE |
|---|---|---|
| Validity | 0.00 % | 20.40 % |
| Avg distance (L2) | 0.0115 | 1.5760 |
| Sparsity | 99.89 % | 92.31 % |
| Features changed | 0.01 | 1.00 |
| Convergence rate | 99.90 % | 100 % |

Wachter never crosses the decision boundary on RF — see Section 8.

### ACS / Logistic Regression

| Metric | Wachter | GLANCE | Wachter − GLANCE | Test | p-value | Sig |
|---|---|---|---|---|---|---|
| Validity | 17.70 % | 10.80 % | +6.90 pp | χ² = 18.92 | 1.4e-5 | *** |
| Avg distance (L2) | 0.1633 | 0.2202 | −0.0569 | t = −8.73 | < 1e-6 | *** |
| Sparsity | 0.08 % | 0.00 % | +0.08 pp | t = 2.14 | 0.032 | * |
| Features changed | 9.99 | 10.00 | −0.01 | — | — | — |
| Convergence rate | 100 % | 100 % | 0 | — | — | — |

### ACS / Random Forest

| Metric | Wachter *(degenerate)* | GLANCE |
|---|---|---|
| Validity | 0.20 % | 21.90 % |
| Avg distance (L2) | 0.0068 | 1.1140 |
| Sparsity | 99.89 % | 80.00 % |
| Features changed | 0.01 | 2.00 |
| Convergence rate | 100 % | 100 % |

### Interpretation

- **Validity (non-degenerate cells):** Wachter beats GLANCE in both LR cells (Adult by 14.8 pp, ACS by 6.9 pp), and both differences are highly significant — a result the original n = 100 study lacked the power to detect.
- **Proximity:** Wachter produces L2-closer counterfactuals on LR cells; on RF cells Wachter's distance is artifactually small because it barely moves the instance at all.
- **Sparsity:** Both methods change most or all features on LR. On RF, sparsity ≥ 80 % for both methods, reflecting that tree-ensemble decision boundaries push counterfactuals into extreme corners of feature space.

---

## Fairness Analysis

### Demographic Parity Gap by Sex

| Cell | Method | Female validity | Male validity | DP gap | 95 % CI | p-value | Sig |
|---|---|---|---|---|---|---|---|
| Adult / LR | Wachter | 33.20 % | 71.20 % | **38.00 %** | [32.27 %, 43.73 %] | < 1e-6 | *** |
| Adult / LR | GLANCE | 19.40 % | 55.40 % | **36.00 %** | [30.43 %, 41.57 %] | < 1e-6 | *** |
| Adult / RF | Wachter *(degen)* | 0.00 % | 0.00 % | 0.00 % | [0, 0] | 1.0 | ns |
| Adult / RF | GLANCE | 6.80 % | 34.00 % | **27.20 %** | [22.50 %, 31.90 %] | < 1e-6 | *** |
| ACS / LR | Wachter | 16.11 % | 19.01 % | 2.90 % | [0, 7.62 %] | 0.266 | ns |
| ACS / LR | GLANCE | 10.60 % | 10.97 % | 0.37 % | [0, 4.23 %] | 0.931 | ns |
| ACS / RF | Wachter *(degen)* | 0.00 % | 0.36 % | 0.36 % | [0, 0.87 %] | 0.505 | ns |
| ACS / RF | GLANCE | 20.71 % | 22.87 % | 2.15 % | [0, 7.29 %] | 0.458 | ns |

### Critical observations

- **Adult sex disparity is robust to classifier choice.** On LR both methods show ~36–38 pp gaps with p < 1e-6. On RF the only non-degenerate method (GLANCE) still shows a 27.2 pp gap with p < 1e-6 and a tight 95 % CI of [22.5 %, 31.9 %]. The gap is a property of the data, not the linear boundary.
- **ACS shows no significant sex gap in any cell.** All non-degenerate methods produce gaps under 3 pp with non-significant p-values, and the 95 % CIs (~ ±5 pp at n = 1000) are tight enough to rule out a publishable disparity.
- **The n = 1000 scale-up tightened CIs roughly 3×** compared to the original n = 100 study (Adult: ±18 pp → ±6 pp; ACS: ±10 pp → ±5 pp), placing the fairness conclusions on much firmer statistical ground.

### Why the Adult vs ACS contrast?

| Factor | Likely impact on the sex gap |
|---|---|
| Temporal shift | 1994 labour-market gender disparities are sharper than 2018 California |
| Class balance | 41 % positive (ACS) vs 24 % (Adult) → different prior structure |
| Training size | 5× more data may smooth idiosyncratic group bias |
| Geographic focus | California's labour market is more gender-balanced than the 1994 US average |

---

## FACTS Framework Evaluation (4 dimensions)

FACTS evaluates counterfactual fairness across four dimensions:

- **Equal Burden** — average cost across groups should be similar
- **Equal Effectiveness** — success rate (validity) should be similar across groups
- **Equal Choice** — the number of options available should be similar
- **Equal Cost of Effectiveness** — cost to reach a target success rate (default 80 %) should be similar

### Equal Burden (cost similarity)

| Cell | Method | Female avg L2 | Male avg L2 | Disparity |
|---|---|---|---|---|
| Adult / LR | Wachter | 0.8170 | 0.7113 | 0.1057 |
| Adult / LR | GLANCE | 0.5262 | 0.4985 | 0.0277 |
| Adult / RF | GLANCE | 1.5760 | 1.5760 | 0.0000 |
| ACS / LR | Wachter | 0.3163 | 0.3215 | 0.0052 |
| ACS / LR | GLANCE | 0.2335 | 0.2316 | 0.0019 |
| ACS / RF | GLANCE | 1.1140 | 1.1140 | 0.0000 |

GLANCE's perfect (0.0000) burden disparity on RF cells is by construction — a single global rule produces identical L2 changes for every instance.

### Equal Effectiveness (success-rate parity)

Same as the DP gap table above. Adult cells **fail** Equal Effectiveness with p < 1e-6 (both classifiers, both methods where non-degenerate). ACS cells **pass** Equal Effectiveness in every non-degenerate combination.

### Equal Choice (option availability)

Mirrors Equal Effectiveness here — GLANCE applies one global rule regardless of group, so "options available" reduces to validity by group. Same conclusions as Equal Effectiveness.

### Equal Cost of Effectiveness (target = 0.80)

| Cell | Method | Female cost | Male cost | Disparity |
|---|---|---|---|---|
| All 8 (cell × method) | both | n/a | n/a | 0.0000 |

The 80 % target is **unreachable** in every cell — the highest group validity observed anywhere is 71.2 % (Adult / LR / Wachter / Male). When the target is unreachable, the metric returns n/a per group and 0 disparity. **This is a real finding, not a defect:** at the default target the dimension adds no incremental signal. Lowering `config.target_effectiveness` to ~ 0.30 would make this dimension discriminative; that sensitivity analysis is out of scope of the current 4-way design.

### FACTS pass/fail summary

| Cell | Equal Burden | Equal Effectiveness | Equal Choice | Equal Cost of Effectiveness |
|---|---|---|---|---|
| Adult / LR | PASS (small disparity) | **FAIL** | **FAIL** | n/a (unreachable target) |
| Adult / RF | PASS (degenerate or single-rule) | **FAIL** (GLANCE) | **FAIL** (GLANCE) | n/a |
| ACS / LR | PASS | PASS | PASS | n/a |
| ACS / RF | PASS | PASS | PASS | n/a |

---

## Featured Finding: Wachter on Tree Ensembles

### The phenomenon

| Cell | Wachter validity | Wachter sparsity (% features unchanged) |
|---|---|---|
| Adult / LR | 52.20 % | 4.60 % |
| ACS / LR | 17.70 % | 0.08 % |
| Adult / RF | **0.00 %** | **99.89 %** |
| ACS / RF | **0.20 %** | **99.89 %** |

Moving from LR to RF as the underlying classifier:

- Validity collapses from 17–52 % to 0–0.2 %
- Sparsity rises to 99.89 % (i.e. Wachter changes essentially no features at all)
- Average L2 distance drops to ~0.01 — counterfactuals stay glued to the original point

### Why it happens

Wachter optimizes
`L = (1 − P(y = 1 | x_cf))² + λ · ‖x_cf − x‖²`
using central finite-difference gradients via `predict_proba`. RandomForest's `predict_proba` is **piecewise-constant**: within a single leaf region (which can span large parts of the input space for tree-ensemble averages), `predict_proba(x ± ε) ≈ predict_proba(x)` for any small ε. The numerical gradient therefore vanishes, the proximity term dominates, and the optimizer takes near-zero steps. The counterfactual never crosses the decision boundary.

### Why GLANCE survives

GLANCE's Phase 2 applies a pre-computed global rule additively: `x_cf = x + Δ_rule`. The rule is learned once on seed instances, after which no gradients are queried at counterfactual-generation time. GLANCE on RF achieves 20–22 % validity in both datasets — substantially lower than its LR performance but functional and informative.

### Practitioner recommendations

| Setting | Recommendation |
|---|---|
| Differentiable classifiers (LR, MLP, logistic with continuous features) | Wachter is fine |
| Tree ensembles (RF, GBM, XGBoost) | **Do not use Wachter with finite-difference gradients.** Either (a) distil a differentiable surrogate model and run Wachter against the surrogate, or (b) use a method family that does not rely on local gradients (GLANCE, FACE, prototype-based methods) |
| Method benchmarks across model families | Always report sparsity and L2 — a near-zero L2 with ~100 % sparsity is the unambiguous fingerprint of gradient collapse |

---

## Statistical Significance

### Method comparison (Wachter vs GLANCE)

| Test | Cell | Statistic | p-value | Result |
|---|---|---|---|---|
| Validity (χ²) | Adult / LR | 43.69 | < 1e-6 | *** |
| Validity (χ²) | ACS / LR | 18.92 | 1.4e-5 | *** |
| Distance (t) | Adult / LR | −7.18 | < 1e-6 | *** |
| Distance (t) | ACS / LR | −8.73 | < 1e-6 | *** |
| Sparsity (t) | Adult / LR | 20.64 | < 1e-6 | *** |
| Sparsity (t) | ACS / LR | 2.14 | 0.032 | * |

RF cells are omitted because the Wachter side is degenerate.

### Sex DP gap (chi-square per cell, per method)

| Cell | Method | χ² | p-value | Result |
|---|---|---|---|---|
| Adult / LR | Wachter | 143.16 | < 1e-6 | *** |
| Adult / LR | GLANCE | 136.85 | < 1e-6 | *** |
| Adult / RF | GLANCE | 122 (approx) | < 1e-6 | *** |
| ACS / LR | Wachter | 1.24 | 0.266 | ns |
| ACS / LR | GLANCE | 0.0075 | 0.931 | ns |
| ACS / RF | GLANCE | 0.55 | 0.458 | ns |

### 95 % confidence intervals (sex DP gap)

| Cell | Wachter CI | GLANCE CI |
|---|---|---|
| Adult / LR | [32.27 %, 43.73 %] | [30.43 %, 41.57 %] |
| Adult / RF | degenerate | [22.50 %, 31.90 %] |
| ACS / LR | [0, 7.62 %] | [0, 4.23 %] |
| ACS / RF | degenerate | [0, 7.29 %] |

---

## Visualizations

All figures are under `results_combined/figures/`. They are regenerated by `results_combined/build_figures.py` from `results_combined/_data.py`.

| Figure | File | Description |
|---|---|---|
| Fig. 1 | `fig1_classifier_performance.png` | Accuracy and AUC across all 4 cells |
| Fig. 2 | `fig2_validity_4way.png` | 2×2 grid of Wachter vs GLANCE validity; degenerate Wachter-RF cells red-bordered and hatched |
| Fig. 3 | `fig3_l2_distance_4way.png` | 2×2 grid of L2 distance, same degenerate annotation |
| Fig. 4 | `fig4_dp_gap_4way.png` | 2×2 grid of sex DP gap with 95 % CI error bars and significance stars |
| Fig. 5 | `fig5_facts_equal_effectiveness.png` | FACTS Equal Effectiveness gap across all 4 cells × 2 methods |
| Fig. 6 | `fig6_wachter_collapse.png` | **Featured** — Wachter sparsity jumps from < 5 % on LR to ~ 99.9 % on RF for both datasets |
| Fig. 7 | `fig7_equal_cost_of_effectiveness.png` | FACTS Equal Cost of Effectiveness across cells (all 0 at the 80 % target, documenting target unreachability) |

Per-cell single-image overviews (`comparison_overview.png`, `distribution_boxplots.png`) live in each cell's `*/comparison/` directory.

---

## Key Findings

### Primary contributions

1. **Dataset effects dominate over classifier effects for fairness conclusions on the data tested.** The Adult sex gap (38 pp on LR, 27 pp on RF) reproduces with p < 1e-6 under both classifiers — the data, not the linear boundary, drives it. ACS shows no significant sex gap under either classifier.

2. **n = 1000 method-comparison conclusions overturn n = 100 conclusions.** At n = 100, Wachter and GLANCE were "indistinguishable on validity" (Adult p = 1.0, ACS p = 0.24). At n = 1000, Wachter beats GLANCE significantly in both LR cells (Adult +14.8 pp p ≈ 0; ACS +6.9 pp p = 1.4e-5). The original study lacked the power to detect a real method-level difference.

3. **Wachter with finite-difference gradients is unsuitable for tree ensembles.** Documented quantitatively (validity 0–0.2 %, sparsity 99.89 %) and mechanistically (gradient vanishes inside RF leaf regions). Concrete practitioner guidance follows.

4. **GLANCE is the more transferable method across model families.** It maintains 20+ % validity on RF where Wachter collapses, because its rule application bypasses local gradients.

5. **FACTS Equal Cost of Effectiveness is uninformative at the default 80 % target.** No group reaches 80 % validity in any cell; the metric is well-defined but vacuous as configured. This is itself a useful observation for the framework's calibration.

### Implications for practitioners

| Recommendation | Rationale |
|---|---|
| Sample size n ≥ 1000 for fairness-sensitive method comparisons | n = 100 hides real method-level differences and produces ±18 pp CIs on gaps |
| Test on contemporary data, not just Adult 1994 | 30-year-old US labour-market disparities exaggerate gaps that may not generalize to current systems |
| Always include a non-linear classifier in fairness benchmarks | LR-only benchmarks cannot separate "data fairness" from "boundary fairness" |
| Report sparsity alongside L2 | A near-zero L2 with ~100 % sparsity is the unambiguous fingerprint of gradient collapse |
| Calibrate FACTS targets to observed validity ceilings | Default 0.80 was unreachable in every cell of this benchmark |

---

## Reproducibility

### Run all 4 cells

```bash
# Setup
pip install -r requirements.txt
pip install -e .

# Adult
python run_adult_experiment.py        # LR + Wachter + GLANCE
python run_adult_comparison.py        # Statistical tests + plots
python run_adult_experiment_rf.py     # Random Forest version
python run_adult_comparison_rf.py

# ACS
python run_acs_experiment.py
python run_acs_comparison.py
python run_acs_experiment_rf.py
python run_acs_comparison_rf.py

# Consolidated 4-way artifacts
cd results_combined
python build_report.py    # → four_way_report.txt
python build_figures.py   # → figures/fig1..fig7.png
```

Per-cell artifacts land under `results_{adult,adult_rf,acs,acs_rf}/`. The pickled counterfactual results (`{wachter,glance}/results.pkl`) are gitignored but regenerated by the scripts; the comparison reports, JSON metrics, and figures under `*/comparison/` are committed.

### Experimental configuration

| Parameter | Value |
|---|---|
| Sample size per cell | n = 1000 negatively-predicted test instances |
| Sex balance | 50 / 50 |
| Random seed | 42 |
| Wachter | λ = 0.1, lr = 0.01, max_iter = 1000, tolerance = 1e-3, gradient ε = 1e-5 |
| GLANCE | k = 5 rules, seed sample = 200, λ = 0.1 |
| Logistic Regression | lbfgs solver, max_iter = 1000 |
| Random Forest | n_estimators = 100, max_depth = 10, n_jobs = -1 |
| FACTS | α = 0.05, target_effectiveness = 0.80 |

### Saved artifacts

| Path | Description |
|---|---|
| `results/comparison/` | Original n = 100 Adult baseline (preserved for reference) |
| `results_adult/comparison/` | Adult / LR — report, JSON, plots |
| `results_adult_rf/comparison/` | Adult / RF — report, JSON, plots |
| `results_acs/comparison/` | ACS / LR — report, JSON, plots |
| `results_acs_rf/comparison/` | ACS / RF — report, JSON, plots |
| `results_combined/RESULTS_4WAY.md` | Paper-ready 4-way narrative (6 sections) |
| `results_combined/four_way_report.txt` | Plain-text consolidated tables |
| `results_combined/figures/` | 7 publication-quality figures |
| `results_combined/_data.py` | Single source of truth for all 4-way numbers |
| `results_combined/build_{report,figures}.py` | Reproducible regeneration of consolidated artifacts |

Raw counterfactual artifacts (gitignored, regenerable):
- `results_*/wachter/results.pkl` — Wachter counterfactuals per cell
- `results_*/glance/results.pkl` — GLANCE counterfactuals per cell
- `results_*/classifier.pkl` — Trained classifier per cell

---

## References

1. Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual Explanations without Opening the Black Box.* arXiv:1711.00399
2. Kavouras, L., et al. (2024). *GLANCE: Global Actions in a Nutshell for Counterfactual Explainability.* arXiv:2405.18921
3. Kavouras, L., et al. (2023). *FACTS: Fairness-Aware Counterfactuals for Subgroups.* arXiv:2306.14978
4. Ding, F., et al. (2021). *Retiring Adult: New Datasets for Fair Machine Learning.* NeurIPS Datasets and Benchmarks.
5. Dua, D. & Graff, C. (2017). *UCI Machine Learning Repository.* University of California, Irvine.

---

*Results generated using the Counterfactual Fairness Comparison framework. For questions or issues, please open a GitHub issue.*
