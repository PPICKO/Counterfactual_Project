# Counterfactual Fairness: A 2x2 Benchmark of Method and Classifier

**Wachter (gradient-based) vs GLANCE (rule-based)** evaluated on two datasets
(Adult, ACS) and two classifiers (Logistic Regression, Random Forest).

All numbers in this document are reproduced verbatim from the per-cell
`comparison_report.txt` files under `results_{adult,acs}{,_rf}/comparison/`.
Classifier accuracy and AUC are computed on the full held-out test split from
the saved `classifier.pkl` artifacts. No values are estimated, smoothed, or
re-run.

---

## 1. Experimental design

Each of the four cells uses identical preprocessing, sampling, and method
configuration. The only axes that vary are dataset and classifier:

| Axis | Levels |
|------|--------|
| Dataset | UCI Adult (n_test = 15,060), ACS PUMS 2018 1-Year (n_test = 39,133) |
| Classifier | LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=100, max_depth=10) |
| Counterfactual method | Wachter (finite-difference gradient descent on `predict_proba`), GLANCE (rule discovery + additive application) |
| Sample size | 1,000 instances per cell, drawn from individuals predicted negative, balanced 50/50 by `sex` |
| Protected attribute | `sex` (binary) |

Adult uses 13 features after preprocessing (`age`, `workclass`, `education`,
`education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`,
`capital-gain`, `capital-loss`, `hours-per-week`, `native-country`). ACS uses
10 (`AGEP`, `COW`, `SCHL`, `MAR`, `OCCP`, `POBP`, `RELP`, `WKHP`, `SEX`,
`RAC1P`). Sex is encoded 0=Female / 1=Male on Adult and 1=Male / 2=Female on
ACS.

The negative-prediction pool is sub-sampled to a 50/50 sex split using
`numpy.random.default_rng(config.random_seed)` so that any per-group rate
difference is not an artifact of class imbalance.

---

## 2. Classifier performance

|                      | Adult / LR | Adult / RF | ACS / LR | ACS / RF |
|----------------------|-----------:|-----------:|---------:|---------:|
| Test accuracy        |     0.8191 |     0.8548 |   0.7866 |   0.8144 |
| Test AUC             |     0.8482 |     0.9132 |   0.8613 |   0.8953 |
| n_test               |     15,060 |     15,060 |   39,133 |   39,133 |

Random Forest improves both metrics on both datasets, as expected. The lift is
modest (+3-4pp accuracy, +3-7pp AUC) but meaningful: it confirms that the RF
cells are not trivially worse classifiers and that any counterfactual-method
failure on RF must come from the interaction with the decision surface, not
from a degraded base model.

---

## 3. Method comparison (Wachter vs GLANCE)

### Validity (fraction of counterfactuals that cross the decision boundary)

| Method   | Adult / LR | Adult / RF       | ACS / LR | ACS / RF        |
|----------|-----------:|-----------------:|---------:|----------------:|
| Wachter  |     52.20% | **0.00%** (degen) |  17.70% | **0.20%** (degen) |
| GLANCE   |     37.40% |           20.40% |  10.80% |          21.90% |

### Mean L2 distance (proximity)

| Method   | Adult / LR | Adult / RF        | ACS / LR | ACS / RF         |
|----------|-----------:|------------------:|---------:|-----------------:|
| Wachter  |     0.3894 | **0.0115** (degen) |  0.1633 | **0.0068** (degen) |
| GLANCE   |     0.4796 |            1.5760 |  0.2202 |           1.1140 |

### Sparsity (fraction of features left unchanged)

| Method   | Adult / LR | Adult / RF       | ACS / LR | ACS / RF        |
|----------|-----------:|-----------------:|---------:|----------------:|
| Wachter  |      4.60% | **99.89%** (degen) |   0.08% | **99.89%** (degen) |
| GLANCE   |      0.00% |           92.31% |   0.00% |          80.00% |

The pattern on the three non-degenerate cells is consistent: when both methods
actually run, Wachter produces counterfactuals that are closer in L2 to the
factual (its objective directly penalises that distance) but achieves higher
validity than GLANCE on Adult-LR (+14.8pp) and ACS-LR (+6.9pp). GLANCE, by
contrast, is a one-step additive rule applied uniformly across instances; it
trades off proximity for speed (single iteration vs ~76 mean iterations for
Wachter on Adult-LR) and stability.

GLANCE's Random-Forest cells deserve a separate note. The L2 distance jumps
sharply (Adult: 0.48 -> 1.58; ACS: 0.22 -> 1.11) because the global rules
discovered in Phase-1 must shift larger continuous features (e.g. capital
gains, hours per week) to push instances across the more jagged RF boundary.
Validity nevertheless holds (Adult: 37.4% -> 20.4%; ACS: 10.8% -> 21.9%),
confirming that the *method* still produces counterfactuals --- it has not
collapsed. This is the key contrast with Wachter on RF.

---

## 4. Fairness analysis

We measure the demographic-parity gap on sex as
`P(valid | male) - P(valid | female)`, with a 95% CI from bootstrap and a
chi-square test (Fisher's exact when an arm is empty).

| Method   | Adult / LR                       | Adult / RF                       | ACS / LR                         | ACS / RF                         |
|----------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| Wachter  | **38.00%** *** (CI 32.3, 43.7)   | **0.00%** n/a (degen)             |  2.90% ns (CI 0.0, 7.6)          | 0.36% ns (degen) (CI 0.0, 0.9)   |
| GLANCE   | **36.00%** *** (CI 30.4, 41.6)   | **27.20%** *** (CI 22.5, 31.9)    |  0.37% ns (CI 0.0, 4.2)          | 2.15% ns (CI 0.0, 7.3)           |

*** = p < 0.001 (chi-square); ns = not significant at alpha = 0.05.

### Addressing the linear-boundary objection

A natural reviewer concern after seeing the Adult-LR result is that the large
sex disparity (~37%) might be an artifact of the linear decision boundary ---
specifically, that LR over-weights the `sex` indicator or the features most
correlated with it, and that a more expressive classifier would absorb the
disparity into other dimensions.

**The RF cells refute this.** On Adult-RF, GLANCE recovers a 27.2% DP gap
(p < 1e-6, chi-square = 112.2, 95% CI [22.5%, 31.9%]) on the exact same
sample. The disparity contracts somewhat (from 36% to 27%) but remains
statistically and practically large. The Wachter-RF cell is degenerate so it
cannot be used as evidence either way, but the GLANCE-RF cell --- which is a
methodologically valid run --- carries the signal cleanly.

The ACS cells show the opposite robustness: the gap is small and not
significant under either classifier and under either method, with all four
estimates in the 0.4%-2.9% range. The headline conclusion --- *Adult exhibits
a large, real, classifier-robust sex disparity in recourse; ACS does not* ---
is therefore not a classifier artifact and survives the 2x2 robustness check.

### FACTS Equal-Effectiveness gap

The FACTS framework's Equal-Effectiveness axis measures success-rate parity
under recourse and, by construction, coincides with the per-group validity
gap. We include it explicitly because the framework's other axes
(Equal Burden, Equal Choice) are reported in the per-cell artifacts and a
reader cross-checking the FACTS section will expect this number.

| Method   | Adult / LR | Adult / RF       | ACS / LR | ACS / RF |
|----------|-----------:|-----------------:|---------:|---------:|
| Wachter  |     38.00% *** | 0.00% (degen)  |  2.90% ns |  0.36% ns (degen) |
| GLANCE   |     36.00% *** | 27.20% ***     |  0.37% ns |  2.15% ns |

Equal-Burden cost disparity (|mean L2 female - mean L2 male|) is uniformly
small under GLANCE (<= 0.028 on every cell) and modest under Wachter on Adult
(0.106). The Adult Wachter cell is the only place where one sex faces a
materially more expensive recourse, and the size of that effect should be
reported alongside the validity gap rather than independently of it.

### FACTS Equal Cost of Effectiveness (4th dimension)

The fourth FACTS axis (`evaluate_equal_cost_of_effectiveness` in
`src/facts_fairness.py`) asks a different question from Equal Burden. Where
Equal Burden looks at the *average* cost over all successful counterfactuals
in a group, Equal Cost of Effectiveness looks at the cost required for the
*cheapest 80% of subjects in each group* to succeed --- i.e. the cost at a
fixed effectiveness threshold. The two can diverge: a group with a long
right-tail of expensive counterfactuals can have a low average cost but a
high cost-at-target.

| Method   | Adult / LR | Adult / RF | ACS / LR | ACS / RF |
|----------|-----------:|-----------:|---------:|---------:|
| Wachter  | N/A (target unreachable) | N/A (degen) | N/A (target unreachable) | N/A (degen) |
| GLANCE   | N/A (target unreachable) | N/A (target unreachable) | N/A (target unreachable) | N/A (target unreachable) |

With the project's configured target (`config.target_effectiveness = 0.80`),
the 4th dimension is well-defined but **vacuous in every cell of this
benchmark**: the highest observed group success rate is 71.2%
(Adult / LR, Wachter, Male), and no group reaches 80% in any other cell. The
FACTSEvaluator therefore returns `NaN` for the per-group cost-at-target and
`0.0000` for the disparity in all 8 (cell, method) pairs --- consistent with
its design when the target is not reachable, not a bug.

The practical reading: at the chosen 80% target, the 4th dimension adds no
incremental fairness signal beyond Equal Burden and Equal Effectiveness on
this benchmark. To make it informative one would lower the target to a value
the data actually reaches (e.g. 0.30 would be reachable in every
non-degenerate cell, and 0.50 would isolate Adult / LR Wachter Male as the
only cell that clears it). We keep the 80% target as configured to preserve
the existing experimental settings, and report the empty result here for
completeness rather than dropping the dimension.

---

## 5. Featured methodological finding: finite-difference gradient collapse of Wachter on tree ensembles

The two Random-Forest cells under Wachter exhibit a sharp, reproducible
failure mode that is worth elevating from a footnote to a section in its own
right, because it has direct implications for any practitioner planning to
apply Wachter-style counterfactual generation to a non-differentiable model.

### The phenomenon

Across both datasets, switching the classifier from Logistic Regression to
Random Forest collapses the Wachter optimiser in the same way:

|                | Adult / LR | Adult / RF | ACS / LR | ACS / RF |
|----------------|-----------:|-----------:|---------:|---------:|
| Validity       |     52.20% |      0.00% |   17.70% |    0.20% |
| Mean L2        |     0.3894 |     0.0115 |   0.1633 |   0.0068 |
| Sparsity       |      4.60% |     99.89% |    0.08% |   99.89% |
| Mean iters     |       76.0 |       20.9 |     46.3 |      ~21 |
| Convergence    |    100.00% |     99.90% |  100.00% |  100.00% |

Validity collapses from 17-52% to essentially zero. Mean L2 drops by roughly
two orders of magnitude (Adult: 0.39 -> 0.01; ACS: 0.16 -> 0.007). Sparsity
rises to 99.89% --- on average fewer than 0.02 of the ~10 features have
shifted at all. The optimiser still reports "converged" because its loss
plateaus, but the produced point is essentially unchanged from the factual
and never crosses the decision boundary.

### Why it happens

Wachter's loss

    L(x') = (f(x') - y_target)^2 + lambda * d(x, x')

is minimised by gradient descent over `x'`. Because `RandomForestClassifier`
is not differentiable, the implementation in `src/wachter_method.py` falls
back to a central-difference numerical gradient:

    grad_i ~= (f(x' + h * e_i) - f(x' - h * e_i)) / (2 * h)

This works for any continuous `f`, including `predict_proba` of an LR. But
`RandomForestClassifier.predict_proba` is **piecewise-constant**: the output
is the average over the trees of leaf-level class frequencies. Inside any
single leaf-region of any single tree, `predict_proba(x' + h * e_i)` returns
the **same value** as `predict_proba(x' - h * e_i)` whenever the perturbation
`h` is small enough that the perturbed point stays in the same leaf in every
tree.

Concretely, with `h = 1e-4` on standardised features and `max_depth = 10`,
that condition is met for virtually every starting point. The numerical
gradient therefore evaluates to zero on every coordinate, the optimiser takes
a near-zero step, and after the maximum number of iterations the
counterfactual sits within the same leaf-region as the factual --- still
classified as negative, never having moved.

This is not a bug in our implementation; it is a fundamental incompatibility
between finite-difference optimisation and piecewise-constant decision
surfaces.

### Why GLANCE survives

GLANCE has the same Phase-1 (it actually relies on Wachter internally to
discover successful counterfactual examples), but the headline observation is
that Phase-1 only needs to find a handful of successful directions across the
*global* dataset --- not at every instance. Crucially, the gradient collapse
described above produces near-zero perturbations but doesn't actually need to
succeed for GLANCE's Phase-2 to work: GLANCE's apply-time operation is a
fixed additive rule, not an iterative optimisation. It does not query the
classifier gradient at apply time at all. So even when Wachter fails to move
points across the RF boundary in Phase-1, the Phase-2 application can still
push points across by applying the discovered rule additively, with no
gradient dependence.

(In our specific runs, both RF cells degrade GLANCE's Phase-1 rule discovery
to a single rule that is then applied to all 1,000 instances --- the "Rule 0:
1000 instances" line in the comparison reports --- but the rule itself still
moves ~20% of instances across the boundary, which is the validity figure
reported.)

### Implication for practitioners

Gradient-based counterfactual methods designed for differentiable models
(Wachter, DiCE, CEM, several others in the recent literature) should not be
naively combined with `finite_difference` or `numerical_grad` flags when the
underlying classifier is a tree ensemble. The visible failure mode in this
benchmark --- 99.89% sparsity, ~0% validity, "converged: True" --- looks
superficially like an exceptional result (perfect proximity!) and can pass
sanity checks that look only at L2 distance or convergence flags.

Recommended alternatives:

1. **Surrogate-distillation**: train a small differentiable surrogate
   (e.g. a 2-layer MLP with smooth activations) on `predict_proba` outputs of
   the tree ensemble; run Wachter against the surrogate; verify validity
   against the original ensemble.
2. **Methods that don't query gradients at apply time**: GLANCE-style
   rule-based methods, FACE (Feasible and Actionable Counterfactual
   Explanations), prototype-based methods. These remain operational on
   tree ensembles by construction.
3. **Tree-aware methods**: `FOCUS` (Lucic et al.), `MACE`, and direct
   tree-traversal counterfactual algorithms that reason explicitly about
   leaf-region boundaries.

A reasonable default-defensive rule is: if `hasattr(classifier, 'estimators_')`
or the classifier's `predict_proba` is not analytically differentiable, do
not use Wachter with a finite-difference gradient. The runtime cost of the
sanity check is negligible and would have caught both degenerate cells in
this benchmark.

---

## 6. Conclusions

The 2x2 design lets us separate two questions that get conflated in
single-classifier benchmarks: *is a fairness finding a property of the
classifier or of the data?* and *is a counterfactual-method comparison robust
to choice of classifier?*

The answers in this benchmark are, respectively:

- **Fairness conclusions are dataset-driven, not classifier-driven.** The
  Adult sex disparity (~36-38% DP gap on LR) persists at 27.2% under
  GLANCE-RF (p < 1e-6), and the ACS non-disparity (gaps <= 2.9%, all
  non-significant) holds across all four cells. The headline fairness result
  --- *Adult exhibits a large real sex disparity in recourse; ACS does not*
  --- is robust to the classifier choice.

- **Method comparisons are *not* uniformly robust across classifiers.**
  Wachter with finite-difference gradients fails catastrophically on tree
  ensembles. The Adult-LR and ACS-LR cells are interpretable as comparing two
  working methods; the RF cells are not. Any meta-analysis that pools the
  four cells without flagging the degenerate ones would draw incorrect
  conclusions about Wachter's effectiveness.

The combined artifact set (this document, `four_way_report.txt`, and the
figures in `figures/`) gives a reviewer enough material to verify both
conclusions independently and to inspect the degeneracy directly through the
sparsity column.

---

## Artifact index

| File | Purpose |
|------|---------|
| `four_way_report.txt` | Plain-text consolidated tables (this document's data, machine-readable) |
| `figures/fig1_classifier_performance.png` | Accuracy / AUC across the 4 cells |
| `figures/fig2_validity_4way.png` | Validity, 2x2 grid, degenerate cells annotated |
| `figures/fig3_l2_distance_4way.png` | L2 distance, 2x2 grid, degenerate cells annotated |
| `figures/fig4_dp_gap_4way.png` | Sex DP gap with 95% CI error bars |
| `figures/fig5_facts_equal_effectiveness.png` | FACTS Equal-Effectiveness gap with significance stars |
| `figures/fig6_wachter_collapse.png` | Featured figure: Wachter sparsity collapse on tree ensembles |
| `figures/fig7_equal_cost_of_effectiveness.png` | FACTS Equal Cost of Effectiveness (4th dimension); empty because 80% target is unreachable in every cell |
| `_data.py` | Single source of truth for all numbers in this document |
