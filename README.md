# Counterfactual Explanations Fairness Comparison

A 2×2 benchmark of instance-specific (Wachter) and global (GLANCE) counterfactual explanation methods across two datasets (UCI Adult 1994, ACS Folktables CA 2018) and two classifiers (Logistic Regression, Random Forest), evaluated through the FACTS fairness framework.

## Overview

The project answers two reviewer questions raised against single-dataset / single-classifier counterfactual fairness studies:

1. **Do dataset characteristics or linear decision boundaries drive observed fairness gaps?**
   We replicate every comparison across {Adult, ACS} × {LR, RF}, at **n = 1000** instances per cell (balanced 50/50 by sex). The Adult sex-disparity reproduces under Random Forest (GLANCE: 27.2 pp gap, p < 1e-6), refuting the linearity hypothesis. ACS shows no significant sex gap under either model.

2. **Are gradient-based counterfactual methods transferable across model families?**
   No — Wachter with finite-difference gradients catastrophically fails on tree ensembles (validity collapses from 17–52 % on LR to 0–0.2 % on RF; sparsity rises to 99.89 %). Reported as a featured methodological finding in `results_combined/RESULTS_4WAY.md` Section 5.

### Headline results (n = 1000 per cell)

| Cell | Classifier acc / AUC | Wachter validity | GLANCE validity | Sex DP gap (GLANCE) |
|---|---|---|---|---|
| Adult + LR | 0.819 / 0.848 | 52.20 % | 37.40 % | **36.00 % (p < 1e-6)** |
| Adult + RF | 0.855 / 0.913 | 0.00 % (degenerate) | 20.40 % | **27.20 % (p < 1e-6)** |
| ACS + LR | 0.787 / 0.861 | 17.70 % | 10.80 % | 0.37 % (n.s.) |
| ACS + RF | 0.814 / 0.895 | 0.20 % (degenerate) | 21.90 % | 2.15 % (n.s.) |

**Conclusion:** the Adult sex-disparity is a property of the data, not the model class; the ACS-CA dataset shows uniformly small, non-significant sex gaps across both classifiers.

## Project structure

```
Counterfactual_Project/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central Config dataclass (n=1000, seeds, hyperparams)
│   ├── counterfactual_base.py       # Abstract counterfactual generator
│   ├── data_loader.py               # Adult (UCI) loader
│   ├── acs_data_loader.py           # ACS (Folktables) loader
│   ├── classifiers.py               # LR + RF wrappers with fairness evaluation
│   ├── wachter_method.py            # Instance-specific, finite-difference Wachter
│   ├── glance_method.py             # Global K-means rule extraction (GLANCE)
│   ├── facts_fairness.py            # FACTS framework — 4 dimensions
│   └── comparison_analysis.py       # Statistical tests, plots, reports
├── data/                            # Datasets (gitignored; see Installation)
├── run_adult_experiment.py          # Adult + LR runner
├── run_adult_comparison.py          # Adult + LR comparison + plots
├── run_adult_experiment_rf.py       # Adult + RF runner
├── run_adult_comparison_rf.py       # Adult + RF comparison + plots
├── run_acs_experiment.py            # ACS + LR runner
├── run_acs_comparison.py            # ACS + LR comparison + plots
├── run_acs_experiment_rf.py         # ACS + RF runner
├── run_acs_comparison_rf.py         # ACS + RF comparison + plots
├── run_adult_glance_rf_only.py      # Recovery script: GLANCE-only on saved RF classifier
├── analyze_decision_boundary.py     # Decision boundary analysis
├── generate_cross_dataset_plots.py  # Adult vs ACS cross-dataset plots
├── results/                         # n=100 Adult baseline (preserved for reference)
├── results_acs/comparison/          # ACS + LR reports + figures
├── results_adult/comparison/        # Adult + LR reports + figures
├── results_acs_rf/comparison/       # ACS + RF reports + figures
├── results_adult_rf/comparison/     # Adult + RF reports + figures
├── results_combined/                # Paper-ready 4-way artifacts
│   ├── RESULTS_4WAY.md              # 6-section narrative (paper-ready)
│   ├── four_way_report.txt          # Plain-text consolidated tables
│   ├── figures/                     # 7 publication-quality figures
│   ├── _data.py                     # Single source of truth for numbers
│   ├── build_report.py              # Regenerate four_way_report.txt
│   └── build_figures.py             # Regenerate figures/
├── tests/                           # pytest suite
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/<your-username>/Counterfactual_Project.git
cd Counterfactual_Project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Datasets

- **Adult (UCI 1994)** — place `adult.data` and `adult.test` in `data/`. Source: https://archive.ics.uci.edu/ml/datasets/adult
- **ACS (Folktables CA 2018)** — auto-downloaded by `ACSDataLoader` on first run into `data/2018/1-Year/`. Requires internet on first run.

Datasets are gitignored. Each experiment script will preprocess and cache locally.

## Usage

### Reproduce all 4 cells

```bash
# Adult
python run_adult_experiment.py        # Generate Wachter + GLANCE CFs (LR)
python run_adult_comparison.py        # Statistical tests + plots
python run_adult_experiment_rf.py     # Same with Random Forest
python run_adult_comparison_rf.py

# ACS
python run_acs_experiment.py
python run_acs_comparison.py
python run_acs_experiment_rf.py
python run_acs_comparison_rf.py
```

Per-cell artifacts land under `results_{adult,adult_rf,acs,acs_rf}/`. The pickled counterfactual results (`{wachter,glance}/results.pkl`) are gitignored but regenerated by the scripts; the comparison reports, JSON metrics, and figures under `*/comparison/` are committed.

### Build the consolidated 4-way artifacts

```bash
cd results_combined
python build_report.py     # → four_way_report.txt
python build_figures.py    # → figures/fig1..fig7.png
```

`_data.py` is the single source of truth — it lifts numbers verbatim from each cell's `comparison_report.txt`. Editing it and re-running the build scripts is idempotent.

### Using individual components

```python
from src.acs_data_loader import ACSDataLoader
from src.wachter_method import WachterCounterfactual
from src.glance_method import GLANCECounterfactual
from src.facts_fairness import FACTSEvaluator
from sklearn.ensemble import RandomForestClassifier

# Load data
loader = ACSDataLoader(data_dir="data")
dataset = loader.load_processed_data(states=["CA"])

# Train any sklearn-compatible classifier (LR or RF)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(dataset["X_train"], dataset["y_train"])

# Generate counterfactuals (Wachter uses finite-difference gradients on any predict_proba)
wachter = WachterCounterfactual(
    classifier=model,
    feature_names=dataset["feature_names"],
    continuous_features=dataset["continuous_features"],
    categorical_features=dataset["categorical_features"],
)
results = wachter.generate_batch(X_test, target_class=1)

# Evaluate fairness with FACTS (4 dimensions)
evaluator = FACTSEvaluator(alpha=0.05)
fairness_results = evaluator.evaluate_all(grouped_results)
evaluator.print_report(fairness_results, "Wachter")
```

## Methods

### Wachter (instance-specific)
Per-instance optimization that finds the closest counterfactual:

```
L = (1 − P(y=1 | x_cf))² + λ ‖x_cf − x‖²
```

Gradients are computed by central finite differences via `predict_proba` — model-agnostic by construction. **Limitation:** on piecewise-constant decision surfaces (tree ensembles), gradients vanish inside leaf regions and the optimizer never crosses the boundary. See `results_combined/RESULTS_4WAY.md` §5.

### GLANCE (global)
Discovers population-level recourse rules by clustering instance-specific change vectors:

1. Generate seed counterfactuals for a sample population.
2. Extract change vectors `Δ = x_cf − x`.
3. K-means cluster on `Δ`.
4. Pick the rule with the highest validity and apply additively to new instances.

GLANCE survives the tree-ensemble case because its rule application is decoupled from local gradients.

### FACTS fairness framework
Evaluates counterfactual fairness across 4 dimensions:

- **Equal Burden** — average cost similarity across groups
- **Equal Effectiveness** — success-rate parity
- **Equal Choice** — option availability
- **Equal Cost of Effectiveness** — cost to reach a target success rate (vacuous at the default 80 % target on these benchmarks; useful at lower targets)

## Citation

```bibtex
@article{counterfactual_fairness_2026,
  title={Dataset Transferability and Classifier Robustness in Counterfactual Fairness: A 2×2 Benchmark of Adult and ACS under Logistic Regression and Random Forest},
  author={Pinto Ickowicz, Priscila and El Bekkaoui, Souhayla},
  journal={INFO-H512 Course Project, Universit\'{e} Libre de Bruxelles},
  year={2026}
}
```

## References

1. Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual Explanations without Opening the Black Box*. arXiv:1711.00399.

2. Guidotti, R. (2022). *Counterfactual Explanations and How to Find Them*. Data Mining and Knowledge Discovery.

3. Kavouras, L., et al. (2024). *GLANCE: Global Actions in a Nutshell for Counterfactual Explainability*. arXiv:2405.18921.

4. Ustun, B., Spangher, A., & Liu, Y. (2019). *Actionable Recourse in Linear Classification*. ACM Conference on Fairness, Accountability, and Transparency (FAccT).

5. Mothilal, R. K., Sharma, A., & Tan, C. (2020). *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations*. ACM Conference on Fairness, Accountability, and Transparency (FAccT).

6. Karimi, A. H., Barthe, G., Schölkopf, B., & Valera, I. (2021). *A Survey of Algorithmic Recourse*. arXiv:2010.04050.

7. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. Available at fairmlbook.org.

8. Dua, D., & Graff, C. (2017). *UCI Machine Learning Repository*. University of California, Irvine.

9. Kavouras, L., Sacharidis, D., Arvanitidis, G., & Palpanas, T. (2023). *FACTS: Fairness-Aware Counterfactuals for Subgroups*. arXiv:2306.14978.

10. Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). *Retiring Adult: New Datasets for Fair Machine Learning*. Advances in Neural Information Processing Systems (NeurIPS).

11. Mitchell, S., et al. (2021). *Algorithmic Fairness: Choices, Assumptions, and Definitions*. Annual Review of Statistics and Its Application.


## License

MIT License — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

- UCI Machine Learning Repository (Adult dataset)
- Folktables (ACS dataset)
- The authors of the Wachter, GLANCE, and FACTS papers
