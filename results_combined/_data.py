"""
Single source of truth for the 4-cell results matrix.

All numbers are copied directly from:
  results_{adult,acs}{,_rf}/comparison/comparison_report.txt
  results_{adult,acs}{,_rf}/comparison/metrics_summary.json (where present)
  Classifier accuracy/AUC re-scored from the saved *.pkl on each
    dataset's held-out test split (see retrieval note in the report).

DO NOT edit by hand without re-cross-checking the source reports.
"""

CELLS = ["adult_lr", "adult_rf", "acs_lr", "acs_rf"]

CELL_LABELS = {
    "adult_lr": "Adult / LR",
    "adult_rf": "Adult / RF",
    "acs_lr":   "ACS / LR",
    "acs_rf":   "ACS / RF",
}

# Classifier accuracy / AUC on the full held-out test split.
# Adult test n=15060 (after NA-drop); ACS test n=39133.
CLASSIFIER = {
    "adult_lr": {"accuracy": 0.8191, "auc": 0.8482, "n_test": 15060},
    "adult_rf": {"accuracy": 0.8548, "auc": 0.9132, "n_test": 15060},
    "acs_lr":   {"accuracy": 0.7866, "auc": 0.8613, "n_test": 39133},
    "acs_rf":   {"accuracy": 0.8144, "auc": 0.8953, "n_test": 39133},
}

# Wachter-on-RF cells are flagged as degenerate (finite-difference
# gradient collapse on piecewise-constant tree decision surfaces).
DEGENERATE = {"adult_rf__wachter", "acs_rf__wachter"}

# Counterfactual results per (cell, method).
# Values are taken verbatim from the comparison_report.txt for each cell.
# n_sample = 1000 in every cell.
CF = {
    # ---------------- Adult / LR ----------------
    ("adult_lr", "wachter"): {
        "validity": 0.5220, "l2": 0.3894, "sparsity": 0.0460,
        "features_changed": 12.40, "convergence": 1.0000,
        "validity_female": 0.3320, "validity_male": 0.7120,
        "l2_female": 0.8170, "l2_male": 0.7113,
        "dp_gap": 0.3800, "dp_gap_ci": (0.3227, 0.4373), "dp_p": 0.000000,
        "facts_ee_gap": 0.3800, "facts_ee_p": 0.000000,
        "cost_disparity": 0.1057,
        # FACTS Equal Cost of Effectiveness at target=0.80.
        # Target is unreachable in every cell (max observed group validity
        # = 71.2% < 80%), so all per-group costs are N/A and disparity = 0.
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },
    ("adult_lr", "glance"): {
        "validity": 0.3740, "l2": 0.4796, "sparsity": 0.0000,
        "features_changed": 13.00, "convergence": 1.0000,
        "validity_female": 0.1940, "validity_male": 0.5540,
        "l2_female": 0.5262, "l2_male": 0.4985,
        "dp_gap": 0.3600, "dp_gap_ci": (0.3043, 0.4157), "dp_p": 0.000000,
        "facts_ee_gap": 0.3600, "facts_ee_p": 0.000000,
        "cost_disparity": 0.0277,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },

    # ---------------- Adult / RF ----------------  (Wachter is DEGENERATE)
    ("adult_rf", "wachter"): {
        "validity": 0.0000, "l2": 0.0115, "sparsity": 0.9989,
        "features_changed": 0.01, "convergence": 0.9990,
        "validity_female": 0.0000, "validity_male": 0.0000,
        "l2_female": float("nan"), "l2_male": float("nan"),
        "dp_gap": 0.0000, "dp_gap_ci": (0.0000, 0.0000), "dp_p": 1.000000,
        "facts_ee_gap": 0.0000, "facts_ee_p": 1.000000,
        "cost_disparity": 0.0000,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },
    ("adult_rf", "glance"): {
        "validity": 0.2040, "l2": 1.5760, "sparsity": 0.9231,
        "features_changed": 1.00, "convergence": 1.0000,
        "validity_female": 0.0680, "validity_male": 0.3400,
        "l2_female": 1.5760, "l2_male": 1.5760,
        "dp_gap": 0.2720, "dp_gap_ci": (0.2250, 0.3190), "dp_p": 0.000000,
        "facts_ee_gap": 0.2720, "facts_ee_p": 0.000000,
        "cost_disparity": 0.0000,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },

    # ---------------- ACS / LR ----------------  (sex coded 1.0/2.0)
    ("acs_lr", "wachter"): {
        "validity": 0.1770, "l2": 0.1633, "sparsity": 0.0008,
        "features_changed": 9.99, "convergence": 1.0000,
        "validity_female": 0.1611, "validity_male": 0.1901,   # 1.0=453, 2.0=547
        "l2_female": 0.3163, "l2_male": 0.3215,
        "dp_gap": 0.0290, "dp_gap_ci": (0.0000, 0.0762), "dp_p": 0.266130,
        "facts_ee_gap": 0.0290, "facts_ee_p": 0.266130,
        "cost_disparity": 0.0052,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },
    ("acs_lr", "glance"): {
        "validity": 0.1080, "l2": 0.2202, "sparsity": 0.0000,
        "features_changed": 10.00, "convergence": 1.0000,
        "validity_female": 0.1060, "validity_male": 0.1097,
        "l2_female": 0.2335, "l2_male": 0.2316,
        "dp_gap": 0.0037, "dp_gap_ci": (0.0000, 0.0423), "dp_p": 0.930845,
        "facts_ee_gap": 0.0037, "facts_ee_p": 0.930845,
        "cost_disparity": 0.0019,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },

    # ---------------- ACS / RF ----------------  (Wachter is DEGENERATE)
    ("acs_rf", "wachter"): {
        "validity": 0.0020, "l2": 0.0068, "sparsity": 0.9989,
        "features_changed": 0.01, "convergence": 1.0000,
        "validity_female": 0.0000, "validity_male": 0.0036,
        "l2_female": float("nan"), "l2_male": 1.3756,
        "dp_gap": 0.0036, "dp_gap_ci": (0.0000, 0.0087), "dp_p": 0.504707,
        "facts_ee_gap": 0.0036, "facts_ee_p": 0.504707,
        "cost_disparity": 0.0000,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },
    ("acs_rf", "glance"): {
        "validity": 0.2190, "l2": 1.1140, "sparsity": 0.8000,
        "features_changed": 2.00, "convergence": 1.0000,
        "validity_female": 0.2071, "validity_male": 0.2287,
        "l2_female": 1.1140, "l2_male": 1.1140,
        "dp_gap": 0.0215, "dp_gap_ci": (0.0000, 0.0729), "dp_p": 0.457687,
        "facts_ee_gap": 0.0215, "facts_ee_p": 0.457687,
        "cost_disparity": 0.0000,
        "facts_coe_target": 0.80,
        "facts_coe_female": float("nan"),
        "facts_coe_male": float("nan"),
        "facts_coe_disparity": 0.0000,
    },
}


def is_degenerate(cell: str, method: str) -> bool:
    return f"{cell}__{method}" in DEGENERATE


def fmt_pct(x, decimals=2):
    if x != x:  # NaN
        return "n/a"
    return f"{100*x:.{decimals}f}%"


def fmt_float(x, decimals=4):
    if x != x:
        return "n/a"
    return f"{x:.{decimals}f}"


def fmt_p(p):
    if p is None or p != p:
        return "n/a"
    if p < 1e-6:
        return "<1e-6"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def sig_stars(p):
    if p is None or p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
