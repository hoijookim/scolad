"""Verify the S1/S2 reader exports from archived, per-image scores (CPU only)."""
import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")
SEEDS = (42, 43, 44)
METHODS = ("ScoLAD", "SALAD_released_maps_0.9469", "SALAD_regenerated_maps_0.9340")
BRANCHES = ("reconstruction", "patch", "composition")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(row, field, expected):
    actual = row[field]
    if isinstance(expected, (bool, np.bool_)):
        matches = actual == str(bool(expected)).lower()
    elif isinstance(expected, str):
        matches = actual == expected
    else:
        matches = math.isclose(float(actual), expected, rel_tol=0, abs_tol=2e-12)
    if not matches:
        raise ValueError(f"{row['record_id']} {field}: expected {expected}, found {actual}")


def unique_index(rows, key):
    by_key = {key(row): row for row in rows}
    if len(rows) != len(by_key):
        raise ValueError("Duplicate reader-table key")
    return by_key


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def cell_scores(root, category, seed, salad):
    scores = root / "per_image_scores"
    test_r = npz(scores / ("ead_s_seed42_repro" if seed == 42 else "ead_s") /
                 f"scores_{category}_seed{seed}.npz")
    val_r = npz(scores / "ead_s_val" / f"val_good_{category}_seed{seed}.npz")
    patch = npz(scores / "pc_branch" / f"pc_L17_{category}_seed{seed}.npz")
    comp = scores / "psad_composition"
    test_c = npz(comp / f"psad_scores_{category}_seed{seed}_hc_tta_merge_test.npz")
    val_c = npz(comp / f"psad_scores_{category}_seed{seed}_hc_tta_merge_val.npz")
    paths = test_c["paths"].tolist()
    val_paths = val_c["paths"].tolist()
    axis = np.array([p.split("/")[1].replace("_anomalies", "") for p in paths])
    if not np.array_equal(axis, test_r["label_type"]):
        raise ValueError(f"{category}/{seed}: inconsistent test labels")
    if [p.split("/")[-1] for p in val_paths] != val_r["fnames"].tolist():
        raise ValueError(f"{category}/{seed}: inconsistent validation IDs")
    if len(salad) != len(paths) + len(val_paths) or set(salad) != set(paths + val_paths):
        raise ValueError(f"{category}/{seed}: inconsistent SALAD sample IDs")
    test = np.column_stack((test_r["score"], patch["test"], test_c["scores"]))
    val = np.column_stack((val_r["scores"], patch["val"], val_c["scores"]))
    mu, sigma = val.mean(axis=0), val.std(axis=0)
    test_z = (test - mu) / sigma
    val_z = (val - mu) / sigma
    test_all = np.column_stack((test_z.sum(axis=1), [salad[p] for p in paths]))
    val_all = np.column_stack((val_z.sum(axis=1), [salad[p] for p in val_paths]))
    return paths, axis, test_z, val_z, test_all, val_all


def verify_s1(row, scores, axis, threshold):
    alarms = scores > threshold
    normal = axis == "good"
    positive = axis == row["anomaly_axis"]
    tp = (alarms & positive).sum()
    fn = positive.sum() - tp
    fp = (alarms & normal).sum()
    tn = normal.sum() - fp
    for field, value in (
        ("fixed_validation_threshold_score", threshold), ("TP_count", tp), ("FN_count", fn),
        ("shared_normal_FP_count", fp), ("shared_normal_TN_count", tn),
        ("recall_fraction", tp / (tp + fn)), ("FNR_fraction", fn / (tp + fn)),
        ("axis_precision_fraction", tp / (tp + fp)), ("shared_normal_FPR_fraction", fp / (fp + tn)),
    ):
        check(row, field, value)


def verify_s2(row, axis, branch, branch_thr, combined, thresholds, seed):
    check(row, "anomaly_type", axis)
    check(row, "illustrated_seed43", seed == 43)
    for j, name in enumerate(BRANCHES):
        check(row, f"ScoLAD_{name}_z_score", branch[j])
        check(row, f"ScoLAD_{name}_own_threshold_5pct_budget", branch_thr[j])
        check(row, f"ScoLAD_{name}_own_alarm_5pct_budget", branch[j] > branch_thr[j])
    for j, name in enumerate(("ScoLAD", "SALAD_released_maps", "SALAD_regenerated_maps")):
        score_field = "ScoLAD_fused_score" if j == 0 else f"{name}_combined_score"
        th_field = "ScoLAD_fixed_fused_threshold_5pct_budget" if j == 0 else f"{name}_fixed_threshold_5pct_budget"
        decision = "ScoLAD_fusion_alarm_5pct_budget" if j == 0 else f"{name}_alarm_5pct_budget"
        check(row, score_field, combined[j])
        check(row, th_field, thresholds[.05][j])
        check(row, decision, combined[j] > thresholds[.05][j])
    check(row, "ScoLAD_auxiliary_fusion_alarm_10pct_budget_threshold_unavailable", combined[0] > thresholds[.10][0])
    check(row, "SALAD_internal_branch_scores_unavailable", "")


def verify(package):
    root = package / "results"
    supp = root / "supplementary"
    with gzip.open(supp / "salad_operating_scores.csv.gz", "rt", newline="") as stream:
        paired = defaultdict(dict)
        for r in csv.DictReader(stream):
            cell = paired[r["category"], int(r["seed"])]
            path = r["dataset_path"]
            if path in cell:
                raise ValueError(f"Duplicate score for {path}")
            cell[path] = (float(r["released_map_score"]), float(r["regenerated_map_score"]))
    s1 = unique_index(read_csv(supp / "Table_S1_operating_results_reader.csv"), lambda r: (
        r["category"], int(r["seed"]), r["method_lineage"],
        float(r["nominal_false_alarm_budget_fraction"]), r["anomaly_axis"]))
    s2 = unique_index(read_csv(supp / "Table_S2_selected_cases_reader.csv"), lambda r: (r["case"], int(r["seed"])))
    if set(s2) != {(c, s) for c in "abcdefghij" for s in SEEDS}:
        raise ValueError("Expected S2 cases a–j for every seed")
    for letter in "abcdefghij":
        if len({(s2[letter, s]["category"], s2[letter, s]["dataset_test_path"], s2[letter, s]["subtype"]) for s in SEEDS}) != 1:
            raise ValueError(f"Case {letter} has inconsistent metadata across seeds")
    seen_s1 = seen_s2 = 0
    for cat in CATEGORIES:
        for seed in SEEDS:
            paths, axis, test_z, val_z, scores, validation = cell_scores(root, cat, seed, paired.pop((cat, seed)))
            thresholds = {b: np.quantile(validation, 1 - b, axis=0) for b in (.05, .10)}
            branch_thr = np.quantile(val_z, .95, axis=0)
            for j, method in enumerate(METHODS):
                for budget, th in thresholds.items():
                    for kind in ("logical", "structural"):
                        row = s1.pop((cat, seed, method, budget, kind))
                        verify_s1(row, scores[:, j], axis, th[j])
                        seen_s1 += 1
            for (letter, n), row in s2.items():
                if n == seed and row["category"] == cat:
                    check(row, "record_id", f"S2-{letter}-{n}")
                    i = paths.index(row["dataset_test_path"])
                    verify_s2(row, axis[i], test_z[i], branch_thr, scores[i], thresholds, seed)
                    seen_s2 += 1
    if paired or s1 or seen_s1 != 180 or seen_s2 != 30:
        raise ValueError("Unexpected combination of rows, categories or seeds")
    print(f"Verified {seen_s1} S1 operating rows and {seen_s2} S2 selected-case score/decision rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Root of the companion repository")
    verify(parser.parse_args().pkg)
