"""Recalculate the published test-ROC diagnostics from the packaged per-image scores.

Both tables require the indicated recall on each anomaly axis. Table 5 reports
fractions, Table 6 percentages. These are test-label-conditioned diagnostics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reproduce_from_package import CATS, SEEDS, load


TABLE5_TARGETS = (80, 85, 90, 95)
TABLE6_TARGETS = (90, 95, 99)


def single_fpr(scores, types, target):
    thresholds = []
    for axis in ("logical", "structural"):
        values = np.sort(scores[types == axis])
        # Integer ceiling avoids roundoff at an exact recall boundary.
        required = (len(values) * target + 99) // 100
        thresholds.append(values[len(values) - required])
    return float(np.mean(scores[types == "good"] >= min(thresholds)))


def or_fprs(branches, types):
    normal = types == "good"
    n = int(normal.sum())
    # Retain the complement operation used by the original rank-grid search.
    # Linear interpolation followed by strict > is sensitive to floating ties.
    if n < 2:
        raise ValueError("The shared OR grid requires at least two normal images.")
    tail = 1.0 - np.arange(n) / (n - 1)
    thresholds = np.quantile(branches[:, normal], 1.0 - tail, axis=1)
    alarms = (branches.T[None, :, :] > thresholds[:, None, :]).any(axis=2)
    min_recall = np.minimum(alarms[:, types == "logical"].mean(axis=1),
                            alarms[:, types == "structural"].mean(axis=1))
    fprs = alarms[:, normal].mean(axis=1)
    results = {}
    for target in TABLE5_TARGETS:
        feasible = fprs[min_recall >= target / 100]
        if not feasible.size:
            raise ValueError(f"No shared OR grid level attains {target}% recall on both axes.")
        results[target] = float(feasible.min())
    return results


def calculate(pkg):
    rows = {}
    for seed in SEEDS:
        for cat in CATS:
            *branches, labels, types = load(pkg, cat, seed)
            if not np.array_equal(labels, types != "good") or set(types) != {"good", "logical", "structural"}:
                raise ValueError(f"Invalid labels for {cat}, seed {seed}")
            z = []
            for test, val in branches:
                if (test.shape != labels.shape or val.ndim != 1 or not np.isfinite(test).all()
                        or not np.isfinite(val).all() or not val.size or val.std() == 0):
                    raise ValueError(f"Invalid test/validation scores for {cat}, seed {seed}")
                z.append((test - val.mean()) / val.std())
            reconstruction, patch, composition = z
            fusion = reconstruction + patch + composition
            p = pkg / "results/per_image_scores/salad" / f"{cat}_seed{seed}.npz"
            with np.load(p, allow_pickle=True) as salad:
                if not np.array_equal(labels, salad["labels"]) or not np.array_equal(types, salad["label_types"]):
                    raise ValueError(f"SALAD labels differ for {cat}, seed {seed}")
                released = salad["combined"]
            if released.shape != labels.shape or not np.isfinite(released).all():
                raise ValueError(f"Invalid SALAD scores for {cat}, seed {seed}")
            scores = {"Reconstruction alone": reconstruction, "Patch memory alone": patch,
                      "Composition alone": composition, "R+P+C sum": fusion,
                      "R+C sum": reconstruction + composition}
            table5 = {name: {t: single_fpr(sc, types, t) for t in TABLE5_TARGETS}
                      for name, sc in scores.items()}
            table5["R∨C OR"] = or_fprs(np.stack([reconstruction, composition]), types)
            table5["R∨P∨C OR"] = or_fprs(np.stack(z), types)
            table6 = {name: {t: single_fpr(sc, types, t) for t in TABLE6_TARGETS}
                      for name, sc in (("ScoLAD", fusion), ("SALAD", released))}
            rows[seed, cat] = {"table5": table5, "table6": table6}

    result = {}
    for table, unit, digits in (("table5", 1, 3), ("table6", 100, 2)):
        result[table] = {}
        for name, targets in rows[SEEDS[0], CATS[0]][table].items():
            result[table][name] = {}
            for target in targets:
                per_seed = [unit * np.mean([rows[seed, c][table][name][target] for c in CATS])
                            for seed in SEEDS]
                result[table][name][str(target)] = {
                    "mean": round(float(np.mean(per_seed)), digits),
                    "sample_sd": round(float(np.std(per_seed, ddof=1)), digits),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="write the recomputed cells after verification")
    args = parser.parse_args()
    reference = args.pkg / "results/tables/tables_5_6.json"
    try:
        expected = json.loads(reference.read_text(encoding="utf-8"))
        result = calculate(args.pkg)
        if result != expected:
            raise ValueError(f"Computed cells differ from {reference}")
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError, IndexError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print("PASS: Table 5 (28 mean/SD cells, FPR fraction) and Table 6 "
          "(6 mean/SD cells, FPR percent) match at published precision.")


if __name__ == "__main__":
    main()
