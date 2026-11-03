# ScoLAD — Reproduction Package

[한국어](README.ko.md) · [Repository guide](docs/REPOSITORY_GUIDE.md) · [Execution setup](code/RETARGET.md)

Companion package for *ScoLAD: One Fused Score Detects Both Appearance Defects and Composition Errors*, submitted to the *Journal of Imaging*. The [revision guide](docs/REPOSITORY_GUIDE.md) maps the revised tables to available artifacts. The [supplementary workbook](results/supplementary/ScoLAD_Supplementary_Tables_S1_S2.xlsx) contains Tables S1 and S2 with their definitions. Dated scripts and results retain their original identifiers.

**Main setting — image-level AUROC (L+S) = 0.9723 ± 0.0011** (seeds 42/43/44)
logical 0.9685 · structural 0.9762 · **+2.54 pp** over the released-map SALAD reproduction (0.9469)

---

## 1. Verify the main result

The headline is recomputed from per-image scores alone. No 18 GB feature cache, no 9 GB of
checkpoints, no GPU — `numpy` and `scikit-learn` suffice.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```

> **Windows**: the JSON artifacts contain Korean text, so run under UTF-8 —
> `set PYTHONUTF8=1` (or `python -X utf8 ...`). The scripts pass
> `encoding="utf-8"` explicitly, so this is a safety net rather than a requirement.
> The distributed reader CSV files retain their original line endings. The manifest hashes the checked-out file bytes.

| Seed | L+S | Logical | Structural |
|---|---:|---:|---:|
| 42 | 0.9721 | 0.9662 | 0.9780 |
| 43 | 0.9714 | 0.9682 | 0.9745 |
| 44 | 0.9735 | 0.9710 | 0.9759 |

The verification script standardizes each branch using normal-validation statistics, sums the three scores with equal weights, and reports the mean of logical and structural AUROC. This calculation uses the stored branch scores.

### Why ship scores rather than code alone

For the composition branch, mean Spearman rank correlation was 0.9381 across seeds on the same machine, 0.9378 for an independent rebuild with the same seed on a different machine, and 0.9388 for independent training runs on the same machine with seed 42. Both the run and the machine changed in the second comparison. A separate rebuild comparison yielded AUROC differences of −0.0026 for the composition branch and −0.0007 after fusion. [Rank correlations](results/tables/psad_variance_decomp.json) · [AUROC comparison](results/tables/psad_3way_compare.json)

---

## Environment

The verification in §1 needs only Python 3 with `numpy` and `scikit-learn` (`requirements-verify.txt`).
The revision reports computational-cost measurements on an NVIDIA GeForce RTX 5090 (32 GB). See §7 for the upstream implementations used by the branch scripts.

---

## 2. Method

Reconstruction and patch memory measure appearance deviations with different features. The composition branch compares component area and appearance. More than one branch can respond to the same anomaly.

| Branch | What it measures | Code |
|---|---|---|
| **Reconstruction** — EfficientAD-S | student–teacher feature discrepancies and autoencoder reconstruction for larger-scale deviations | `code/branches/reconstruction_ead/` |
| **Patch memory** — DINOv3-L L17, PatchCore-style | maximum patch distance to a nearest normal-bank neighbor | `code/branches/patch_memory_pc/` |
| **Composition** — PSAD-hc | component area histogram (h) + appearance embedding (c) | `code/branches/composition_psad/` |

The composition segmenter is trained on masks prepared from the labels released by CSAD. For pushpins, each foreground component is assigned to the cell containing its centroid in a 3×5 tray grid. Two postprocessing steps are applied to segmentation outputs.

- In the row-normalized confusion matrix between normal-training labels and predictions, candidate pairs must each have at least 1,000 labeled pixels per class and a summed bidirectional confusion rate above 0.10. Only screw bag classes c4 and c5 qualify (0.168/0.261/0.142 across three seeds, next pair ≤0.012).
- Screw bag alone met the criterion for rotation averaging, a decrease in both the mean and maximum validation-normal tail statistic across seeds. Predictions at 0°, 90°, 180° and 270° are rotated back and averaged as probability maps, followed by per-pixel class selection and c4/c5 merging. This procedure is used for the training bank and for validation and test scoring.

---

## 3. Recalculate selected paper tables

| From the package root | Scope |
|---|---|
| `python3 code/analysis/reproduce_from_package.py --pkg .` | Three headline means and sample standard deviations, plus the L+S means for seven Table 2 baselines, from packaged image scores. The DINOv2-B, DINOv3-B and full CSAD rows have stored summaries. |
| `python3 code/analysis/reproduce_tables_5_6.py --pkg .` | Tables 5–6, including the OR rules evaluated using test labels at required recall levels. |
| `python3 code/analysis/verify_supplementary_scores.py --pkg .` | S1 operating results and S2 selected cases, using the score versions prepared for the supplement. |

See the [table-by-table guide](docs/REPOSITORY_GUIDE.md) for input files, averaging rules, results supplied as summaries and historical analyses. The [score index](results/per_image_scores/README.md) identifies the branch, split and seeds in each directory.

---

## 4. What is not included, and why

| Item | Size | How to regenerate |
|---|--:|---|
| DINOv3-L feature cache | ~18 GB | `code/branches/patch_memory_pc_extra/I_dinov3_sl_cache.py` |
| U-Net segmenter checkpoints | ~9 GB | `code/branches/composition_psad/train_unet_seeds.sh` |
| EfficientAD training weights | ~2 GB | `code/branches/reconstruction_ead/train_ead_seeds.sh` |
| MVTec LOCO AD dataset | ~6 GB | from MVTec Software GmbH |
| DINOv3-L/16 weights | — | Hugging Face `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated license; accept the terms, then download) |
| Rejected exploratory PSAD variants | — | flags in `score_psad.py` |

**Per-image scores are included**, because they are sufficient for the verification in §1.
The bulk items above are needed only to rebuild those scores from scratch — and, per §1,
rebuilding does not reproduce them exactly.

Not available: the human-annotation variant (5 categories × 3 seeds were never produced, so
the corresponding column was removed from the paper).

---

## 5. Running the scripts

The score-recalculation command in Section 1 reads the package directory specified by
`--pkg`; it requires neither a GPU nor the original training workspace.

Training, feature-extraction and some analysis scripts retain paths and imports from the
original research workspace. Running them requires configuring the dataset, checkpoint,
cache and upstream-code locations for each script. Changing only the repository-root
string does not recreate those dependencies. See [execution setup](code/RETARGET.md).

---

## 6. Integrity

The [manifest](MANIFEST.md) records the size, origin and SHA-256 prefix of each packaged file except itself. The scripts in `code/audit/` document the checks performed on the original submission. The current reader-table files are listed separately in the manifest.

## License

MIT for our code and derived artifacts — see `LICENSE`. Third-party components (dataset, model
weights, upstream implementations) remain under their own licenses and are not redistributed;
they are listed in `THIRD_PARTY_NOTICES.md`.

## 7. Upstream implementations

The reconstruction branch follows a public third-party reimplementation of EfficientAD (its README states
that it is unofficial), the composition branch follows PSAD's public implementation, and the component
pseudo-labels are those released in CSAD's official repository. Pinned commits (component names are
local folder names):

| Component | Repository | Commit |
|---|---|---|
| `efficient_ad_official` | https://github.com/nelson1425/EfficientAD.git | `fcab514` |
| `PSAD_official` | https://github.com/oopil/PSAD_logical_anomaly_detection.git | `1ae3146` |
| `CSAD_official` | https://github.com/Tokichan/CSAD.git | `d46cf85` |

Only derived artifacts are redistributed here; the upstream code and any model
weights must be obtained from those repositories.
