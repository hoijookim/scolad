# Repository guide

This guide links the *Journal of Imaging* revision to the companion files. The [English README](../README.md) and [Korean README](../README.ko.md) describe score verification.

## Directory map

| Location | Contents | Use |
|---|---|---|
| `code/analysis/` | Score-based analyses, package verification and package-building scripts | Start with the verification command documented in the README; other scripts may require the original workspace. |
| `code/branches/` | Reconstruction, patch-memory and composition branch scripts | Training and inference need external data, weights and configured paths. |
| `code/fusion/` | Score normalization and fusion scripts | Check each script's inputs and imports before use. |
| `code/baselines/` | Baseline alignment and diagnostic scripts | Retained research scripts; not a single portable training entry point. |
| `code/selection_criteria/` | Selection criteria and exploratory comparisons | Historical design evidence, separate from the fixed main configuration. |
| `code/localization/` | Localization analyses | Historical supplementary analyses; old appendix labels are retained. |
| `code/audit/` | Research and reproducibility audits | Check the original-workspace dependencies before use. |
| `code/shared_modules/` | Shared scoring and evaluation helpers | Used by individual research scripts. |
| `results/` | Archived per-image scores and JSON/Markdown results | See the per-image score index for directory roles and seed coverage. |
| `docs/` | Dated research notes | Read as historical evidence with their original scope and dates. |

## Entry points

- [Execution setup](../code/RETARGET.md): score verification versus rebuilding branches.
- [Per-image score index](../results/per_image_scores/README.md): archived score directories and coverage.
- [Manifest](../MANIFEST.md): file paths, origins, sizes and hash prefixes.
- [Third-party notices](../THIRD_PARTY_NOTICES.md): external implementations and assets.

The lightweight score-verification dependencies are listed in
[`requirements-verify.txt`](../requirements-verify.txt). Installing those dependencies
alone does not supply the assets needed for branch training or inference.

## Revised tables and appendices

The [workbook](../results/supplementary/ScoLAD_Supplementary_Tables_S1_S2.xlsx) includes both reader tables and a Definitions sheet. The CSV exports contain the same data. The table numbers below follow the revised manuscript. Older scripts and dated audit logs retain their original numbering.

| Manuscript location | Companion material and scope |
|---|---|
| Table 1 and Algorithm 1 | Branch settings and inference procedure. [Execution setup](../code/RETARGET.md) lists the datasets, masks, upstream modules and weights used by the research scripts. |
| Table 2 and Appendix B | The [CPU verifier](../code/analysis/reproduce_from_package.py) recomputes the main fusion and seven Table 2 baselines from image scores. [DINOv2-B and DINOv3-B summaries](../results/tables/patchcore_family_trainonly_3seed.json) use L11 and DINOv3-L uses L17. These patch-memory rows score the maximum nearest-neighbor distance against a training-only bank. [Full CSAD scores](../results/tables/csad_ph_lgst_seed42_scores.csv.gz) cover seed 42; [baseline summaries](../results/tables/revision_costs_and_baselines.json) cover all three seeds. B1 summarizes ScoLAD by seed. B2 includes regenerated-map SALAD (0.9340); Table 2 uses the released-map reproduction (0.9469). |
| Tables 3–4 | [Individual branches and fusion](../results/tables/testfree_supplements_3seed.json), [pairs by category](../results/figures/testfree_pairwise_percat_3seed.json) and [axis decomposition](../results/tables/branch_axis_decomposition.json). The paper's paired bootstrap conditions on the fitted models and normalization statistics. |
| Tables 5–6 | The [current verifier](../code/analysis/reproduce_tables_5_6.py) calculates the ROC diagnostics using test labels at fixed recall requirements and checks the [saved results](../results/tables/tables_5_6.json). The [older requirement curve](../results/tables/requirement_cost_curve.json) uses a different threshold rule. |
| Table 7, E1a, E1b and S1 | [S1, 180 rows](../results/supplementary/Table_S1_operating_results_reader.csv), uses thresholds from normal validation scores. Table 7 averages the five category rates equally. E1a sums confusion counts across categories separately for each seed. E1b derives rates from those seed-specific sums. [Supplementary score sources](../results/supplementary/SCORE_SOURCES.md) identify the exact inputs. |
| Tables 8–10 | [Revised cost summaries](../results/tables/revision_costs_and_baselines.json) cover the five timed implementations and the operation estimates. [Parameter counts](../results/tables/param_counts_inference.json) and the [module profile](../results/tables/forward_profile.json) distinguish loaded, executed and trainable parameters. These are recorded results. Repeating the GPU timing requires the implementations, images, weights and matching hardware/software. The June speed study used an earlier configuration. |
| Appendix A | Metric definitions. The archived localization files document earlier analyses. |
| Tables C1–C2 and Figure 8 | [C1 backbone/layer summaries](../results/tables/tableC1_full_grid.json) and [C2 readout summaries](../results/tables/tableC2_true_cardinality.json) use five bank/codebook draws (42–46). C1 selects the highest mean structural test AUROC among evaluated layers for each category/backbone. C2 fixes DINOv3-L L17. The diagnostic scripts use each complete cached normal-feature array. Their cache-generation code pools normal training and validation features. The main Table 2 bank excludes validation rows. The historical cache bytes and sampled patch indices are separate inputs. |
| Figures D1–D2 and S2 | [S2, 30 rows](../results/supplementary/Table_S2_selected_cases_reader.csv), covers ten cases over three seeds. The panels illustrate seed 43. [Score sources](../results/supplementary/SCORE_SOURCES.md) distinguish the SALAD supplement inputs from the archived headline scores. The dataset distributor supplies source images. |
| Table F1 | The [3×5 diagnostic](../results/diagnostics/criterion_v2_csad_pseudo_seg.json) records means and standard deviations for four simulated errors on pushpins. The [comparison note](../results/diagnostics/PREREG_branch_improvement.md) records the four 5×5 means. Both grid structure and component assignment differ. |

## Reading the dated records

- The [July results](260716_v25_numbers_canonical.md) and [August corrections](260820_paper_corrections.md) retain the tables, numbers and baseline conditions in use at the time.
- The [axis-decomposition history](260821_axis_decomposition_results.md) includes earlier logical/mixed groupings as well as the corrected count-axis definition. Use the C1/C2 summaries above for the final comparison.
- The [layer-selection history](260822_l17_basis_audit.md) and the [24-layer diagnostic](260822_layer_full_sweep_result.md) describe different stages. The retrospective test sweep and the normal-validation diagnostic answer different questions.
- The [composition rebuild study](260823_psad_3way_compare.md) concerns the composition segmenter. The [replicate summary](../results/tables/psad_variance_decomp.json) includes the same-machine, fixed-seed repeat that was pending in the earlier investigation.
