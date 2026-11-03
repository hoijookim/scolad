# Execution setup

## Recalculate archived scores

From the package root, run:

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```

This command uses the packaged per-image scores and requires Python with NumPy and
scikit-learn (`requirements-verify.txt`). It does not require a GPU, model checkpoints
or the original research workspace.

## Configure training and inference

Training, feature-extraction and some analysis scripts retain paths such as
`/workspace/ai-vision-research` and imports from the original research workspace.
Before running an individual script:

1. Obtain the datasets, model weights and upstream implementations it requires.
2. Configure its input, output, cache and import paths for the local directory layout.
3. Check that all referenced files and modules are available.

For example, `code/fusion/prereg_testfree.py` references the original
`scripts/psad_rebuild/`, `cache/` and `external/` directories. The public package uses
`code/` and `results/` and does not include all external assets. Replacing a root-path
string alone is therefore insufficient to make every script runnable.

See the package README and `THIRD_PARTY_NOTICES.md` for asset and license information.

## Inputs by task

| Task | Inputs and interpretation |
|---|---|
| Tables 5–6 | Run `python3 code/analysis/reproduce_tables_5_6.py --pkg .` from the package root. It uses stored branch scores and the revised threshold rules. Earlier requirement-curve and operating-point scripts target the original workspace's submission/results tree. |
| S1 and S2 | Run `python3 code/analysis/verify_supplementary_scores.py --pkg .`. The reader CSVs and the [supplementary score source](../results/supplementary/SCORE_SOURCES.md) identify the table-generation scores. |
| Patch-feature extraction | The bundled DINOv3-S/L extractor takes positional variant `L` for the main backbone. Its default is `S`. Obtain the matching pretrained model and MVTec LOCO images, configure cache and dataset paths, and verify per-category split lengths and feature keys against the input list before using a cache. The five-backbone comparison also requires caches for DINOv3-B, DINOv2-B and DINOv2-L. |
| Composition scoring | The original workspace supplies PSAD's modules, the `scripts/psad_rebuild/_shims/pl_compat` compatibility module, prepared CSAD labels, segmenter weights, RGB images and PatchCore per-image .pt inputs read by the scoring script. Pushpins mask preparation uses connected-component centroids in a 3×5 grid before resizing. The main scoring configuration enables both `--tta` and `--merge`; the rotation rule was adopted only for screw bag. |
| EfficientAD training | Obtain the upstream implementation, teacher weights and training images. The full-ImageNet penalty set is required to reproduce the current EAD-M condition. An imagenette fallback gives a different baseline. |
| C1 and C2 | The table scripts read multi-layer feature caches, LOCO defect configurations and subtype ground truth. They import the STRAND assignments from `scripts/countgd/footprint_resolved_allcat.py` in the original workspace. C1 also reads a canonical category score file for its label-type fallback. The stored JSONs can be read without the feature caches. |
| Research training, audits and package assembly | Shell launchers and workspace imports assume the original tree. Confirm that each training command completed and that each requested seed produced its checkpoint and score files. Building the package requires the historical reports and source data referenced by the builder. A package checkout alone supplies the offline verification commands above. |

The main paper setting uses three separate model seeds (42, 43 and 44), validation-mean/standard-deviation normalization of raw scores and an equal-weight sum. The archived decision-rule narrative includes an exploratory robust-normalization comparison. For performance from another machine, record its software and model versions separately; the lightweight requirements file covers score analysis only.

For the paired SALAD inputs, the supplemental verifier uses stored dataset paths. In the historical baseline archives, some arrays have labels without paths. The checks on lengths, classes and binary label order cannot distinguish a permutation within one class. Consult the [score index](../results/per_image_scores/README.md) when using per-image comparisons.

The current integrity audit checks the manifest and Git tracking in an actual checkout. A directory copied without its Git metadata can still be used for numerical score checks. Historical audit and package-building scripts may instead require the original checkout, reports or intermediate outputs. The paper's main results use validation normalization and a train-only patch bank; the 24-layer study and the appendix readout comparisons have their own inputs and selection rules.
