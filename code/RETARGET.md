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
