# Third-party components

Research artifacts in this package include derived anomaly scores, aggregate measurements and reader tables. Table S2 references dataset-relative image paths.
No third-party model weights or datasets are included. The pipeline builds on:

- EfficientAD — reconstruction branch; the package follows the public, unofficial
  [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD/tree/fcab514)
  reimplementation pinned in README §7. The local folder name
  `efficient_ad_official` is not an authorship or official-status claim.
- PatchCore — patch-memory branch formulation
- DINOv3-L/16 (Meta) — frozen backbone; distributor gated license applies
- PSAD — composition branch; public
  [oopil/PSAD_logical_anomaly_detection](https://github.com/oopil/PSAD_logical_anomaly_detection/tree/1ae3146)
  implementation pinned in README §7.
- CSAD (Tokichan/CSAD, commit d46cf85) — component pseudo-labels
- MVTec LOCO AD (MVTec Software GmbH) — dataset, redistributed by the original
  rights holder only; not included here

Each component remains under its own license. Users must obtain the dataset and
model weights from their original distributors.
