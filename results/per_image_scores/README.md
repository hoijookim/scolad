# per-image 점수 — 무엇을 뒷받침하나

각 npz 는 이미지 단위 이상 점수다. 이 파일들만으로 헤드라인이 재계산된다 (`code/analysis/reproduce_from_package.py`). 캐시·체크포인트는 필요 없다.

| 디렉터리 | 파일 | 시드 | 용도 |
|---|--:|---|---|
| `comad` | 15 | 42, 43, 44 | 표 2 ComAD 테스트 점수. |
| `ead_m_imagenette` | 15 | 42, 43, 44 | EAD-M 의 imagenette 폴백 학습본. 이전 조건(0.7846)의 점수. 현행 표 2의 full-ImageNet EAD-M 점수는 puad_m의 ead 배열에 있다. |
| `ead_s` | 25 | 0, 42, 43, 44, 1234 | EAD-S 재구성 분기 test 점수. 표 2의 EAD-S 행과 융합의 첫째 항. |
| `ead_s_seed42_repro` | 5 | 42 | seed42 재학습본 test 점수. 정본 seed42 는 원본 모델 산출물이라 val(재학습본)과 출처가 섞였다. 융합은 이 파일을 쓴다. |
| `ead_s_val` | 15 | 42, 43, 44 | EAD-S 검증분할 정상 표본 점수. z-정규화의 기준값 — 없으면 융합이 재현되지 않는다. |
| `original_pipeline_reference` | 50 | 0, 42, 1234 | 원본 PSAD/PatchCore 파이프라인 참조 점수 (재구축 대조군). |
| `pc_branch` | 15 | 42, 43, 44 | DINOv3-L L17 PatchCore 분기 (coreset 50000 / k=1 / max / train-only 뱅크). 표 2의 DINOv3-L 행과 융합의 둘째 항이 같은 값이다. |
| `psad_composition` | 60 | 42, 43, 44 | PSAD 구성분기. hc_tta_merge = 주 설정, hcp_tta_merge_k5는 보관된 비교 설정. |
| `puad_m` | 15 | 42, 43, 44 | ead 열은 표 2의 full-ImageNet EfficientAD-M 점수이며, puad 열은 그 모델에 공식 Mahalanobis 점수를 결합한 PUAD-M 점수다. 기준 파일에서 이상 유형을 가져온다. 파일 길이와 정상/이상 라벨이 일치해도 같은 그룹 내 이미지 순서까지 독립적으로 검증되지는 않는다. |
| `puad_s` | 15 | 42, 43, 44 | PUAD-S 재현 (EAD-S 체크포인트 + 공식 Mahalanobis, 학습 없이 재채점). 구 산출물은 시드마다 체크포인트 루트가 달라 셋 다 `_seeds_std` 에서 다시 매겼다. |
| `salad` | 15 | 42, 43, 44 | 공개 지도 SALAD의 주 AUROC용 점수. [보충자료 점수 출처](../supplementary/SCORE_SOURCES.md)에서 S1/S2의 입력을 구분한다. |

## 시드에 관해

주 표는 **{42, 43, 44}**를 사용한다. 시드 45·46은 [EAD-S 5시드 확장 검증의 요약](../reproduction/ead_5seed_verdict.json)에 기록되어 있다. 이 폴더에 동봉한 EAD-S 개별 테스트 점수 파일의 시드는 0·42·43·44·1234이다. 0·1234는 원본 PSAD 시드 프로토콜과 맞춘 대조에 쓰였다.

## 담지 않은 것

PSAD 구성분기의 기각된 탐색 변형(k=5·개수거리·소프트마스크 등 14계열)은 제외했다. 기각 근거는 `results/diagnostics/PREREG_branch_improvement.md` 와 `sweep_*.json` 에 있고, 재실행에 필요한 외부 코드·데이터·체크포인트는 [실행 안내](../../code/RETARGET.md)에서 확인할 수 있다.
