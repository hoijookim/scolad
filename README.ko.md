# ScoLAD — 재현 패키지

[English](README.md) · [저장소 구성 안내](docs/REPOSITORY_GUIDE.md) · [실행 환경 안내](code/RETARGET.md)

*Journal of Imaging*에 제출한 논문 *ScoLAD: One Fused Score Detects Both Appearance Defects and Composition Errors*의 동반 패키지입니다. [개정본 안내](docs/REPOSITORY_GUIDE.md)에서 현재 표·부록과 파일의 관계를 확인할 수 있습니다. [보충 워크북](results/supplementary/ScoLAD_Supplementary_Tables_S1_S2.xlsx)에는 표 S1·S2와 열 정의가 들어 있습니다. 과거 분석 파일은 원래 이름을 유지합니다.

**주 설정 — image-level AUROC (L+S) = 0.9723 ± 0.0011** (시드 42/43/44)
logical 0.9685 · structural 0.9762 · 공개 지도 SALAD 재현(0.9469) 대비 **+2.54%p**

---

## 1. 주 결과 검증

per-image 점수만으로 헤드라인이 재계산된다. 18GB 특징 캐시도, 9GB 체크포인트도, GPU도
필요 없다 — `numpy` 와 `scikit-learn` 이면 된다.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```

> **Windows**: JSON 산출물에 한국어가 들어 있어 UTF-8 로 돌려야 한다 —
> `set PYTHONUTF8=1`(또는 `python -X utf8 ...`). 스크립트가 `encoding="utf-8"` 를
> 명시하므로 필수는 아니고 안전망이다. 독자용 CSV는 원래 줄바꿈을 유지하며 파일 목록의 해시는 실제 파일 바이트를 기준으로 합니다.

| 시드 | L+S | 논리 | 구조 |
|---|---:|---:|---:|
| 42 | 0.9721 | 0.9662 | 0.9780 |
| 43 | 0.9714 | 0.9682 | 0.9745 |
| 44 | 0.9735 | 0.9710 | 0.9759 |

### 왜 점수를 배포하는가 — 코드만으로는 부족하다

구성 분기의 점수 순위를 세 가지 재학습 조건에서 비교했다. 범주에 걸쳐 집계한 Spearman 상관은 같은 기계의 서로 다른 시드에서 0.9381, 같은 시드를 다른 기계에서 독립적으로 재구축했을 때 0.9378, 같은 기계·같은 시드 42에서 독립적으로 다시 학습했을 때 0.9388이었다. 두 번째 비교에서는 기계와 실행이 함께 바뀌었다. 별도의 재구축 비교에서 AUROC 차이는 구성 분기 −0.0026, 융합 후 −0.0007이었다. [순위 상관](results/tables/psad_variance_decomp.json) · [AUROC 비교](results/tables/psad_3way_compare.json)

검증 스크립트는 저장된 세 분기 점수를 정상 검증 표본의 통계로 표준화하고 등가중치로 합산한 뒤 논리·구조 AUROC의 평균을 보고합니다.

---

## 환경

§1 의 검증에는 `numpy` 와 `scikit-learn` 이 있는 Python 3 만 있으면 된다(`requirements-verify.txt`).
개정 원고의 계산 비용 측정에는 NVIDIA GeForce RTX 5090(32 GB)을 사용했습니다. 분기별 기반 구현은 §7에서 확인할 수 있습니다.

---

## 2. 방법

재구성과 패치 메모리는 서로 다른 특징으로 외관 이상을 측정하고, 구성 분기는 부품의 면적과 외형을 비교한다. 한 이상에 여러 분기가 반응할 수 있다.

| 분기 | 무엇을 보나 | 코드 |
|---|---|---|
| **재구성** EfficientAD-S | 학생–교사 특징 차이와 오토인코더 복원을 통한 더 큰 규모의 이탈 | `code/branches/reconstruction_ead/` |
| **패치 메모리** DINOv3-L L17, PatchCore 방식 | 각 패치의 정상 뱅크 최근접 거리 중 최댓값 | `code/branches/patch_memory_pc/` |
| **구성** PSAD-hc | 부품 면적 히스토그램(h) + 외형 임베딩(c) | `code/branches/composition_psad/` |

구성 분할기는 CSAD가 공개한 의사 레이블에서 준비한 마스크로 학습한다. pushpins에서는 정상 마스크의 연결 성분을 중심점이 속한 3×5 트레이 칸에 배정한다. 분할 후 다음 두 처리를 적용했다.

- 정상 훈련 레이블과 예측의 행 정규화 혼동행렬에서 두 클래스의 픽셀이 각각 1,000개 이상이고 양방향 혼동률의 합이 0.10을 넘는 쌍을 찾았다. screw bag의 c4·c5만 기준을 충족했다(세 시드 0.168/0.261/0.142, 다음 쌍 ≤0.012).
- 회전 평균은 세 시드에 걸쳐 정상 검증 점수 꼬리 통계의 평균과 최댓값이 모두 감소한 screw bag에 채택했다. 0·90·180·270도 예측 확률맵을 역회전해 평균하고 픽셀별 최상위 클래스를 고른 뒤 c4·c5를 병합한다. 훈련 뱅크, 검증 및 테스트 채점에 같은 처리를 사용한다.

---

## 3. 선택한 원고 표 재계산

| 패키지 루트에서 실행할 명령 | 확인 범위 |
|---|---|
| `python3 code/analysis/reproduce_from_package.py --pkg .` | 동봉된 이미지별 점수에서 주 융합의 평균·표본 표준편차 3쌍과 표 2 기준선 7개의 L+S 평균을 재계산한다. DINOv2-B·DINOv3-B·전체 CSAD 행은 저장된 요약으로 확인한다. |
| `python3 code/analysis/reproduce_tables_5_6.py --pkg .` | 테스트 정답으로 요구 재현율을 맞추는 표 5·6의 진단값을 OR 규칙까지 재계산한다. |
| `python3 code/analysis/verify_supplementary_scores.py --pkg .` | 보충자료에 사용한 점수 버전으로 S1 운영 결과와 S2 선정 사례를 확인한다. |

[표별 안내](docs/REPOSITORY_GUIDE.md)에서 입력 파일·평균 방식·요약만 제공하는 결과·이전 분석을 구분했다. [이미지별 점수 색인](results/per_image_scores/README.md)에서 폴더별 분기·분할·시드를 확인할 수 있다.

---

## 4. 담기지 않은 것과 그 이유

| 항목 | 크기 | 재생성 |
|---|--:|---|
| DINOv3-L 특징 캐시 | ~18 GB | `code/branches/patch_memory_pc_extra/I_dinov3_sl_cache.py` |
| UNet 분할 체크포인트 | ~9 GB | `code/branches/composition_psad/train_unet_seeds.sh` |
| EfficientAD 학습 가중치 | ~2 GB | `code/branches/reconstruction_ead/train_ead_seeds.sh` |
| MVTec LOCO AD 데이터셋 | ~6 GB | 원저작권자 배포 (MVTec) |
| PSAD 구성분기의 기각된 탐색 변형 점수 | — | `score_psad.py` 의 해당 플래그 |

**per-image 점수는 담았다** — 이것만으로 §1 의 검증이 되기 때문이다. 위 대용량은 그 점수를
처음부터 다시 만들려 할 때만 필요하다.

**ComAD 시드별 표준편차는 260831 에 해소했다** — 이전 판은 "per-image 점수가 이 머신에
없다" 고 적었으나 있었다(`external/ROMAD_baselines/ComAD/output_scores_seed{42,43,44}/`,
시드당 1,568개). `results/per_image_scores/comad/` 15개가 그것을 정본 순서로 묶은 것이고,
`results/tables/comad_axis_perseed.json` 이 축별 값이다. 산출 전에 재현부터 검증했다 —
기존 3시드 기록과 pooled 15조합, 정본과 축별 10칸 모두 **최대 절대차 0**.

산출 불가로 남은 것: human-annotation 변형(5범주×3시드가 없어 논문에서 해당 열을 제거했다).

**PUAD-S 시드 43 은 260821 에 해소했다** — 학습이 아니라 재채점으로 풀리는 문제였다.
`results/per_image_scores/puad_s/` 15개가 그 결과다.

---

## 5. 스크립트 실행

저장된 점수의 재계산은 `--pkg`로 지정한 패키지 폴더를 사용하며, GPU나 원래 학습 서버가
필요하지 않다.

학습·특징 추출 및 일부 분석 스크립트에는 원래 연구 환경의 경로와 import가 남아 있다.
실행하려면 스크립트별로 데이터셋·체크포인트·캐시·상류 코드의 위치를 설정해야 한다.
저장소 루트 문자열만 바꿔서는 이 의존성이 준비되지 않는다. [실행 환경 안내](code/RETARGET.md)를 참고한다.

---

## 6. 무결성과 감사

[파일 목록](MANIFEST.md)은 목록 자체를 제외한 파일별 출처, 크기, SHA-256 앞 16자리를 기록한다. 독자용 보충표도 이 목록에 포함한다.

날짜가 붙은 `docs/`의 문서는 당시 분석과 정정 이력이다. 예를 들어 7월의 표 번호와 수치는 뒤의 감사에서 갱신되었다. 원본 조립·감사 스크립트는 당시 연구 작업공간과 그 입력을 요구한다. 자세한 범위는 [실행 안내](code/RETARGET.md)를 따른다.

### 데이터

MVTec LOCO AD (Bergmann et al., IJCV 2022). 학술 목적 무상 배포이며 재배포는 하지 않는다.
주 설정의 패치 뱅크는 공식 정상 훈련 표본을 쓰며, 분기 점수의 표준화와 운영 임계값에는 정상 검증 표본을 쓴다. 부록 C의 정상 참조 특징과 표 5·6의 테스트 정답 기반 진단은 [표별 안내](docs/REPOSITORY_GUIDE.md)에 별도로 설명했다.

## 라이선스

우리 코드와 파생 산출물은 MIT — `LICENSE` 참고. 제3자 구성 요소(데이터셋, 모델 가중치, 상류 구현)는
각자의 라이선스를 따르며 재배포하지 않는다. 목록은 `THIRD_PARTY_NOTICES.md`.

## 7. 상류 구현

재구성 분기는 EfficientAD 의 공개 제3자 재구현(README 에 비공식이라고 명시)을 따르고, 구성 분기는 PSAD 의 공개 구현을 따르며, 부품 의사 레이블은 CSAD 공식 저장소가 공개한 것을 쓴다. 고정 커밋(구성 요소 이름은 로컬 폴더 이름):

| 구성 요소 | 저장소 | 커밋 |
|---|---|---|
| `efficient_ad_official` | https://github.com/nelson1425/EfficientAD.git | `fcab514` |
| `PSAD_official` | https://github.com/oopil/PSAD_logical_anomaly_detection.git | `1ae3146` |
| `CSAD_official` | https://github.com/Tokichan/CSAD.git | `d46cf85` |

여기에는 파생 산출물만 담는다. 상류 코드와 모델 가중치는 위 저장소에서 받아야 한다.
