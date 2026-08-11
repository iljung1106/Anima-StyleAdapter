# Per-reference Resampler pilot 결과

- 입력: C-RADIO L18/L24 full spatial + L24 SigLIP teacher CLS
- 출력: 16×768 직접 style token
- 데이터: 1,000 artists × 10 images
- artist split: meta-train 800 / validation 100 / meta-test 100
- 학습: 5,000 steps, 8 artists × 4 images, BF16, fused AdamW
- checkpoint: `l18_l24_native_siglip_l24/checkpoint.pt` (256,537,948 bytes)

## Retrieval과 reconstruction

| Split | 1-ref Top-1 | 2-ref | 4-ref | 8-ref | 평균 Top-1 | L18 rec cosine | L24 rec cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 34.5% | 39.5% | 47.0% | 56.5% | 44.38% | 0.6254 | 0.7147 |
| Meta-test | 31.0% | 44.0% | 54.0% | 59.0% | **47.00%** | 0.6253 | 0.7140 |

Meta-test MRR은 1/2/4/8-reference에서 0.4492/0.5882/0.6770/0.7103이다. Reference 수 증가에 따라 보존한 작가에서도 성능이 일관되게 상승하므로 per-reference token이 multi-reference aggregation에 사용할 수 있는 공통 style 신호를 포함한다.

학습 말기 reconstruction loss는 약 0.33, slot prototype loss는 batch에 따라 약 0.18~0.34였다. 시작부 prototype loss 약 2.07에서 크게 감소했다. Validation과 meta-test reconstruction이 거의 같아 decoder가 train 작가만 암기한 징후는 작다.

## 처리 성능

- 선택된 10,000장 local FP16 cache: 33,296,304,080 bytes
- step time: 대체로 0.083~0.095 s
- data wait: 대체로 0.045~0.060 s/step
- padding efficiency: 대체로 0.80~0.85
- 학습 중 GPU memory: 약 9.9 GB

Production cache는 shape 정렬, sequence reorder, 재사용 pinned A/B buffer와 double buffering으로 약 97 images/s까지 도달했다. 학습은 FP16 cache를 승격하지 않고 container-local subset cache에서 읽으며, 다음 batch H2D를 현재 forward/backward와 겹친다.

W&B run: https://wandb.ai/1wndrla17-kyung-hee-university/anima-style-adapter/runs/per-reference-l18-l24-siglip-l24-v1

## 8,000-step 재학습과 중간 검증

5,000스텝 결과만으로 학습량의 적절성을 판단할 수 없었던 문제를 해결하기 위해 새 seed의 독립 실행을 8,000스텝까지 수행했다. 250스텝마다 고정 episodic validation loss를 계산하고, 500스텝마다 validation 100 artists 전체에 대해 1/2/4/8-reference retrieval과 reconstruction을 평가했다. 각 500스텝 체크포인트를 보존하고 validation 평균 Top-1이 가장 높은 모델을 최종 모델로 선택했다.

| Step | 1-ref | 2-ref | 4-ref | 8-ref | 평균 Top-1 | L18 rec | L24 rec |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 22.5% | 27.5% | 34.5% | 39.5% | 31.00% | 0.5998 | 0.6933 |
| 2,000 | 34.5% | 39.5% | 46.5% | 52.0% | 43.13% | 0.6137 | 0.7056 |
| 3,500 | 34.5% | 49.5% | 52.0% | 58.5% | 48.62% | 0.6199 | 0.7105 |
| 5,000 | 34.0% | 50.5% | 52.5% | 61.5% | 49.62% | 0.6262 | 0.7153 |
| 6,500 | 39.5% | 46.5% | 56.5% | 61.5% | 51.00% | 0.6327 | 0.7202 |
| **7,000** | **36.5%** | **50.5%** | **54.5%** | **63.5%** | **51.25%** | 0.6344 | 0.7215 |
| 7,500 | 33.5% | 46.0% | 56.0% | 63.0% | 49.63% | 0.6374 | 0.7237 |
| 8,000 | 33.5% | 42.0% | 51.0% | 66.0% | 48.13% | **0.6389** | **0.7248** |

7,000스텝 체크포인트가 validation 평균 Top-1 51.25%로 선택되었다. 8,000스텝까지 reconstruction은 계속 개선됐지만 평균 retrieval은 7,000 이후 3.12%p 하락했다. 따라서 decoder 복원은 아직 과소학습 방향인 반면 style prototype 목적은 7,000 부근부터 과적합 또는 목적 간 간섭이 시작된 것으로 해석한다. 단일 validation 측정의 변동성이 있으므로 정확한 임계점이라기보다 6,500~7,000을 현재 학습 예산의 유효 구간으로 본다.

선택된 7,000스텝 모델의 보존 meta-test 100 artists 결과는 다음과 같다.

| Split | 1-ref Top-1 | 2-ref | 4-ref | 8-ref | 평균 Top-1 | L18 rec cosine | L24 rec cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Meta-test | 36.5% | 47.5% | 59.5% | 61.0% | **51.13%** | 0.6347 | 0.7211 |

이는 기존 5,000스텝 실행의 meta-test 평균 47.00%보다 4.13%p 높다. 다만 서로 다른 초기화의 실행이므로 순수한 추가 스텝 효과만을 분리한 비교는 아니다. 새 실행의 W&B 기록은 https://wandb.ai/1wndrla17-kyung-hee-university/anima-style-adapter/runs/per-reference-l18-l24-siglip-l24-8k-val-v2 에 있다.

## 32×1024 joint-token 재학습

표현 용량을 `16×768`에서 `32×1024`로 늘리고, 전체 ordered token을 flatten한 뒤
global LayerNorm/L2 normalization한 32,768차원 descriptor에 joint prototype loss를
적용했다. Artist loss는 joint 0.13과 보조 slot-wise 0.02로 총 0.15이며, batch 평균을
제거한 slot별 image-dependent variation의 중복을 약한 diversity loss 0.01로 억제했다.
모델은 111,197,696 parameters이며 체크포인트는 약 425 MiB다.

8,000스텝까지 학습했으며 validation 평균 Top-1이 가장 높은 6,500스텝 체크포인트를
선택했다.

| Split | 1-ref Top-1 | 2-ref | 4-ref | 8-ref | 평균 Top-1 | L18 rec cosine | L24 rec cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 42.0% | 55.0% | 62.0% | 70.0% | **57.25%** | 0.6667 | 0.7458 |
| Meta-test | 39.0% | 52.0% | 63.0% | 69.5% | **55.875%** | 0.6669 | 0.7454 |

기존 선택된 `16×768` 7,000-step 모델과 비교하면 meta-test 평균 Top-1은 51.13%에서
55.875%로 4.745%p, L18 reconstruction은 0.6347에서 0.6669로 0.0322, L24는
0.7211에서 0.7454로 0.0243 개선됐다. 학습 중 raw slot-variation diversity loss는
초기 중복 붕괴값 약 0.64에서 말기 약 0.15~0.20으로 감소했다.

- W&B: https://wandb.ai/1wndrla17-kyung-hee-university/anima-style-adapter/runs/per-reference-l18-l24-siglip-l24-32x1024-joint-v2
- checkpoint: `per_reference_resampler_l18_l24_siglip_l24_32x1024_joint_global_ln/runs/l18_l24_native_siglip_l24/checkpoint.pt`
- held-out result: `per_reference_resampler_l18_l24_siglip_l24_32x1024_joint_global_ln/final_test.json`

## 해석과 다음 단계

이번 수치는 보존 작가에 대한 token-space retrieval와 C-RADIO feature reconstruction 결과다. 실제 Anima 생성 품질을 직접 측정한 값은 아니다. 다음 단계에서는 Resampler를 우선 동결하고 slot-aligned Set Aggregator, shared full-rank K/V base, 28-block low-rank delta와 timestep gate를 연결한다. 초기 Adapter 안정화 뒤에만 Resampler 상위 1~2층을 낮은 학습률로 해제한다.
