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

## 해석과 다음 단계

이번 수치는 보존 작가에 대한 token-space retrieval와 C-RADIO feature reconstruction 결과다. 실제 Anima 생성 품질을 직접 측정한 값은 아니다. 다음 단계에서는 Resampler를 우선 동결하고 slot-aligned Set Aggregator, shared full-rank K/V base, 28-block low-rank delta와 timestep gate를 연결한다. 초기 Adapter 안정화 뒤에만 Resampler 상위 1~2층을 낮은 학습률로 해제한다.
