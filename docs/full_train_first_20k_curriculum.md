# Fresh 20k 본학습 커리큘럼

## 목적

96/24 exact-self 실험은 현재 Adapter 구조가 학습 가능함을 확인하는 진단으로만 보존한다.
본 실행은 그 2,000-step checkpoint나 과거 Connector checkpoint를 초기값으로 사용하지 않는다.
Frozen Anima의 block별 pretrained K/V/O 좌표와 사전학습된 Per-reference Resampler는 그대로
사용하되, Style Adapter는 새로 초기화하고 optimizer/scheduler도 step 0부터 시작한다.

모든 단계는 소규모 subset이 아니라 production cache의 전체 train split에서 target과
reference를 뽑는다. Validation split의 작가는 학습에 사용하지 않는다.

## 0–20k reference 커리큘럼

| Optimizer step | 총 reference 수 | target 포함 확률 | 설명 |
|---:|---:|---:|---|
| 0–2,000 | 1 | 100% | target 한 장으로 같은 target 예측 |
| 2,001–8,000 | 1–4 | 100% → 50% | target과 같은 작가의 다른 이미지 혼합 |
| 8,001–20,000 | 1–8 | 50% → 0% | target-excluded multi-reference로 전환 |

`총 reference 수`는 target을 포함한 수다. Target을 사용하는 episode에서는 기존
same-artist reference 하나를 target으로 교체하며 상한에 한 장을 더하지 않는다. Exact-self
residual/direction/x0 보조 손실은 target이 실제 reference에 포함된 episode에만 적용한다.
Target이 제외된 episode는 표준 rectified-flow loss로 학습한다.

Minimal Set Aggregator는 step 2,001부터 학습한다. Anima와 C-RADIO는 계속 동결하고,
Per-reference Resampler도 이 20k 검증 구간에서는 동결한다. 별도의 oracle distillation은
사용하지 않는다.

## 고정 정성 평가 패널

매 500 optimizer step마다 다음 여덟 장을 생성한다.

- train split에서 서로 다른 작가 4명
- artist-disjoint validation split에서 서로 다른 작가 4명
- 0–2k는 exact-self reference, 이후에는 target-excluded held-out reference
- 512×512, 30 denoising steps, Text CFG 4, Style CFG 1
- 작가마다 서로 다른 고정 seed를 사용하고 체크포인트 사이에는 같은 seed를 재사용

각 sheet에는 같은 seed의 frozen-Anima baseline, styled output, target과 reference가 함께
들어간다. W&B에도 `sample/train_artist_1..4`와
`sample/validation_artist_1..4`로 기록한다.

## 실행 경계

- 총 20,000 optimizer steps
- validation/checkpoint: 250 steps
- 고정 패널 생성: 500 steps
- output: `style_transfer_full_train_fresh_20k_v1`
- W&B: `anima-style-transfer-full-train-fresh-20k-v1`

20k 이후 단계는 이번 실행 결과를 보고 별도로 결정한다. 특히 train/validation 작가의
스타일 분화, target 제외 전환기의 품질, reference 수 증가 효과를 확인하기 전에는
Resampler 공동학습이나 더 긴 production schedule을 열지 않는다.
