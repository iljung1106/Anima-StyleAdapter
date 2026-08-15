# 순수 Style Token 주입에서 복제 K/V로 확장하는 계획

## 목표

먼저 확장형 StyleTokenizer가 만드는 `32 x 1024` 토큰을 Anima의 native
text-context 좌표에 직접 정렬한다. 이 단계에서는 별도 style attention,
style K/V, block alpha를 두지 않는다. 작동하는 token-only 모델을 얻은 뒤에만
native K/V의 복사본과 zero-init delta를 추가한다.

이 순서는 다음 두 문제를 분리한다.

1. C-RADIO/Resampler 표현을 Anima가 이해하는 context token으로 변환하는 문제
2. 정렬된 visual token을 위한 K/V 용량을 원본 text K/V와 독립적으로 늘리는 문제

## Phase T: 순수 토큰 주입

- Frozen Resampler cache `128 x 1024`를 입력으로 사용한다.
- 각 reference는 learned query decoder 3층을 거쳐 독립적인 32개 slot이 된다.
- 여러 reference는 slot별 attention으로 합치고 cross-slot Transformer 2층을
  통과한다.
- 최종 `32 x 1024` 토큰을 각 prompt의 실제 text token 직후 null/padding
  위치에 삽입한다. 전체 Anima context 길이 512는 유지한다.
- Frozen Anima의 원본 단일 Q/K/V/O와 단일 softmax만 사용한다.
- 별도 style branch, style alpha, output gate는 사용하지 않는다.

### 학습 curriculum

- step 0--2k: exact target 1장을 reference로 사용한다.
- step 2k--8k: reference 1--4장, target 포함 확률 `1.0 -> 0.5`.
- step 8k--20k: reference 1--8장, target 포함 확률 `0.5 -> 0.0`.
- 모든 단계는 전체 train artist/image split을 사용한다.

### Loss

Frozen Anima의 text-only 예측을 `v_base`, style token을 넣은 예측을
`v_style`, rectified-flow target을 `v_target`이라 한다.

- 표준 flow MSE: `MSE(v_style, v_target)`
- normalized residual Huber:
  `Huber((v_style-v_base)/RMS(r), r/RMS(r))`, `r=v_target-v_base`
- target-aligned coefficient floor:
  `c=<d,r>/(||r||^2+eps)`, `d=v_style-v_base`,
  `ReLU(c_min-c)^2`
- aligned floor는 exact-self에서 `0.05 -> 0.25`로 ramp하고 target이 실제
  reference에 포함된 sample에만 적용한다. 따라서 target 포함 curriculum과
  함께 자연스럽게 사라진다.
- 약한 per-reference reconstruction, artist contrastive, slot diversity loss를
  유지해 표현 붕괴를 막는다.

총 output RMS에만 하한을 주지 않는다. 잘못된 직교 방향으로 크기만 키우는
shortcut을 막기 위해 반드시 target 방향으로 투영된 성분에 하한을 둔다.

### 검증과 샘플

- validation: 250 step마다 train-self, validation-self, heldout, wrong-artist
- checkpoint와 8개 정성 샘플: 500 step마다
- 샘플: train artist 4명 + validation artist 4명, 768 x 768, 30 steps,
  text CFG 4, style CFG 1
- 핵심 지표: paired flow improvement, correct-vs-wrong advantage,
  aligned coefficient, delta/desired RMS, direction cosine, orthogonal ratio

## Phase KV-J: 복제 K/V + joint softmax

Phase T checkpoint가 self/heldout 수치와 정성 샘플에서 유효한 reference
효과를 보인 뒤 진행한다.

- tokenizer checkpoint를 그대로 사용한다.
- `K_style=K_native+DeltaK`, `V_style=V_native+DeltaV`로 초기화한다.
- `DeltaK/DeltaV`는 zero-init low-rank delta로 둔다.
- text K/V와 style K/V는 계산만 분리하되 logits를 합쳐 **하나의 softmax**를
  사용하고 native O를 공유한다.
- padding token 수와 순서를 Phase T와 동일하게 유지하면 초기 함수가 Phase T와
  같아야 한다. 전환 전 max/mean output error를 자동 검증한다.
- 처음 500--1,000 step은 tokenizer를 동결하고 K/V delta만 학습한다.
- 이후 tokenizer 상위 층을 K/V LR의 `0.1--0.25x`로 열고 reconstruction과
  artist loss를 함께 유지한다.

## 선택적 Phase KV-S

분리 softmax가 실제 품질 또는 제어성에서 필요하다는 증거가 있을 때만
joint-softmax 모델을 teacher로 삼아 진행한다. 분리 softmax는 Phase T와 함수가
동일하지 않으므로 임의 alpha로 시작하지 않고 동일 noisy latent/timestep에서
teacher velocity와 block residual을 증류한다.

