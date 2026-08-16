# Dual-query Style Tokenizer 10k Pilot 계획

## 목적

Artist-summary token을 사용하는 Dual-query Style Tokenizer를 전체 train split에서
10,000 optimizer step 학습한다. Frozen Anima의 기본 생성 품질을 유지하면서
올바른 reference가 만드는 작가별 velocity 방향과 충분한 효과 크기를 학습하고,
작가와 무관한 공통 출력 및 직교 방향의 큰 변형을 억제하는 것이 목표다.

## 고정 구조

- Per-reference encoder: 학습 완료된 Dual-query Resampler, 동결
- Reference 표현: `80×1024` query + `4×1024` artist-summary
- Set Style Tokenizer: artist-summary **ON**, 출력 `32×1024`
- 주입: Anima post-LLM `512×1024` context의 실제 text 뒤 빈 위치
- Frozen Anima의 native Q/K/V/O와 shared text/style CFG 사용
- 학습 대상: Set Style Tokenizer만

## 10k 커리큘럼

| Step | Reference | 목적 |
|---:|---|---|
| 0–500 | exact target 1장 | native context 정렬과 초기 방향 형성 |
| 500–1,000 | target-excluded 같은 작가 1–2장 | exact-self shortcut 제거 |
| 1,000–4,000 | target-excluded 같은 작가 1–4장 | correct-vs-wrong ranking 학습 |
| 4,000–10,000 | target-excluded 같은 작가 1–8장 | 실제 multi-reference 일반화 |

Reference 수는 1장을 가장 자주, 2장을 그다음으로 뽑는다. 3장 이상은
aggregation 능력을 유지할 만큼만 노출한다. Step 500 이후 target 이미지는
reference에 포함하지 않는다.

## Loss

주 loss는 Frozen Anima rectified-flow MSE다.

### 항상 유지

- flow MSE: weight `1.0`
- artist token contrastive: weight `0.01`, 2 step마다 계산
- slot diversity: weight `0.001`
- subset consistency: weight `0.002–0.005`
- normalized residual Huber:
  - exact-self `0.05`
  - target-excluded `0.01–0.02`

### Target-aligned coefficient floor

단순 출력 RMS가 아니라 `delta = v_style - v_base`가
`residual = v_target - v_base` 방향으로 갖는 투영 계수에 하한을 둔다.

- step 0–500: `c_min 0.02 → 0.15`, weight `0.20–0.25`
- step 500–10,000: `c_min 0.03 → 0.06`, weight `0.05–0.10`
- 학습 종료까지 유지하되 target-excluded 구간에는 더 낮은 하한을 사용한다.

### Correct-vs-wrong cyclic flow ranking

- step 1,000부터 활성화
- 배치 안의 다른 작가 reference를 cyclic donor로 사용
- wrong prediction은 detach하여 wrong 출력을 고의로 망치는 shortcut 차단
- weight `0 → 0.00075–0.001`, 1,000 step 동안 ramp
- 4 step마다 batch row 2개에서 계산
- correct/wrong direction ranking과 약한 centered-direction 항을 함께 사용

### Bounded aligned-effect 보조항

원본 대비 raw 출력 변형량을 직접 보상하지 않는다. 4 step마다 target 방향의
양의 투영량에 하한과 상한을 두고, 직교 성분이 커지면 패널티를 준다.
Target-excluded 구간의 초기 범위는 `r_min 0.04–0.06`, `r_max 0.20`으로 둔다.

### Common-output penalty

- step 0–1,000: 측정만 수행
- step 1,500부터 활성화
- 8 step마다 작은 controlled microbatch에서 velocity delta의 공통 성분 계산
- `ReLU(common_output_ratio - threshold)^2` 형태의 hinge loss
- weight `0.001–0.005`
- denominator는 detach하고, flow alignment/ranking과 함께 사용하여 작가별
  무작위 노이즈를 만드는 shortcut을 억제한다.

Controlled probe에 동일 noisy latent·prompt·timestep을 사용하는 것은 내부
지표/loss 계산에만 한정한다. 주기적 정성 샘플의 작가별 target 프롬프트와
시드는 기존 방식대로 유지한다.

## 학습과 기록

- optimizer: AdamW, peak LR `1e-4`, warmup 250–500 step, cosine decay,
  minimum LR `1e-5`
- batch: 4, gradient accumulation 4
- validation: 250 step마다
- checkpoint 및 train 4명/validation 4명 패널: 500 step마다
- 고정 외부 reference sheet: 1,000 step마다
- 1/2/4/8-reference 평가: 1,000 step마다
- W&B에 각 raw loss, weighted loss, gradient norm, LR과 curriculum 상태 기록

필수 지표:

- self/heldout/wrong paired-flow improvement와 positive fraction
- correct-vs-wrong paired advantage
- aligned coefficient와 floor violation
- direction cosine, delta/desired RMS, orthogonal/desired RMS
- style-output ratio와 style-token RMS
- common-output ratio
- within/between artist centered cosine와 artist retrieval
- reference subset consistency

## 선택 및 중단 기준

다음 조건을 함께 만족하는 checkpoint를 선택한다.

1. heldout paired improvement가 양수이고 여러 validation에서 유지된다.
2. correct-vs-wrong advantage가 양수이며 증가한다.
3. 출력 효과가 지나치게 작아지지 않고 orthogonal ratio가 폭증하지 않는다.
4. common-output ratio가 감소하거나 허용 범위 안에 머문다.
5. 1-reference 성능을 유지하면서 multi-reference가 품질을 개선한다.
6. 정성 샘플에서 content 보존, 작가별 차이, 이미지 안정성이 함께 확인된다.

연속된 validation에서 heldout 성능과 correct-vs-wrong advantage가 함께
악화되거나, common/orthogonal 성분 증가와 생성 붕괴가 나타나면 중단하고
마지막 안정 checkpoint로 되돌린다. 이 10k run은 최종 모델 확정이 아니라
flow-level 작가 구분과 비붕괴 학습 가능성을 검증하는 pilot으로 취급한다.

## 2k 런타임 개입

v1은 1,500→1,750→2,000 validation에서 heldout paired improvement와
correct-vs-wrong advantage가 함께 하락했다. 고정 reference sheet의 baseline 대비
pixel RMS도 step 1,000의 `0.2191`에서 step 2,000의 `0.1595`로 감소했다.
반면 controlled artist retrieval top-1은 `1.0`, common-output ratio는
`0.9256→0.9233`이었다. 즉 tokenizer에는 작가 구분 정보가 남았지만 Anima가
사용하는 flow 방향과 효과 크기가 약해진 것으로 판단한다.

원시 학습 loss를 확인하면 step 2,000에서 normalized residual의 평균 기여는
약 `0.0094`, token contrastive는 약 `0.0010`이지만 aligned floor는 약
`0.00007`, cyclic direction/common 항은 각각 대략 `1e-5` 수준이었다. 또한
기존 cyclic 구현은 cosine direction만 비교하고 실제 flow MSE advantage를
직접 최적화하지 않았다.

따라서 v1 step 1,500을 마지막 안정 초기값으로 보존하고 v2를 이어간다.

- cyclic wrong-reference 항에 normalized correct-vs-wrong flow-improvement
  ranking을 추가하며 wrong prediction은 계속 detach한다.
- ranking 대상 행을 2→4, 최종 weight를 `0.00075→0.003`으로 높인다.
- heldout aligned-floor weight를 `0.075→0.30`, bounded-effect weight를
  `0.015→0.03`으로 높여 출력 축소 shortcut을 막는다.
- 이미 포화된 token contrastive를 `0.01→0.005`, subset consistency를
  `0.003→0.0015`로 낮춰 Anima가 사용하지 않는 토큰 방향이 주목적을
  압도하지 않도록 한다.
- common-output penalty와 나머지 커리큘럼은 그대로 유지한다.

v1 산출물은 삭제하거나 덮어쓰지 않으며, v2는 별도 output/W&B run에서
step 1,500 optimizer·RNG 상태를 복원한다. step 1,750/2,000/2,250의 같은
validation과 step 2,000 fixed sheet로 개입 효과를 먼저 확인한 뒤 10k까지
계속한다.

## v2 정성 붕괴와 v3 기능 공간 교정

v2는 step 7,610에서 중단했다. heldout paired-flow improvement와
correct-vs-wrong advantage는 양수였지만, 고정 외부 reference sheet에서는
레퍼런스와 무관한 약하고 잘못된 변화가 반복되었다. 같은 prompt·seed의 최종
이미지 delta를 측정하면 전체 변화의 약 `77–86%`가 reference 사이에 공통이었다.
반면 tokenizer 출력은 step이 증가할수록 reference 사이 구분이 커졌다. 이는
token collapse가 아니라, token 차이가 Frozen Anima가 스타일로 사용하는
방향에 정렬되지 않은 기능적 붕괴다.

v2의 target-excluded normalized residual은 매 step 약 `0.0097`의 weighted
loss를 만들었지만 common-output 항은 8 step 평균 약 `1e-5`였다. 레퍼런스로
예측할 수 없는 content/noise 성분이 섞인 target residual을 강하게 회귀하면서
dataset-average residual이 가장 쉬운 해가 된 것으로 판단한다.

v3는 v1 step 1,500을 다시 초기값으로 사용하고 다음처럼 교정한다.

- target-excluded normalized residual `0.015→0.0015`
- heldout aligned-floor weight `0.30→0.04`, coefficient floor `0.01→0.02`
- bounded aligned effect `0.03→0.01`, 허용 구간 `0.01–0.15`
- token contrastive `0.005/every 2→0.001/every 4`
- correct-vs-wrong flow ranking `0.003/every 4→0.01/every 2`
- common-output `0.002/every 8→0.03/every 2`, threshold `0.60→0.55`
- 같은 작가의 서로 겹치지 않는 두 reference view가 동일한 controlled
  prompt·noise·timestep에서 만드는 Frozen Anima velocity residual의 방향과
  크기를 맞추는 functional consistency `0.02/every 2`
- 전체 작가 공통 residual을 뺀 centered artist effect ratio에 `0.35→0.55`
  하한을 두는 loss `0.03/every 2`

Functional probe는 작가 4명과 작가별 reference 4장을 사용한다. 두 view 중
하나는 detach된 목표로 계산하여 H100에서 두 개의 Anima backward graph를
동시에 보존하지 않는다. 로그에는 각 raw/weighted/per-step loss 외에 다음을
반드시 기록한다.

- same-artist functional cosine과 magnitude 오차
- between-artist functional cosine과 pairwise RMS
- common-output ratio와 common RMS
- centered artist-effect ratio와 RMS
- 전체 functional probe의 cadence-adjusted weighted contribution

`same-artist cosine↑`와 함께 `between-artist cosine/common ratio↑`가 나타나면
공통 출력 붕괴, effect RMS가 함께 감소하면 약한 출력 붕괴,
centered ratio는 증가하지만 heldout flow와 정성 샘플이 나빠지면 임의의
off-manifold artist direction으로 진단한다. v3도 fixed-reference의 작가별
차이와 안정성을 회복하지 못하면 수치상 paired-flow improvement만으로
10k까지 계속하지 않는다.

### v3 초기화 교정

첫 v3 시도는 과거 계획대로 v1 step 1,500의 tokenizer·optimizer·RNG에서
재개했으나 step 1,710 부근에서 중단했다. 이 방식은 새 objective의 exact-self
bootstrap과 functional ramp를 건너뛰며, 기존의 강한 residual/token objective가
만든 optimizer moment와 표현을 새 loss가 먼저 되돌려야 한다. 실제로 초기
same-artist functional cosine이 약 `0.88→0.71`로 급락하여 새 설계 자체와
과거 가중치 제거 과도기를 구분할 수 없었다.

따라서 이 시도는 진단용으로만 보존하고 최종 v3 검증은
`dual_query_style_tokenizer_summary_on_10k_pilot_v4_scratch`에서 수행한다.
Frozen Anima와 frozen Resampler token cache만 재사용하고, StyleTokenizer와
AdamW optimizer는 step 0에서 새로 초기화한다. 어떠한 과거 StyleTokenizer
checkpoint, optimizer state, history 또는 RNG도 불러오지 않는다.
