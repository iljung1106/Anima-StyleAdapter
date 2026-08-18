# Compact Dual-query 5K Dual-domain Style Tokenizer 계획

## 목적

현재까지 가장 안정적이었던 소형 Compact Dual-query Style Tokenizer의 직접적인
Anima context 주입 방식과 16-token 출력을 유지하면서 다음 문제를 고친다.

- 500명 teacher의 반복 노출과 제한된 작가 범위
- 단일·2-reference 학습 비중 부족
- 긴 caption 중심 학습으로 인한 fixed-reference 일반화 부족
- 작가별 효과보다 공통 출력으로 수렴하는 현상
- reference마다 `84 -> 1` 가중 평균으로 생기는 정보 병목
- 모든 샘플에 하나의 global RMS를 적용하는 출력 강도 제약

기존 소형 checkpoint를 이어 학습하지 않고 새 초기화로 학습한다. Frozen
Dual-query Resampler와 Frozen Anima는 유지하며 새 Style Tokenizer만 학습한다.

## 데이터와 teacher

- 전체 5,000명에 대해 native Anima centered-effect teacher bank를 만든다.
- 분할은 Train 4,000 / Validation 500 / Meta-test 500으로 고정하며 validation과
  meta-test 작가는 optimizer에 노출하지 않는다.
- Human reference와 Anima synthetic reference는 서로 별개의 domain으로 취급한다.
  두 domain의 token이나 residual을 서로 같게 만드는 loss는 사용하지 않는다.
- Synthetic reference 확장은 한 번에 5,000명 전체를 생성하지 않는다. 기존
  `500명 x 8 content x 2 seed = 8,000장`은 그대로 재사용하고, 우선 기존 500명과
  겹치지 않는 1,500명을 골라 `6 content x 1 seed = 9,000장`만 추가한다. 이 단계의
  결과를 확인한 뒤 나머지 작가 생성 여부를 결정한다.
- 기존 synthetic image ID 영역과 합칠 때 충돌하지 않도록 신규 1,500명 cache는
  `20,000,000,000`부터 별도 image ID namespace를 사용한다.
- Human/synthetic 양쪽 모두 1/2/4-reference teacher batch를 만든다. Synthetic
  domain에서는 기존 500명 캐시와 신규 1,500명 캐시를 하나의 학습 pool로 합치되,
  동일 작가가 두 캐시에 중복되지 않도록 생성 plan 단계에서 기존 manifest를
  명시적으로 제외한다.
- 동일 content, seed, noise, timestep에서 `@artist` 유무에 따른 Frozen Anima
  velocity 차이를 구하고 content/timestep별 작가 평균을 뺀 centered residual을
  detached teacher target으로 사용한다.
- Teacher의 직접 residual 회귀를 주 손실로 두고 direction과 magnitude 회귀는
  약한 보조 손실로 사용한다. Human과 synthetic 손실은 따로 기록한다.

### Native artist effect의 timestep 보정

Frozen Anima에서 작가 태그가 만드는 실제 영향은 timestep마다 다르므로 5K native
teacher bank에서 timestep별 centered residual RMS의 median과 25--75 percentile을
미리 측정한다. Teacher target의 방향이나 절대 크기 자체는 바꾸지 않고 다음과 같이
teacher 관련 학습 강도만 완만하게 보정한다.

- timestep별 robust scale을 `s_t`, 전체 timestep median을 `s_med`라 할 때
  `w_t = clip((s_t / s_med)^0.25, 0.75, 1.33)`를 시작점으로 사용한다.
- 전체 평균 weight가 1이 되도록 정규화한다. Bank의 이산 timestep 사이에서 학습
  timestep이 샘플링되면 인접 weight를 선형 보간한다.
- `w_t`는 human/synthetic teacher residual, direction, magnitude, teacher 기반
  correct-vs-wrong ranking과 기본 rectified-flow MSE에 동일하게 적용한다. Flow MSE는
  샘플별로 먼저 계산한 뒤 각 샘플 timestep의 `w_t`를 곱해 평균한다.
- 원래 centered residual을 그대로 목표로 삼으므로 작가 효과의 절대 크기 증류는
  유지된다. 작은-effect 구간을 인위적으로 큰 residual로 만들거나 큰-effect 구간을
  정규화해 없애지 않는다.
- Validation에는 timestep별 unweighted metric과 weighted metric을 모두 기록한다.
  특정 timestep 개선만으로 전체 성능이 좋아 보이는 착시를 막기 위해 p25/median/p75
  effect 구간의 cosine, projection, RMS error도 별도로 기록한다.

이 보정은 초기에는 위의 제한된 범위로 고정한다. 실측 gradient와 heldout 결과 없이
가중 범위를 넓히거나 inverse-RMS weighting으로 약한 timestep을 과도하게 강조하지
않는다.

## Reference와 prompt 분포

Reference 수 1~8장의 확률은 다음과 같이 둔다.

| Reference 수 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 확률 | 45% | 25% | 12% | 7% | 4% | 3% | 2% | 2% |

일반 배치에서는 target을 reference에서 제외한다. Prompt mode는 다음과 같다.

- Full caption 30%
- Tag dropout 40%
- Short caption 20%
- Empty 10%
- Empty는 반드시 single exact-self reference로 구성한다.
- Empty가 아닌 배치의 50%에는 `masterpiece, best quality, score_7` quality
  prefix를 넣는다.

Dropout은 text tag에만 적용한다. 순수 style-token 주입에서 no-style은 Frozen
Anima 원본 경로이므로 learned null-style token이나 style dropout은 사용하지 않는다.

## 새 single-stage Compact typed-attention 구조

입력은 reference마다 Frozen Dual-query Resampler가 만든 `84 x 1024` token이다.
Reference를 먼저 각각 압축한 뒤 다시 합치는 2단계 구조를 사용하지 않는다. 대신
모든 reference의 token을 타입별 permutation-invariant set으로 직접 읽는다.

1. 모든 reference의 spatial-query `R x 64 x 1024`를 `R*64` memory로 펼친다.
   8개의 learned spatial query가 이 memory를 읽는다.
2. 모든 reference의 global-query `R x 16 x 1024`를 `R*16` memory로 펼친다.
   4개의 learned global query가 이 memory를 읽는다.
3. 모든 reference의 artist-summary `R x 4 x 1024`를 `R*4` memory로 펼친다.
   4개의 learned artist query가 이 memory를 읽는다.

세 타입은 별도 input LayerNorm, type embedding과 learned query를 사용한다. Spatial
memory에는 기존 8x8 위치 embedding을 reference마다 반복해 더한다. Reference 순서
embedding은 사용하지 않으며 유효하지 않은 reference에는 attention mask를 적용한다.

세 타입이 독립된 대형 attention stack을 갖지는 않는다. 하나의 pre-norm
cross-attention과 residual FFN을 세 타입이 공유하고 각 타입의 query와 memory만
구분한다. Query는 attention의 routing에만 사용하며 sample-independent learned query
자체를 value residual로 출력에 더하지 않는다.

각 attention의 attended value가 곧 `8 spatial + 4 global + 4 artist = 16`개의 최종
style token이 된다. 별도의 descriptor별 Set Attention, `8 -> 16` grouped MLP,
cross-slot Transformer 또는 추가 global output-query 계층은 두지 않는다. 이로써
기존 `84 -> 1` 병목은 없애면서 전체 Style Tokenizer를 약 8~12M 규모로 유지한다.

공유 attention 내부에는 필요한 pre-norm과 learned Q/K/V/O projection을 사용하지만
최종 16개 token을 만든 뒤에는 다음을 두지 않는다.

- post-output LayerNorm
- 학습 가능한 단일 global RMS
- token RMS 목표나 출력 크기 하한·상한 loss

최종 token의 방향과 크기는 flow 및 teacher loss가 직접 학습하게 한다. Attention과
FFN의 선형층은 Xavier 초기화하며 zero-init이나 고정 output scale을 사용하지 않는다.

## Loss와 curriculum

주 손실은 기존 rectified-flow MSE다. 여기에 human/synthetic centered teacher
residual 회귀를 각각 독립적으로 더한다.

- Step 0~500: flow MSE와 dual-domain teacher alignment를 매 step 학습
- Step 500~1,500: correct-vs-wrong cyclic flow ranking을 0에서 최대 weight까지 ramp
- Step 1,500~8,000: 전체 손실 유지

Dual-domain teacher update는 step 500까지 매 step 수행하고 step 501부터는 2 step마다
수행한다. 후반 teacher update의 loss를 2배로 보정하지 않으며, 반복 노출과 전체 teacher
비중을 함께 낮춘다.

Ranking은 동일 content/noise/timestep에서 batch artist를 cyclic shift하여 만든다.
Wrong reference의 출력을 임의로 망가뜨리지 않고 correct reference가 자신의 centered
teacher residual에 더 가까워지도록 margin ranking한다. 최대 weight는 기존 소형
baseline의 `0.00075`를 시작점으로 사용한다.

다음 손실은 사용하지 않는다.

- raw same-artist functional consistency
- 강한 common-output ratio penalty
- output RMS/floor/bounded-effect loss
- human-synthetic cross-domain consistency

## 학습과 평가

- 최대 8,000 optimizer steps, fused AdamW, LR `1e-4`, 200-step warmup과 cosine decay
- Batch 4 x gradient accumulation 4를 기준으로 시작하고 H100 실측 후 batch를 늘린다.
- Frozen Anima, Frozen Resampler와 cached text/VAE/reference token을 사용한다.
- 250 step마다 validation/checkpoint, 500 step마다 panel, 1,000 step마다 외부
  fixed-reference sample을 생성한다. Optimizer와 scheduler 상태를 함께 저장한다.
- W&B에는 heldout paired-flow improvement, correct-vs-wrong advantage,
  1/2/4/8-reference 성능, human/synthetic teacher cosine·projection·RMS,
  common-output ratio와 최종 token RMS 분포를 기록한다.

### 캐시 및 실행 순서

1. `synthetic-reference-additional`로 신규 1,500명 x 6장의 image, Qwen VAE latent,
   text conditioning을 resumable shard로 만든다.
2. `synthetic-reference-token-cache`에서 WebP decode를 미리 prefetch하고 C-RADIO
   L18/L24 spatial feature와 Frozen Dual-query Resampler를 같은 GPU pipeline에서
   연속 실행한다. L18/L24 중간 feature는 NFS에 저장하거나 다시 읽지 않고 최종
   `84 x 1024` token과 512-D descriptor만 저장한다.
3. 학습 loader는 기존 500명 token root와 신규 1,500명 token root를 하나의
   artist-balanced synthetic pool로 읽는다. Root마다 같은 `part-*.safetensors`
   파일명이 존재하므로 `(root, shard)`를 cache key로 사용한다.
4. `single-stage-typed-attention-smoke`로 real Anima forward, 두 synthetic root,
   timestep weighting과 resume checkpoint를 검증한 뒤
   `single-stage-typed-attention-train`을 시작한다.

이미지 생성, direct token cache와 8K 학습은 각각 독립적으로 재개 가능하다. 생성이
완료되지 않았거나 신규 token manifest가 없으면 학습을 시작하지 않는다.

4,000-step 중간 gate에서 기존 최고 소형 baseline의 heldout improvement `0.00546`와
selection score `0.00961`을 비교한다. 정량 성능과 fixed-reference의 선화·명암·형태
분리가 함께 개선될 때만 같은 optimizer 상태로 8,000 step까지 진행한다.
