# Global-query multi-prompt Style Tokenizer 학습 계획

## 목표

기존 Compact Style Tokenizer의 `84 tokens -> reference descriptor -> set
descriptor` 조기 압축을 제거한다. Frozen Dual-query Resampler가 이미지마다
출력하는 세 종류의 memory를 보존한 채, 16개의 global learned query가 여러
레퍼런스에서 직접 필요한 정보를 읽어 Anima용 `16 x 1024` style token을 만든다.

Dual-query Resampler와 Anima는 이번 실험에서 동결한다. 기존 2,000-step
체크포인트는 초기값으로 사용하지 않고 새 Style Tokenizer를 처음부터 학습한다.

## 모델 구조

각 레퍼런스는 다음의 `84 x 1024` frozen memory를 제공한다.

- spatial-query token 64개: 별도 type embedding과 기존 8x8 공간 위치를 유지
- global-query token 16개: 별도 type embedding 사용
- artist-summary token 4개: 별도 type embedding 사용

세 token type을 단순히 같은 slot으로 취급하지 않는다. 먼저 모든 레퍼런스에
가중치를 공유하는 reference-local Transformer block 1층과 reference register
token 1개를 적용한다. 이후 유효한 레퍼런스 memory를 하나의 masked memory set으로
이어 붙인다.

레퍼런스 번호나 순서를 나타내는 ordinal embedding은 사용하지 않는다. 학습 중
레퍼런스 순서를 무작위화하여 set permutation에 둔감하게 만든다.

최종 tokenizer는 다음 구조를 사용한다.

1. learned global query 16개, width 1024
2. Pre-LN cross-attention -> self-attention -> FFN block 2층
3. final LayerNorm 뒤에 Linear output projection 적용
4. Anima post-LLM context에 삽입되는 `16 x 1024` style token 출력

Linear projection 뒤에는 LayerNorm, RMS normalization, `log_output_rms` 또는 전역
고정 scale을 적용하지 않는다. 따라서 최종 token의 크기는 샘플과 slot에 따라
달라질 수 있으며, flow/teacher loss가 필요한 작가 효과의 방향과 강도를 함께
결정한다. Final LayerNorm은 Transformer hidden state의 안정화에만 사용한다.

Output projection은 calibration batch에서 초기 출력 RMS가 대략 `0.10--0.15`가
되도록 초기 weight를 한 번만 축소한다. 이는 초기화 시점의 보정일 뿐 런타임
normalization이나 학습 중의 강도 제한이 아니다. 단순히 기존 `x 0.15`만 제거하여
초기 RMS가 약 1.0으로 급증하는 구현은 피한다.

고정 output RMS나 같은 번호의 reference slot끼리 pooling하는 경로는 두지 않는다.
Padding reference 및 memory에는 attention mask를 적용한다.

## Prompt curriculum

각 일반 flow-training sample의 prompt mode를 독립적으로 선택한다.

- Full caption: 30%
- Tag Dropout: 40%
- Short caption: 20%
- Empty: 10%

Tag Dropout은 문자열 단계에서 일반 태그의 20--60%를 제거한 뒤 실제 Qwen text
encoder와 Anima LLM adapter를 통과시킨다. Rating, 인물 수와 주요 character tag는
보존한다. Short caption은 rating/인물 수/주요 character와 소수의 핵심 content
tag만 유지한다.

Empty sample은 반드시 다음 조건을 모두 만족한다.

- reference 한 장
- reference와 target이 동일한 이미지
- 실제 empty text conditioning 사용
- full target flow MSE 적용
- quality prefix 적용 금지

Empty가 아닌 sample에는 50% 확률로
`masterpiece, best quality, score_7` quality prefix를 붙인다. 기존 quality tag는
중복하지 않고 rating과 충돌하는 조합을 만들지 않는다.

## Loss

새 구조의 효과를 분리해 보기 위해 현재 dual-domain 실험의 loss 종류를 유지한다.

- target flow MSE
- human-reference centered native artist-effect teacher
  - normalized Huber, direction, magnitude
- synthetic-reference centered native artist-effect teacher
  - normalized Huber, direction, magnitude
- 기존 functional common-output penalty를 sparse teacher student residual에 재사용
- final-token slot diversity (`slot_diversity_weight: 0.003`)
- 약한 attention-map diversity (`attention_diversity_weight: 0.001`): 16개 global
  query가 동일 memory에 집중하는 현상 억제
- 약한 reference-conditioned token diversity
  (`reference_conditioned_diversity_weight: 0.001`): batch 공통 slot 성분을 제거한 뒤
  reference에 의해 달라진 token 성분의 중복 억제

Attention-map 및 reference-conditioned diversity는 초기 500 step 동안 0에서
목표값까지 ramp한다. 두 항은 고정 query/slot embedding만 서로 다르게 만드는
shortcut을 피하도록 각각 attention map과 batch-centered token delta에 적용한다.
`slot_diversity_weight=0.003`은 최종 16개 token에 적용하되, 세 diversity 항의
weighted contribution과 pairwise cosine을 별도로 기록하여 flow loss를 압도하지
않는지 확인한다.

Common-output loss는 같은 content/noise/timestep에서 여러 작가의 student velocity
delta를 계산한 뒤 아래 비율이 threshold를 넘는 부분만 처벌한다.

`common_output_ratio = RMS(mean_artist(delta)) / mean_artist(RMS(delta))`

새 Anima forward를 추가하는 기존 full functional probe는 켜지 않는다. Human과
synthetic domain 각각에서 이미 수행한 sparse teacher forward의 student delta를
재사용하고, 두 domain을 서로 같은 표현으로 강제하지 않도록 loss도 독립적으로
계산한다. 초기 설정은 per-call weight `0.04`, threshold `0.85 -> 0.70`, ramp
`500 -> 1500 step`으로 한다. Teacher interval이 4이므로 평균 기여는 약
`0.01/step`이다. 분모는 detach하여 단순히 전체 출력 크기를 키워 loss를 피하지
못하게 한다.

기존 centered artist-effect teacher가 이미 올바른 작가별 방향을 제공하므로,
별도의 artist contrastive head나 centered-effect floor는 우선 활성화하지 않는다.
Centered floor는 임의의 틀린 작가 차이도 확대할 수 있기 때문이다. 이번 실험에는
correct-vs-wrong ranking, 강제 token RMS 또는 추가 prototype loss도 새로 넣지
않는다. 각 prompt mode별 flow loss와 qualitative sample을 따로 기록하여
Empty/Short가 content leakage를 증가시키는지 확인한다.

## Sparse native-artist teacher

Human 및 synthetic teacher는 매 step이 아니라 optimizer step 4회마다 fused
forward 한 번을 수행한다. 시작 설정은 다음과 같다.

- base learning rate: `2e-4`
- warmup: 400 optimizer steps
- cosine decay, minimum LR ratio: 0.10
- teacher interval: 4 optimizer steps
- human teacher per-call weight: 0.10
- synthetic teacher per-call weight: 0.10
- 기존 max grad norm 유지

이는 teacher의 평균 가중치를 각 domain에서 `0.025/step`으로 만들어 기존
`0.05/step`의 절반으로 낮추고, 반복적인 teacher bank 노출과 계산량을 줄인다.
Teacher 전용 optimizer를 추가하지 않으며, 호출 간격과 loss weight로 유효 강도를
조절한다.

Teacher content/timestep은 raw training step으로 선택하지 않는다. 별도의
`teacher_update_index`를 checkpoint에 저장하고, 8 content x 8 timestep의 64개
조합을 매 cycle마다 seeded permutation하여 모두 균등하게 사용한다. 따라서
`teacher_every=4`와 bank 크기의 공약수로 인해 일부 probe만 반복되는 aliasing을
방지한다.

## 학습 및 판정

- 총 10,000 optimizer steps
- reference 수는 1장이 가장 많고 2장, 4장, 8장 순으로 감소하는 분포 사용
- 250 step마다 validation
- 500 step마다 고정 train/validation artist panel 생성
- 1,000 step마다 외부 fixed-reference sample 생성
- W&B에 전체 및 prompt-mode별 flow loss, teacher 호출 수/조합 coverage,
  human/synthetic teacher cosine/projection/magnitude, token diversity,
  reference-count별 성능을 기록
- 출력 강도 진단으로 전체 style-token RMS 평균/표준편차, 샘플별 RMS 분산,
  16개 slot별 RMS, artist 간 RMS 분산, Anima velocity-delta RMS,
  teacher 대비 student delta RMS 비율과 RMS--paired-improvement 상관관계를 기록

주 판정 기준은 fixed-reference의 작가별 시각적 분리, prompt 길이에 대한 견고성,
target-excluded paired flow improvement, teacher direction/projection, 1-reference와
multi-reference 성능이다. Teacher 지표만 좋아지고 실제 reference style 차이가
줄어들면 teacher weight를 늘리지 않고 구조 또는 prompt curriculum을 우선
재검토한다.
