# Typed Multi-Descriptor Compact Style Tokenizer 계획

## 목적

성공적이었던 소형 Style Tokenizer의 안정적인 압축, 명시적인 최종 slot,
출력 scale을 유지하면서 Dual-query Resampler의 `84 -> 1` 조기 정보 병목을
`84 -> 8` typed descriptor로 완화한다. Frozen Dual-query Resampler와 Frozen
Anima를 사용하며, 과도한 content 보존과 global-query slot 붕괴를 피한다.

## 모델 구조

1. 각 reference의 `84 x 1024` cached token을 타입별로 읽는다.
   - spatial 64개 -> descriptor 4개
   - global 16개 -> descriptor 2개
   - artist-summary 4개 -> descriptor 2개
2. 하나의 cross-attention 층과 type mask로 reference마다 `8 x 1024`를 만든다.
   Descriptor query 사이 self-attention은 사용하지 않는다.
3. 같은 descriptor slot끼리 reference attention pooling하여 reference 순서에
   불변인 `8 x 1024` style memory를 만든다.
4. RMS 약 1로 초기화한 명시적 learned query 16개가 style memory를 한 층
   cross-attention하여 Anima용 `16 x 1024` token을 만든다.
5. 최종 query끼리 self-attention하지 않는다. Learned query는 attention의 Q로만
   사용하고 출력 value residual에는 직접 더하지 않는다. 대신 reference-conditioned
   value에 slot별 multiplicative modulation을 적용한다. 이로써 slot identity를
   보존하면서 모든 샘플에 동일한 learned query가 출력되는 shortcut을 막는다.
6. Final LayerNorm 뒤 gain은 첫 2,000-step gate까지 0.15로 고정한다. 첫 실험에서
   learnable gain이 0.15에서 약 0.10으로 축소되어 flow 손상을 피하는 shortcut이
   확인되었기 때문이다. 안정적인 artist-specific 효과가 생긴 뒤에만 bounded
   sample gain을 다시 여는 것을 검토한다.

목표 모델 크기는 약 16--22M이다. 초기 학습에서는 Resampler를 동결하고,
필요성이 지표로 확인된 경우에만 후반에 상단부 일부를 낮은 LR로 연다.

## Prompt와 reference 배치

- Full caption 30%
- General-tag dropout 40%: general tag의 20--60% 제거
- Short caption 20%: rating, `1girl`/`solo`, 주요 내용 태그 4--6개
- Empty 10%: quality tag 없이 반드시 단일 `reference = target`
- 초기 500 step에는 Empty를 20%로 높인 뒤 10%로 내릴 수 있다.
- 작가명과 `@artist` 태그는 모든 경로에서 금지한다.
- Empty가 아닌 배치의 50%에 quality prefix를 혼입한다.
  (`masterpiece, best quality, score_7`)
- Reference 수는 1/2/3/4/5/6/7/8장에 각각
  `45/25/12/7/4/3/2/2%`를 사용한다.
- Empty 이외에는 target-excluded same-artist reference를 기본으로 한다.

## Loss 커리큘럼

### Step 0--500

- Rectified-flow MSE
- Human/Synthetic artist teacher direct alignment
- Empty exact-self 학습
- teacher 방향으로 투영된 절대 효과 크기 하한
- teacher와 직교하는 residual 억제

전체 centered RMS에는 하한을 걸지 않는다. Teacher residual을 `t`, student
residual을 `s`라 할 때 아래 투영 계수를 직접 제약한다.

\[
a = \frac{\langle s,t\rangle}{\lVert t\rVert^2 + \epsilon}
\]

### Step 500--1,500

- 위 loss 유지
- 동일 작가의 서로 겹치지 않는 reference view 사이 flow consistency
- pooled artist-summary에만 약한 contrastive loss

### 이후

Teacher cosine과 projection coefficient가 연속 validation에서 0.1 이상이고
heldout paired-flow improvement가 음수가 아닐 때 다음 loss를 점진적으로 연다.

- 동일 content/timestep의 cyclic correct-vs-wrong flow ranking
- teacher-aligned 절대 크기를 분모로 쓰는 common-output penalty
- artist hard-negative contrastive
- 실제 Frozen Anima K/V 뒤의 약한 functional slot diversity

Attention-map diversity와 Tokenizer 이미지 reconstruction은 사용하지 않는다.
강한 prototype loss를 최종 16개 Anima token에 직접 걸지 않고, typed
descriptor에서 만든 보조 artist-summary에만 약하게 적용한다.

## 주요 판단 지표

- heldout paired-flow improvement 및 95% CI
- teacher direction cosine과 projection coefficient
- teacher-aligned RMS와 orthogonal/teacher RMS
- correct-vs-wrong advantage
- common/artist-specific aligned effect ratio
- 1-reference와 multi-reference 성능
- 최종 token slot cosine 및 Frozen Anima K/V functional diversity
- 고정 prompt가 아닌 panel과 fixed-reference 정성 샘플

첫 실험은 2,000 step으로 제한한다. 위 지표가 소형 baseline을 따라잡는 것이
확인된 뒤에만 8,000 step 본학습과 선택적 Resampler 공동학습을 진행한다.

## 500-step v1 진단과 v2 수정

v1은 heldout paired-flow improvement가 `-0.00488`, human/synthetic teacher
projection coefficient가 `0.033/0.040`이었고, common/teacher-aligned effect
비율이 `56.5/65.8`까지 증가했다. 최종 token RMS도 `0.15 -> 0.10` 방향으로
축소됐다. 정성 샘플에서는 레퍼런스 화풍보다 공통 chibi/고채도 변형이 강했다.

v2에서는 다음을 적용한다.

- descriptor/final learned query의 raw residual 및 additive reinjection 제거
- learned query는 routing Q로만 사용하고 slot별 multiplicative modulation 사용
- 첫 2,000 step 동안 output gain 0.15 고정
- projected-effect loss weight `0.05 -> 0.50`
- common-output 분모를 작은 student aligned RMS가 아니라 teacher centered RMS로
  바꾸고 step 1부터 250까지 weight를 ramp
- v1의 500-step checkpoint는 재사용하지 않고 처음부터 학습

## 2,000-step v2 게이트와 v3 전환

v2는 붕괴하지 않았고 wrong-reference 구분도 학습했지만 2,000 step에서
heldout paired-flow improvement가 `0.00183 ± 0.00287`에 그쳤다. 같은 시점의
소형 Dual-query baseline `0.00369`보다 낮다. Human/Synthetic teacher projection
coefficient도 `0.057/0.084`인 반면 common-output ratio는 `0.831/0.820`이었다.
1/2-reference는 일부 개선됐지만 4/8-reference는 `0.00038/0.00083`에 그쳤고,
고정 레퍼런스 시트에서는 주로 머리색과 채도만 달라졌다. 따라서 v2는
step-2,000 checkpoint를 보존하고 중단한다.

v3는 `84 -> 8 typed descriptor`와 descriptor별 reference pooling은 유지하되,
마지막 `8 -> 16` cross-attention을 네 개의 typed dense group head로 교체한다.

- spatial descriptor 4개: 2개씩 두 group
- global descriptor 2개: 한 group
- artist-summary descriptor 2개: 한 group
- 각 group은 `2x1024 -> 512 -> 4x1024` MLP로 네 개의 명시적 slot을 출력
- 네 group을 이어 `16x1024`로 만들고 마지막 LayerNorm과 고정 RMS `0.15` 적용

이 구조는 learned query의 공통 value를 출력하지 않으면서도 각 slot에 독립적인
조건부 출력 행렬을 준다. 전체 크기는 약 `21.1M`이다. Teacher update를 2/4-step
간격으로 줄일 때는 loss를 cadence만큼 보정해 평균 teacher gradient 세기를
유지한다. Common-output weight는 `0.02 -> 0.10`으로 올리고, 이미 구조적으로
분리된 slot에 대한 functional diversity는 3,000 step 이후로 늦춘다. v3 역시
2,000 step에서 소형 baseline과 정량·정성 비교한 뒤에만 8,000 step까지 연장한다.

### 1,000-step v3 중간 교정

v3는 500-step 패널에서 v2보다 다양한 색면·선화·명암 변화를 보였고 slot
diversity loss도 약 `0.99 -> 0.003--0.010`으로 개선됐다. 그러나 1,000 step에서
self improvement는 `0.00371`, wrong-reference는 `-0.00476`, artist retrieval은
`1.0`인 반면 heldout improvement는 `0.00023`에 불과했다. 4/8-reference도
`-0.00058/0.00047`이었고, controlled reference-view difference ratio는 `0.584`였다.
외부 고정 시트도 레퍼런스별 차이가 거의 없었다. 이는 정보를 읽는 능력은
확보했지만 개별 그림 정보를 같은 작가의 공통 효과로 정제하지 못한 상태다.

따라서 step-1,000 checkpoint에서 optimizer와 scheduler를 이어 다음과 같이
교정한다.

- functional probe cadence `4 -> 2`
- same-artist functional weight `0.005 -> 0.05`
- teacher projected-effect weight `0.50 -> 1.00`
- functional artist-teacher contrastive 시작 `2,000 -> 1,500`

이 변경은 token 자체를 억지로 같게 하지 않고, 동일 prompt/noise/timestep의
Frozen-Anima velocity residual이 두 disjoint same-artist reference view에서
일치하도록 한다. 1,250/1,500/2,000-step에서 heldout·multi-reference 성능과
reference-view difference가 함께 개선되는지 확인한다.

1,250-step 검증에서는 raw same-artist consistency 강화가 전 작가 공통 방향을
키우는 실패가 확인됐다. Human/synthetic common-output ratio가 각각
`0.731/0.659 -> 0.755/0.790`으로 악화되고 selection score도
`0.00365 -> 0.00303`으로 하락했다. 따라서 이 체크포인트는 폐기 후보로
보존하고 v3의 step-1,000에서 다시 분기한다. 새 v3c는 각 reference view에서
동일 prompt/noise/timestep을 공유하는 artist batch 평균 residual을 먼저 뺀 뒤,
남은 artist-specific residual끼리만 consistency를 계산한다. Weight는 과도했던
`0.05`에서 `0.01`로 낮추고 cadence 2와 projected teacher weight 1.0은 유지한다.
새 출력 디렉터리의 자동 resume에 의존하지 않고 v3 step-1,000 checkpoint와
그 시점까지의 history를 `initial_checkpoint`/`initial_history`로 명시한다.

### v3c 2,000-step gate

Centered functional consistency를 적용한 v3c는 1,750 step에서 heldout
improvement `0.00213`, correct-vs-wrong advantage `0.00525`, selection score
`0.00563`을 기록했다. 2,000 step에서는 human/synthetic teacher cosine이
`0.115/0.138`, projection coefficient가 `0.061/0.077`까지 개선됐고
wrong-reference improvement는 `-0.00513`이었다. 구조 붕괴 없이 artist teacher
방향을 학습한다는 점은 검증됐다.

그러나 2,000 step heldout improvement는 `0.00079`로 다시 낮아졌으며,
1/2-reference는 `0.00136/0.00398`인 반면 4/8-reference는
`-0.00113/-0.00060`이었다. Controlled common-output ratio도 `0.814`, 두
reference view의 difference ratio도 `0.579`였다. Fixed-reference 시트에서는
reference별 머리색·피부색·채도 차이는 나타났지만 얼굴 비율·선화·명암은 공통
Anima 출력에 남아 있었다. 따라서 2,000 step은 안정성 gate만 통과했고 최종
스타일 재현 gate는 통과하지 못했다. Artist contrastive ramp가 끝나는 3,000
step까지만 한정 연장해 common-output, 4/8-reference, fixed-reference 표현이
개선되는지를 확인한 뒤 최대 8,000-step 연장 여부를 다시 결정한다.

### 3,000-step gate와 v4 계획

v3c의 3,000-step 결과는 heldout improvement `0.00140 ± 0.00291`,
correct-vs-wrong advantage `0.00484`였다. Human/synthetic teacher cosine은
`0.115/0.141`, projection coefficient는 `0.057/0.073`까지 올랐지만
common-output ratio가 `0.761/0.730`으로 여전히 높았다. Reference 수별
improvement도 2장은 `0.00330`인 반면 1/4/8장은 각각
`-0.00034/-0.00048/-0.00072`였다. 외부 고정 레퍼런스에서는 색상 정도만
달라지고 얼굴·선화·명암이 공통 Anima 출력에 남았다. 따라서 현 구조를 그대로
8,000 step까지 연장하지 않는다.

Grouped head의 sample-independent slot embedding을 추론 시 제거한 진단도
heldout `0.00135`, common-output `0.759/0.728`, 4/8-reference
`-0.00057/-0.00070`로 사실상 동일했다. Additive slot bias 하나가 원인은
아니며, 주된 문제는 teacher가 항상 4-reference에만 적용되는 반면 실제 flow
학습은 1-reference가 45%인 서로 다른 reference-count 분포를 사용한다는 점이다.

다음 v4는 v3c step-3,000을 출발점으로 다음 한 가지만 우선 검증한다.

- Human/synthetic centered teacher 한 번의 Anima forward 안에서 reference 수를
  update마다 `1 -> 2 -> 4`로 순환한다.
- 모든 reference 수가 같은 작가에 대한 동일한 centered native effect의 방향과
  절대 투영 크기를 배우게 한다. Teacher loss 세기와 cadence는 유지한다.
- Teacher update가 사용한 reference 수를 로그에 남기고 validation에서도
  1/2/4-reference teacher 성능을 분리해 확인한다.
- 1,000 step만 추가 학습하여 step-4,000에서 heldout improvement,
  teacher projection, 1/2/4/8-reference 성능과 외부 고정 시트를 비교한다.
- 4/8-reference와 고정 시트가 함께 개선될 때만 8,000 step까지 연장한다.
  개선되지 않으면 단순 loss 증량 대신 reference pooling에 zero-init consensus
  residual을 추가하는 구조 변경으로 전환한다.

### 4,000-step v4 gate와 v5 전환

Reference-count teacher curriculum은 1/2/4-reference의 teacher projection을
비슷하게 맞추는 데는 성공했지만 최종 성능을 개선하지 못했다. Step 4,000의
heldout improvement는 `0.00050 ± 0.00393`이었고 1/2/4/8-reference는 각각
`0.00082/0.00321/0.00032/-0.00033`이었다. Controlled common-output ratio도
`0.806`으로 높았으며 외부 고정 시트는 일곱 reference 모두 거의 같은 얼굴,
선화와 광원을 출력했다. 따라서 v4는 보존하되 8,000 step으로 연장하지 않는다.

추가 진단에서 전체 reference cache는 5,000명(Train 4,000 / Validation 500 /
Test 500)을 포함하지만 centered teacher bank는 500명(Train 450 / Validation
25)뿐임을 확인했다. 21.1M tokenizer가 이 작은 직접 지도 집합을 분류·암기하고
unseen 시각 스타일로 일반화하지 못하는 것이 single-reference 실패의 더 직접적인
원인이다. 기존 500-bank의 artist median effect RMS도 `0.019--0.070` 범위여서
약한 artist tag를 제거하는 것으로 해결될 문제는 아니다.

v5는 구조 변경과 teacher coverage 변경을 섞지 않고 다음과 같이 검증한다.

- 전체 5,000명에 대해 4 content x 8 timestep centered native-effect bank를
  한 번 캐시한다. 예상 크기는 약 20 GiB이다.
- Human teacher loader는 Train 4,000명을 모두 사용한다. Synthetic reference
  cache는 존재하는 기존 500명과 bank의 교집합만 사용하며 두 도메인을 같은
  스타일로 취급하지 않는다.
- Tokenizer는 v4 구조를 처음부터 학습하되 불필요한 additive slot bias는 끈다.
  1/2/4-reference curriculum, multimode prompt와 고정 output RMS는 유지한다.
- 우선 4,000-step gate를 수행한다. Heldout 자체, 4/8-reference 성능과 외부
  고정 시트가 함께 개선될 때만 같은 optimizer 상태로 8,000 step까지 연장한다.
- Coverage를 늘려도 외부 single-reference가 개선되지 않을 때에만 다음 실험에서
  reference mean/std consensus residual을 zero-init으로 추가한다.

## 통합 실행 계획 요약

목표는 Dual-query Resampler가 보존한 공간·전역·작가 정보를 다시 하나의 공통
벡터로 압축하지 않고, Anima가 실제로 사용할 수 있는 소수의 스타일 토큰으로
변환하는 것이다.

### 모델

- Dual-query Resampler는 동결하고 이미지마다 `spatial/global/artist-summary`
  typed descriptor를 출력한다.
- Reference별 descriptor는 타입과 reference identity를 유지한 memory set으로
  구성한다. Reference를 먼저 평균내거나 하나의 artist vector로 축소하지 않는다.
- 타입별 작은 pooling head가 reference consensus와 reference-specific residual을
  함께 읽고, 명시적 slot identity를 가진 최종 `16 x 1024` token을 만든다.
- Anima 연결부는 별도 깊은 style branch 없이 기존 LLM-adapter token sequence에
  이 16개 token을 주입하는 단순한 구조를 사용한다. 최종 token RMS는 강제로
  고정하지 않는다.
- 우선 frozen Resampler로 tokenizer만 검증한다. Resampler 공동학습은 tokenizer의
  heldout 성능과 정성 샘플이 검증된 이후의 별도 단계로 둔다.

### 데이터와 프롬프트

- 전체 Train artist를 사용하고 artist 단위 validation/test 분리를 유지한다.
- Reference 수는 1장이 가장 많고 2장이 그다음이 되도록
  `1/2/3/4/5/6/7/8 = 45/25/12/7/4/3/2/2%`로 샘플링한다.
- Prompt mode는 Full 30%, tag dropout 40%, short 20%, empty 10%로 구성한다.
- Empty는 quality prefix 없이 반드시 단일 `reference = target`으로 학습한다.
- Empty가 아닌 배치에는 quality prefix 포함/미포함을 섞는다. 모든 prompt와
  cache에서 작가명 및 `@artist` 누출을 금지한다.

### 지도 신호와 커리큘럼

- 기본 목표는 rectified-flow MSE다. 초기에는 exact-self와 Human/Synthetic
  centered artist teacher를 조밀하게 사용해 style effect의 방향과 절대 투영
  크기를 먼저 학습한다.
- 같은 작가의 서로 다른 reference view에는 token 자체가 아니라 Frozen Anima가
  만든 artist-centered flow residual의 일관성을 적용한다.
- Artist contrastive와 correct-vs-wrong cyclic ranking은 teacher alignment와
  heldout paired-flow가 안정된 뒤 ramp한다. Wrong reference를 망가뜨리는 것이
  아니라 correct reference의 residual이 더 정확하도록 margin을 둔다.
- Common-output penalty는 작은 출력 자체를 보상하지 않는다. Teacher-aligned
  절대 효과 크기를 보존하면서 artist 간 공통 성분만 억제한다.
- Reconstruction, 강한 token prototype, attention-map diversity처럼 생성 기능과
  직접 연결되지 않은 보조 loss는 사용하지 않는다.

### 실행과 선택 기준

- 먼저 전체 5,000명 centered teacher bank를 캐시하고 Human Train 4,000명을
  직접 지도한다. Synthetic은 이용 가능한 500명 교집합을 별도 도메인으로 쓴다.
- 새 모델은 처음부터 학습하며 4,000 step을 1차 gate로 삼는다.
- 250/500 step 간격의 정량 validation과 500/1,000 step 간격의 panel 및 외부
  fixed-reference 샘플을 함께 확인한다.
- heldout paired-flow, teacher projection, 1/4/8-reference 성능이 개선되고,
  fixed-reference에서 색상뿐 아니라 선화·명암·형태가 reference별로 달라질 때만
  optimizer 상태를 이어 8,000 step까지 학습한다.
- 4,000-step gate를 통과하지 못하면 단순히 loss나 모델 크기를 늘리지 않는다.
  Reference consensus와 reference-specific residual이 어디서 소실되는지 측정한
  뒤 zero-init consensus residual 같은 한 가지 구조 변경만 분리해 검증한다.
