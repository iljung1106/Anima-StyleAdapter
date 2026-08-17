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
