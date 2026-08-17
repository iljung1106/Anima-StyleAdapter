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
5. 최종 query끼리 self-attention하지 않으며, 출력 직전에 slot embedding을
   다시 더한다. 현재 모델의 공통 full-rank output projection은 사용하지 않는다.
6. Final LayerNorm 뒤에는 descriptor로 예측한 bounded sample gain을 사용한다.
   초기 중심은 0.15이며 대략 0.09--0.25 범위로 제한한다. 추론 시 별도의
   style strength를 적용할 수 있다.

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
