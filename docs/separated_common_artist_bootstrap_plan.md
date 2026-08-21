# Separated Common / Artist Bootstrap

## 목적

Native Anima 작가 효과를 Q-conditioned 공통 기반과 reference-conditioned
작가 residual로 분해한다. 두 경로의 softmax와 파라미터를 분리해 공통 출력이
작가 경로의 gradient를 잠식하지 못하게 하면서, 최종 효과는 동일한 frozen
Anima Q/O/`gate_cross` 좌표계에서 합친다.

## 모델

- Common branch: reference와 Reader를 사용하지 않는 16개 direct K/V latent를
  4개 block group별로 둔다. 현재 Anima Q가 읽으므로 출력은 prompt, noisy
  latent, timestep, spatial position에 따라 달라진다.
- Artist branch: frozen Dual-query Resampler cache를 읽는 typed Reader와 4개
  shared Xavier K/V base, block별 rank-64 delta를 유지한다.
- Common/Artist는 separate softmax를 사용하고 attention 결과를 더한 뒤 같은
  pretrained O와 `gate_cross`를 통과한다.
- 학습 가능한 common/artist scalar gain은 두지 않는다. 기존 native-effect
  alpha calibration과 추론용 style strength만 사용한다.

## Phase A: Common bootstrap

- Step 1–500.
- Reader와 Artist K/V를 완전히 동결한다.
- Synthetic Anima artist-tag controlled batch의 raw native effect에서 복원한
  global common target만 학습한다.
- Final velocity cosine과 RMS band만 사용한다.
- Common K/V LR은 `3e-4`; 400스텝까지 native RMS의 0.9배 하한에 도달한다.

## Phase B: Artist residual bootstrap

- Step 501–2,000.
- Common K/V를 완전히 동결하고 Reader와 Artist K/V만 연다.
- Centered artist direction, 약한 RMS band, 16-way all-wrong InfoNCE를 사용한다.
- `frozen Common + mean(Artist)`가 raw native common을 유지하도록 common
  objective를 0.1배로 적용한다. 이는 실제 frozen Common 뒤에 남은 평균 오차를
  Artist residual이 약하게 교정하게 하면서 centered 방향을 주 목표로 유지한다.
- Common/Artist/Reader의 gradient RMS와 활성 phase를 별도로 기록한다.

## 후속 curriculum

Bootstrap 체크포인트를 검증한 뒤 별도 실험에서 진행한다.

1. Exact-Self 비율이 높은 짧은 정렬 구간.
2. 1–4개 reference에서 target을 항상 포함하되 Exact-Self와 다른 self 포함
   reference를 혼합한다.
3. Self 포함 확률을 선형으로 낮추면서 1/2-reference를 가장 자주 사용한다.
4. 마지막에는 target 미포함 1–8 reference로 전환한다.

전환은 고정 step만으로 결정하지 않고 centered artist cosine/projection,
all-wrong margin, common-output ratio와 fixed-reference 정성 샘플을 함께 본다.
