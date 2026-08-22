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

## v29 clean bootstrap correction

v28은 Common-only 500스텝에서 cosine `0.576`, projection `0.227`인 상태로
Common을 동결했다. 이후의 common projection `1.29`는 Common 단독값이 아니라
`Common + mean(Artist)`였으므로, Artist가 아직 남은 공통 오차를 흡수했다. 이를
본 curriculum의 초기값으로 사용하지 않는다.

- v28 `step-0000500.pt`에서 다시 시작한다. 이 시점에는 Reader와 Artist K/V가
  아직 gradient를 받지 않았으므로 Common만 이어 학습할 수 있다.
- Common은 고정 스텝에 열지 않는다. final velocity 기준 cosine `0.70`, native
  projection `0.50`, RMS ratio `0.70~1.30`을 250-step validation 두 번 연속
  만족한 뒤에만 동결하고 Artist를 연다.
- 전환 시점의 Common-only 지표를 고정해 이후 합성 출력의 Artist 평균값과
  혼동하지 않는다.
- Artist 단계의 `artist_mean_weight`는 `0`이다. Centered direction, 약한 RMS
  band, 16-way all-wrong InfoNCE만으로 작가 residual을 학습한다.
- Artist teacher batch는 1/2/4개 reference를 55%/30%/15%로 섞는다. 네 장만
  사용하면 Reader가 reference-set 평균에 의존해 단일 레퍼런스 inference에서
  Common 출력으로 회귀하므로, 실제 주요 사용 조건인 한 장을 가장 자주
  노출하면서 다중 레퍼런스 이득도 보존한다.
- 전체 bootstrap은 Artist cosine `0.28`, projection `0.40`, InfoNCE gap `0.0`
  이상을 두 번 연속 만족해야 종료한다. 최대 5,000스텝 안에 통과하지 못하면
  본 curriculum으로 자동 진입하지 않고 병목을 다시 진단한다.
- 추가 1,250스텝에서 Common cosine은 `0.766`까지 올랐지만 RMS ratio가
  `0.612`에 머물렀다. 이때 magnitude 항은 전체 Common objective의 약 9%에
  불과했으므로, 저장된 optimizer 상태에서 magnitude weight만 `0.25→0.75`로
  올려 방향 정렬을 유지하면서 radial convergence를 가속한다.

통과한 체크포인트만 별도의 본 curriculum 실행에 사용한다. 본 단계는
Exact-Self 비중이 높은 구간에서 시작해 target 포함 1~4장, 포함률 anneal,
target-excluded 1~8장 순서로 진행하며 flow/정성 샘플이 악화되면 전환을 보류한다.

## v30 reference-count correction

v29의 4-reference Artist 상태에서 1-reference 배치를 즉시 55%로 바꾸자
step 3,000→3,500 동안 native cosine `0.230→0.200`, projection
`0.397→0.329`, InfoNCE gap `-0.039→-0.066`으로 악화됐고 functional common
output ratio는 `0.858→0.910`으로 증가했다. 따라서 이 구간은 채택하지 않는다.

- v29 `step-0003000.pt`에서 새 output directory로 분기한다.
- 3,001~4,000스텝 동안 1/2/4-reference 비율을 `10/15/75%`에서
  `50/30/20%`로 선형 보간한다. 3-reference는 사용하지 않는다.
- all-wrong InfoNCE weight는 정보가 불안정한 1/2-reference에서 각각
  `0.35/0.30`, 4-reference에서 `0.20`으로 둔다.
- 기존 multi-reference 검증과 별도로 정확히 한 장의 heldout reference를
  사용한 artist-effect/prototype 지표를 기록한다.
- 동일 prompt/seed의 fixed-reference 7개 출력에 대해 pairwise pixel RMS와
  baseline 변형량 대비 diversity ratio를 기록한다. 출력 크기가 있어도
  reference별 결과가 같으면 실패로 판정한다.

## v31 artist-null residual correction

v30 step 3,500의 component ablation에서 Common-only 변형 RMS는 `0.045`,
Artist-only는 `0.177`이었고 Artist-only 7개 결과도 거의 같았다. 따라서
명시적 Common branch가 Artist 차이를 덮은 것이 아니라, raw Artist attention
자체의 reference-independent 성분이 centered loss의 사각지대에 남은 것이다.

- Artist K/V에 28개 trainable null token을 같은 projection/softmax로 통과시킨다.
- 최종 Artist 성분은 `attention(reference) - attention(artist_null)`로 정의한다.
  초기 null token은 0이라 v29 step-3000 동작을 보존한다.
- frozen Common은 유지하며 `Common + mean(Artist)`가 frozen native common을
  맞추는 objective를 0.5배로 켠다. 이 항이 artist-null에 직접 gradient를 주어
  Artist 평균을 제거하고, centered direction/InfoNCE는 작가별 차이를 보존한다.
- 아키텍처에 새 optimizer group이 추가되므로 v29 모델 상태만 가져오고
  optimizer는 새로 초기화한다.
- fixed-reference diversity와 단일-reference common-output ratio가 실제로
  감소하지 않으면 이 구조도 채택하지 않는다.

## v32 centered Artist strength

v31 step 3,500에서 Artist-only 1×는 fixed-reference diversity ratio `0.704`,
2×는 `0.810`이었다. 2×에서는 이미지 붕괴 없이 선화, 얼굴 비율, 명암 차이가
실제로 드러났다. 이는 Artist 방향이 완전히 소실된 것이 아니라 native artist-tag
scale에 맞춘 centered residual이 style transfer에는 약하다는 뜻이다.

- v31 `step-0003500.pt`의 모델과 optimizer 상태를 이어받는다.
- frozen Common은 1×로 유지하고 null-centered Artist residual만 2×로 고정한다.
- Artist magnitude band는 native-centered teacher의 `1.25–2.25×`로 넓힌다.
  방향/InfoNCE 목표는 변경하지 않는다.
- 4,000스텝 fixed-reference 1×(내부 Artist gain 2×)가 레퍼런스별 차이를
  유지하면서 망가지지 않는지 확인한다. 다시 공통 출력으로 모이면 중단한다.
