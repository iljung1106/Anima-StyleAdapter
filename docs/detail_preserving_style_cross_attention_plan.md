# Detail-preserving typed-slot Style Cross-Attention v1 구현 계약

## 1. 목표와 현재 상태

현재 가장 좋은 정성 결과를 낸 경로는 frozen Dual-query Resampler의
`84 x 1024` 출력을 작은 Style Tokenizer가 `16 x 1024` post-LLM context
token으로 압축하여 Anima의 원래 text cross-attention에 삽입하는 방식이다.
이 방식은 작동하지만 다음 한계가 확인되었다.

- style token이 text token과 같은 K/V 및 하나의 softmax를 공유한다.
- visual token이 native text-context 분포를 흉내 내야 한다.
- `84 -> 16` 압축에서 세부 정보가 소실되고 일부 공통 스타일 방향으로
  수렴한다.
- 강도를 높이면 고유 화풍뿐 아니라 구도ㆍ색ㆍ표정 누출도 함께 증가한다.

다음 실험은 순수 token injection을 대체하는 별도 Style Cross-Attention을
구축한다. 현재 `dual_query_resampler_bprime_v8/step-010000` Resampler는
동결하고 이미 생성된 `84 x 1024` BF16 cache를 그대로 사용한다.

이 문서는 첫 10k 학습에 사용할 v1 구현 계약이다. 아래에서 `확정`으로 적은
항목은 첫 구현 중 임의로 바꾸지 않는다. 변경은 별도 실험 이름과 config로만
수행한다. 아직 구현 완료를 의미하지는 않는다.

## 2. 확정한 설계 원칙

1. Anima의 native Q와 full-rank O는 동결하여 재사용한다.
2. Text와 style은 동일한 Q를 쓰되 서로 다른 K/V와 별도 softmax를 사용한다.
3. Style K/V는 native text K/V 복사본이 아니라 블록별 신규 full-rank
   Xavier 초기화 행렬로 만든다.
4. Anima의 native timestep-conditioned `gate_cross(t)`는 그대로 재사용한다.
5. 블록별 style 강도는 첫 10k 동안 학습하지 않고 고정한다. 자유롭게 0으로 수렴할
   수 있는 gate를 두지 않는다.
6. No-style은 learned null token이 아니라 style branch의 정확한 우회다.
7. 출력 크기 하한은 raw RMS가 아니라 유효한 Teacher 방향으로의 투영에 둔다.
8. Reference 순서는 의미가 없으며, 여러 Reference는 같은 canonical slot끼리
   Set Attention으로 합친다.
9. Spatial slot을 고정 위치에 묶기보다 이미지 안의 선ㆍ채색ㆍ눈ㆍ머리카락ㆍ
   질감 같은 세부 정보를 content-adaptive하게 읽게 한다.
10. 첫 구현은 reference별 canonical slot `28개`, Reader `2 block`, cross-slot
    Mixer `1 block`으로 고정한다.
11. Style K/V의 Linear는 `bias=False`다. 특히 reference와 무관한 공통 residual을
    만들 수 있는 V bias를 두지 않는다.
12. 작가 태그와 작가명은 Synthetic Teacher context에만 허용한다. Student prompt,
    Human 학습 prompt, validation prompt와 Student text cache에는 넣지 않는다.
13. 최종 28개 style token에는 sample 공통 RMS 고정, 강제 global normalization,
    학습 가능한 단일 global gain을 두지 않는다. 내부 pre-norm만 사용한다.

## 3. 전체 데이터 흐름

```text
Frozen Dual-query Resampler cache, reference별 84 x 1024
  spatial 64 ─ separate LN/projection + type + weak 2D position ┐
  global  16 ─ separate LN/projection + type                   ├─ concatenate
  summary  4 ─ separate LN/projection + type                   ┘
                                      |
                     28 canonical learned latent slots
                                      |
                    Perceiver-style reader, 2 blocks
                                      |
                       reference별 28 x 1024
                                      |
         같은 slot끼리 permutation-invariant Set Attention
                                      |
                     cross-slot Transformer, 1 block
                                      |
                         최종 28 x 1024 style set
                                      |
           Anima 28 blocks의 신규 Style K/V + native Q/O
```

## 4. Reference별 detail-preserving reader

### 4.1 입력 타입 처리

입력 84개를 처음부터 동일한 분포로 간주하지 않는다.

- spatial `64`: 별도 LayerNorm과 input/K/V projection을 사용한다.
- global `16`: 별도 LayerNorm과 input/K/V projection을 사용한다.
- artist-summary `4`: 별도 LayerNorm과 input/K/V projection을 사용한다.
- 세 타입에 서로 다른 type embedding을 더한다.
- spatial memory에만 원래 `8 x 8` 좌표의 2D position 정보를 제공한다.

위치는 output slot을 특정 좌표에 고정하는 용도가 아니다. Attention K 또는
attention bias에서 약한 보조 정보로만 사용하며, V가 전달하는 시각 정보에는
위치를 강제로 섞지 않는다.

타입별 전처리 후에는 세 memory를 연결하고 자유로운 cross-type attention을
허용한다. Type embedding 하나만 붙이고 raw token을 바로 섞는 구성은 세 타입의
통계와 역할 차이를 무시하므로 사용하지 않는다.

### 4.2 Canonical latent slot

Reference 하나당 `28 x 1024` canonical latent slot을 사용한다.

- 모든 slot은 고유한 learned identity를 가진다.
- 출력 slot은 고정 spatial grid가 아니다.
- 각 slot은 필요하면 spatial/global/summary를 모두 읽을 수 있다.
- 초기 soft type preference는 `16 spatial / 8 global / 4 summary`로 둔다.
- 이 preference는 hard mask가 아닌 학습 가능한 attention-logit bias다.
- Slot identity는 style value에 그대로 누적하지 않는다. 각 reader block에서
  `LN(content state) + slot identity`를 Q로 사용하고, memory에서 읽은 value만
  content state에 residual로 더한다. 이로써 canonical slot은 유지하면서 모든
  reference에 공통인 learned identity 자체가 Style K/V로 전달되는 것을 막는다.

Reader는 model dim `1024`, `16` attention head의 pre-norm Perceiver-style
cross-attention `2 block`으로 확정한다. 각 block은 typed-memory cross-attention,
slot self-attention, SwiGLU FFN으로 구성한다. Spatial 2D sin/cos position은
spatial K에만 작은 고정 gain `0.1`로 더하고 V에는 더하지 않는다.

### 4.3 Training-only typed reconstruction

28개 slot이 당장 flow loss에 유용한 몇 개 공통 방향만 남기고 세부 정보를
버리는 것을 막기 위해 작은 decoder를 학습 중에만 사용한다.

```text
28 slots
  -> 64 fixed spatial reconstruction queries
  -> 16 fixed global reconstruction queries
  ->  4 fixed summary reconstruction queries
```

- 원본 픽셀이나 Qwen VAE latent를 다시 복원하지 않는다.
- Set Attention 전의 reference별 28개 slot에서 Frozen Resampler가 낸 84개
  token만 타입별 cosine + 약한 Huber로 복원한다.
- Reconstruction은 전체 loss의 작은 보조 항으로 시작하고 후반에는 감소시킨다.
- Decoder는 추론 모델에 포함하지 않는다.

이는 detail 보존에 유리하지만 content leakage도 키울 수 있으므로 주 loss로
사용하지 않는다.

## 5. Multi-reference aggregation

Reference별 출력은 `[B, R, 28, 1024]`다. Reference를 token 축으로 평평하게
섞지 않고 각 canonical slot index에 대해 R축 Set Attention을 수행한다.

```text
z_s = MHA(q = set_query_s, kv = {z_1,s, z_2,s, ..., z_R,s})
```

- Reference order embedding은 사용하지 않는다.
- 유효 Reference mask를 항상 적용한다.
- 단일 Reference도 별도 bypass 없이 동일한 경로를 통과한다.
- Slot마다 Reference 신뢰도가 다를 수 있어야 한다.
- Reference 수가 늘어나도 출력 token 수는 항상 28개다.
- `set_query_s`의 slot identity는 Q에만 사용하고 최종 value에 직접 더하지 않는다.

Set Attention 뒤에는 pre-norm cross-slot Transformer `1 block`을 둔다.
이는 같은 slot 정렬의 작은 오류를 보정하고 spatial/global/summary 정보를 최종
스타일로 조합한다. 이 block에서도 slot identity는 attention Q/K에만 사용하고
V와 최종 style value에는 직접 더하지 않는다. 2 block은 v1에 사용하지 않는다.

## 6. Anima Style Cross-Attention

각 Anima block에서 text와 style은 cross-attention 직전의 같은 normalized hidden
state와 동일한 native Q를 사용한다.

```math
Q_b = W^Q_b(Norm_b(x_b))
```

```math
A^text_b  = Attn(Q_b, K^text_b,  V^text_b)
```

```math
A^style_b = Attn(Q_b, K^style_b, V^style_b)
```

Text와 style의 softmax denominator는 공유하지 않는다. Style K/V는 각 block마다
`1024 -> 2048` full-rank Linear로 새로 만들고 Xavier uniform 초기화한다. 두
Linear 모두 `bias=False`다. Anima의 native `k_norm`, `v_norm`, attention backend와
head layout을 그대로 적용한다.

실제 Anima는 projected V에도 `v_norm`을 적용하므로 Style Tokenizer의 token RMS를
키우는 것 자체는 안정적인 strength 조절 수단이 아니다. 최종 강도는 `alpha_b`,
전역 `s`, attention 방향과 functional loss로 만든다. 최종 token RMS를 강제로
고정하지 않는 이유는 강도를 키우기 위해서가 아니라 불필요한 표현 제약을 없애기
위해서다.

MLP 전 residual은 다음 한 가지 방식으로 구현한다.

```math
x_b <- x_b + gate^cross_b(t) O_b(A^text_b + s alpha_b A^style_b)
```

- `s`: 사용자가 조절하는 전역 style strength
- `alpha_b`: optimizer에 넣지 않는 고정 block별 계수
- `gate_cross(t)`: frozen Anima의 native channel/timestep gate

Native O는 위 식처럼 정확히 한 번만 호출하여 bias도 한 번만 적용한다. Q도
block당 한 번만 계산하며 text residual을 더한 뒤 다시 계산하지 않는다. `s=0`일
때 이 경로는 native text cross-attention과 수치적으로 동일해야 한다.

28개 신규 full-rank K/V는 약 117M parameter다. Reader와 Set Aggregator를 합친
전체 trainable 규모는 약 140--160M을 예상한다. Anima와 Resampler는 동결한다.

## 7. 고정 block strength와 calibration

28개 block에 임의의 같은 alpha를 적용하지 않는다. Synthetic artist-tag Teacher와
Xavier 초기화 Student를 같은 calibration batch, `x_t`, timestep, Q에서 측정한다.

```math
m^teacher_b = median_(artist,content,t) ||T_b||
```

```math
m^student_b = median_(artist,content,t) ||S_b(alpha=1)||
```

```math
alpha_b = clamp(m^teacher_b / (m^student_b + eps), 0.02, 2.0)
```

`S_b(alpha=1)`은 `s=1`, weight-only native O와 native gate까지 적용한 residual이다.
`m_teacher_b`가 28개 block median의 `10%`보다 작은 block은 신뢰하기 어려운 것으로
보고 `alpha_b=0`으로 비활성화한다.
나머지 block의 alpha는 첫 10k 실험 내내 optimizer와 checkpoint의 trainable
parameter에서 제외한다. 추론에는 전역 strength `s`만 노출한다.

### 7.1 최소 목적함수 재시작(v6)

P75 hard-RMS 실험은 각 블록의 `gate_cross(t) * O(style attention)` RMS를
안정적으로 고정했지만, 최종 artist-specific velocity 방향을 만들지 못했다.
매 forward의

\[
r'_b=T_b(t)\frac{r_b}{\operatorname{RMS}(r_b)}
\]

정규화는 출력 크기 방향의 gradient를 제거했고, 모델은 raw token/K/V 크기를
키우면서도 최종 효과를 늘릴 수 없었다. 따라서 v6에서는 calibration을 초기
`alpha_b(t)` 설정에만 사용하고 학습 중 hard RMS 정규화는 사용하지 않는다.

작가 없는 동일 `x_t`, content, timestep의 frozen Anima 출력을 `v_null`, 스타일
출력을 `v_style`이라 하고, 학생 효과와 teacher 효과를 다음처럼 둔다.

\[
\Delta_s=v_{style}-v_{null},\qquad
\Delta_T=\text{centered native artist residual}
\]

한 teacher batch 안에서 학생과 teacher를 각각 artist 축으로 centering한다.
작은 출력 붕괴는 전체 RMS 하한이 아니라 teacher 방향의 signed projection으로
막는다.

\[
a=\frac{\langle\Delta_s,\Delta_T\rangle}
        {\|\Delta_T\|^2+\epsilon},\qquad
L_{floor}=[\rho_{min}-a]_+^2
\]

`rho_min`은 step 1의 `0.25`에서 step 1,000의 `1.0`까지 선형 증가한다. 공통
출력이나 teacher와 직교한 잡음은 이 하한을 만족시킬 수 없다. 직교 성분은

\[
\Delta_\perp=\Delta_s-a\Delta_T,\qquad
L_\perp=\left[\frac{\|\Delta_\perp\|}{\|\Delta_T\|+\epsilon}-0.5\right]_+^2
\]

로 제한한다. 별도의 전체 RMS 하한/고정은 두지 않는다.

초기 학습 목적함수는 다음 다섯 항만 사용한다.

\[
L=1.0L_{flow}
 +0.20L_{teacher\ residual\ Huber}
 +0.15L_{floor}
 +0.05L_{\perp}
 +0.01L_{reconstruction}
\]

teacher residual Huber는 각 행의 teacher RMS로 정규화한다. native timestep
통계의 `[0.75, 1.33]` 가중은 loss 종류가 아니라 teacher/flow sampling 보정으로
그대로 유지한다. 초기 500 step은 teacher를 매 step, 이후에는 두 step마다
사용한다.

다음 항은 끄고 지표 또는 후속 실험으로만 남긴다.

- internal block teacher
- same-artist consistency
- centered-energy 및 common-output penalty
- artist contrastive 및 artist ranking
- correct-vs-wrong functional ranking
- projection coefficient ceiling

heldout paired improvement가 통계적으로 양수가 되기 전에는 위 보조항을 다시
켜지 않는다. 이 최소 실험에서 projection coefficient가 증가하지 않으면 loss
간 충돌이 아니라 Reader/KV 표현력 또는 주입 경로의 구조적 문제로 판단한다.

K/V가 출력을 작게 만드는 방식으로 alpha를 우회할 수 있으므로 고정 alpha만으로
붕괴 방지가 끝나지는 않는다. 아래의 Teacher-aligned magnitude loss를 함께 쓴다.

## 8. No-style와 CFG 계약

No-style 조건에서는 branch를 완전히 건너뛰어 frozen Anima와 수치적으로 같은
출력을 보장한다.

- Learned null style token을 첫 구현에 두지 않는다.
- Style dropout은 해당 sample의 branch를 우회한다.
- `s=0`도 동일한 우회 경로를 사용한다.
- Style attention을 512개 text 길이에 맞추기 위한 zero padding은 하지 않는다.
  정확한 28개 valid style token만 별도 softmax에 넣는다.

Text CFG가 style을 의도치 않게 동일 배율로 증폭하지 않도록 conditional과
unconditional prediction 양쪽에 같은 style condition을 제공하는 것을 기본으로
한다.

```math
v = v_uncond,style + cfg_text (v_cond,style - v_uncond,style)
```

Style 강도는 branch 내부의 `s`로 조절한다. Text/style의 완전히 독립적인 CFG가
필요한 정밀 비교에서는 다음 3-prediction 방식을 사용한다.

```math
v = v_0 + cfg_text (v_text - v_0) + cfg_style (v_text,style - v_text)
```

- `v_0`: unconditional, no-style
- `v_text`: conditional text, no-style
- `v_text,style`: conditional text + style

기본 샘플은 속도가 빠른 2-prediction과 branch 내부 strength `s`를 사용하고,
validation의 고정 subset에서는 3-prediction 결과도 함께 기록하여 style/text
interaction을 확인한다. 학습 자체에는 CFG를 적용하지 않는다.

## 9. Frozen Anima internal Teacher

Synthetic 데이터에서는 Anima가 이미 알고 있는 `@artist` 반응을 직접 Teacher로
사용할 수 있다. 작가명이 괄호를 포함해도 prompt 강조 문법으로 해석되지 않도록
기존 artist-tag escaping 함수를 반드시 사용한다. 가장 중요한 계약은 Teacher와
Student를 같은 Q에서 비교하는 것이다.

Student trajectory의 현재 block hidden에서 Q를 한 번 계산하고, 그 Q에 대해
content-only와 `content + @artist` native text context를 각각 적용한다.
아래의 `O_bar(y)=W_O y`는 native O의 weight-only 선형 변환이며 bias를 포함하지
않는다. Residual 차이에서 native O bias는 상쇄되어야 한다.

```math
T_b(Q_b) = gate_b(t) O_bar_b[
  Attn(Q_b, K_b^(content+artist), V_b^(content+artist))
  - Attn(Q_b, K_b^content, V_b^content)
]
```

```math
S_b(Q_b) = gate_b(t) O_bar_b Attn(Q_b, K_b^style, V_b^style)
```

Raw text K/V는 회귀하지 않는다. 동일한 attention 결과를 만드는 K/V는 유일하지
않기 때문이다. 다음을 Teacher target으로 사용한다.

- 선택된 block의 attended/output residual 방향과 절대 크기
- 전체 Anima의 최종 velocity residual
- 같은 artist가 다른 content에서 만드는 centered residual의 일관성

최종 velocity Teacher와 Student effect는 동일한 `x_t`, timestep에서 다음처럼
정의한다.

```math
T^velocity = stopgrad[
  v^native(x_t,t,content+artist) - v^native(x_t,t,content)
]
```

```math
S^velocity = v^student(x_t,t,content,style) - v^native(x_t,t,content)
```

Native Teacher는 synthetic Anima image에만 직접 적용한다. Human reference에는
정확한 artist-tag Teacher가 없으므로 일반 flow loss, same-artist consistency와
target-inclusive bootstrap을 사용한다. Human exact-self bootstrap에서는
`stopgrad(v_target-v_base)`를 약한 final-velocity 방향 신호로만 사용할 수 있지만,
이를 block-level artist Teacher나 순수 화풍 정답으로 간주하지 않는다. Target이
빠진 뒤에는 이 보조 방향도 제거한다. Human과 synthetic reference를 같은 이미지
분포라고 강제하지 않는다.

Teacher reference와 target은 가능한 한 content와 seed를 교차한다.

```text
reference: artist A, content 1, seed 1
target:    artist A, content 6, seed 4
teacher:   content 6 + @artist A, seed 4
```

같은 content/seed만 반복하면 style 대신 이미지 복사를 학습할 수 있다.

`@artist` 또는 작가명은 위 Teacher context 두 곳 중 `content+artist`에만 존재한다.
Student의 content context, target caption, Human/Synthetic reference token cache,
정성 샘플 prompt에는 작가명이 없어야 한다. Cache 생성 시 원문/정규화 prompt와
token id를 감사하고, validation 시작 전 작가명 누출 검사를 실패 조건으로 둔다.

## 10. 작은ㆍ공통ㆍ무의미한 출력 붕괴 방지

Teacher residual을 `T`, Student style residual을 `S`라 한다. 이 band는 신뢰할
수 있는 Synthetic internal/final Teacher에만 적용한다. Human sample에는 raw
output RMS 하한을 두지 않는다.

```math
u_T = T / (||T|| + eps)
```

```math
p = <S, u_T>
```

### 10.1 Teacher 방향 투영 하한

```math
L_floor = ReLU(rho_min(t) ||T|| - p)^2
```

### 10.2 Soft upper band

```math
L_upper = ReLU(p - rho_max(t) ||T||)^2
```

### 10.3 직교 성분 억제

```math
L_orth = ||S - p u_T||^2 / (||T||^2 + eps)
```

Floor는 모든 timestep과 block에 같은 절대값을 강제하지 않는다. 해당 Teacher
효과의 timestep/block별 크기에 상대적으로 적용하고, Teacher가 너무 약한 표본은
제외하거나 낮게 가중한다.

초기에는 방향 정렬을 먼저 만들고 projection floor를 천천히 높인다. v1의
고정 schedule은 다음과 같다.

- 초기 0--250 step: `rho_min=0`, 방향/flow 정렬 중심
- 250--1,000 step: `rho_min: 0 -> 0.5` 선형 ramp
- 1,000--10,000 step: `rho_min=0.5`, `rho_max=1.5`
- Teacher가 timestep/block별 하위 10 percentile보다 약한 표본은 band에서 제외

공통된 nonzero 출력도 실패이므로 동일 `x_t`, text, timestep에서 다음을 추가한다.

- 같은 artist의 서로 다른 reference residual consistency
- 다른 artist에 대한 centered residual contrastive
- Correct Teacher가 cyclic wrong Teacher보다 가까운 양방향 ranking
- Batch 공통 residual을 제거한 뒤 artist retrieval/margin

Wrong reference를 강제로 망가뜨리는 단방향 loss는 쓰지 않는다. Cyclic batch의
모든 artist가 동시에 자신의 Teacher에 맞도록 구성한다.

Functional 비교에서는 reference만 바꾸고 probe의 `x_t`, timestep, content text와
Q를 고정한다. Synthetic ranking은 다음과 같이 Teacher effect 공간에서 계산한다.

```math
L_rank = ReLU(
  margin + d(S_i,T_i) - d(S_cyclic(i),T_i)
)
```

Human ranking은 동일한 target flow에 대한 correct/wrong prediction error로
계산한다. Cyclic shift된 reference도 batch의 다른 sample에서는 자신의 correct
reference로 동시에 학습되므로 단순히 wrong 출력을 파괴하는 경로로 만들지 않는다.
Same-artist consistency도 서로 다른 reference를 같은 probe 조건에 넣은 effect끼리
비교하며, 서로 다른 content의 raw velocity를 직접 같게 만들지 않는다.

## 11. 전체 loss 구성

모든 항은 batch와 feature dimension으로 mean reduction하고, Teacher 크기로
나누는 항은 denominator를 clip하여 작은 Teacher를 증폭하지 않는다. v1의 loss와
초기 계수는 다음으로 확정한다.

```text
L = L_flow
  + 0.25 * L_internal_teacher
  + 0.10 * L_teacher_direction
  + 0.05 * (L_floor + L_upper)
  + 0.02 * L_orth
  + ramp(0, 0.10) * L_correct_vs_wrong
  + ramp(0, 0.05) * L_same_artist_consistency
  + ramp(0, 0.05) * L_centered_artist_contrast
  + decay(0.05, 0.01) * L_84_token_reconstruction
```

- `L_flow`: 실제 rectified-flow target에 대한 기본 loss
- `L_internal_teacher`: 같은 Q에서의 block별 native artist residual
- `L_teacher_direction`: final velocity/internal residual cosine 또는 normalized Huber
- `L_floor/L_upper`: Teacher 방향의 절대 작용 크기 band
- `L_orth`: 의미 없는 직교 출력 억제
- `L_correct_vs_wrong`: 공통 artist-independent 출력 억제
- `L_same_artist_consistency`: content가 달라도 유지되는 작가 효과
- `L_centered_artist_contrast`: batch 공통 effect를 제거한 residual의 supervised
  contrastive/retrieval margin
- `L_84_token_reconstruction`: detail 보존용 약한 training-only 항

`rank/consistency/centered contrast`는 500 step부터 1,500 step까지 선형 ramp한다.
Reconstruction은 0--2,000 step `0.05`, 이후 10,000 step까지 `0.01`로 선형
감소한다. 각 항의 gradient contribution이 주 학습부에서 `L_flow`의 0.1배 미만
또는 2배 초과가 100 step 지속되면 학습을 자동 변경하지 않고 경고와 진단
checkpoint만 남긴다.

Anima native artist effect의 timestep별 median profile `m(t)`로 다음 가중치를
미리 계산한다.

```math
w(t) = clamp(m(t) / mean_t[m(t)], 0.75, 1.33)
```

동일한 `w(t)`를 `L_flow`와 Synthetic Teacher 계열 loss에 적용한다. 따라서
Teacher가 강한 timestep을 보되 특정 구간만 과도하게 반복하지 않는다.

Hard token RMS normalization, raw output norm floor와 모든 block의 강제 동일 출력은
사용하지 않는다.

## 12. 10k 학습 curriculum

Resampler와 Anima는 10k 전체에서 동결한다. Reader, Set Aggregator, Mixer와 신규
full-rank K/V는 step 0부터 optimizer에 넣으며 자유 gate warmup은 두지 않는다.

### 12.1 Prompt mode

각 batch sample은 다음 분포를 사용한다.

- Full caption `30%`
- Tag dropout `40%`: rating/character/general tag group별 dropout과 token dropout
- Short caption `20%`: 핵심 subject/action/background tag만 유지
- Empty `10%`: quality prefix도 넣지 않고 반드시 단일 `reference=target`

Empty를 제외한 sample은 quality prefix 포함/미포함을 `50:50`으로 뽑는다. Negative
prompt는 학습 flow condition에 넣지 않는다. 작가명과 `@artist`는 Student prompt의
어떤 mode에도 들어가지 않는다.

### Phase A — 0--500: exact/reference influence bootstrap

- Reference는 한 장이며 target을 항상 포함한다.
- Human과 Synthetic을 artist-balanced하게 `50:50`으로 뽑는다.
- Synthetic internal/final Teacher를 매 optimizer step 사용한다.
- 28개 block 모두에 같은-Q internal Teacher를 계산한다.
- Correct-vs-wrong과 artist contrast는 아직 사용하지 않는다.
- Reader reconstruction과 flow/Teacher 정렬로 신규 K/V의 방향을 먼저 만든다.

### Phase B — 500--2,000: cross-content와 Set Attention

- Reference 수 `1/2/4`를 `50/35/15%`로 뽑는다.
- Target 포함 확률을 `1.0 -> 0.5`로 선형 감소시킨다.
- Reference와 target content/seed를 가능한 한 교차한다.
- Synthetic Teacher는 두 optimizer step마다 사용한다.
- Correct-vs-wrong, same-artist consistency, centered artist contrast를
  `500--1,500` step에서 ramp한다.

### Phase C — 2,000--6,000: target 제거

- Reference 수 `1/2/4/8`을 `45/30/17/8%`로 뽑는다.
- Target 포함 확률을 `0.5 -> 0`으로 선형 감소시킨다.
- Human/Synthetic 비율을 `70:30`으로 바꿔 실제 그림 일반화를 우선한다.
- Synthetic Teacher는 두 optimizer step마다 계속 사용한다.

### Phase D — 6,000--10,000: held-out-reference 본학습

- Target 포함 확률은 `0`으로 고정한다.
- Reference 수와 Human/Synthetic 분포는 Phase C를 유지한다.
- Reconstruction weight를 `0.01`까지 낮추고 functional/artist loss를 유지한다.
- Empty mode만 예외적으로 단일 exact-self를 유지하여 무지시 reference 경로가
  사라지지 않게 한다.

첫 2k는 pilot 판단 지점이지만 지표가 정상이라면 같은 run으로 10k까지 이어간다.
No-style 보존 실패, NaN, validation artist correct-vs-wrong 역전이 3회 연속 발생한
경우에만 중단한다. 단순 flow loss 정체만으로 자동 중단하지 않는다.

### 12.2 Optimizer

- Fused AdamW, BF16 autocast, `betas=(0.9, 0.95)`, `eps=1e-8`, weight decay `0.01`
- Reader/Set/Mixer/decoder: LR `1e-4`
- Block별 신규 full-rank K/V: LR `5e-5`
- 500 optimizer-step linear warmup 후 최종 LR 10%까지 cosine decay
- Global gradient norm clip `1.0`
- Alpha, Anima, Resampler는 optimizer와 optimizer checkpoint에서 제외

## 13. 데이터ㆍ캐시ㆍ효율 계약

- Frozen Resampler의 `84 x 1024` cache를 재사용한다.
- Target latent와 post-LLM text condition cache를 재사용한다.
- Student용 post-LLM cache와 Synthetic Teacher의 `content+artist` cache는 물리적
  경로와 manifest column을 분리한다. Student cache에 artist token id가 검출되면
  학습을 시작하지 않는다.
- Train/validation artist는 artist 단위로 분리하고, train artist 내부에서도
  reference/target image split을 유지한다. Validation artist의 artist-tag Teacher는
  train에서 사용하지 않는다.
- Reader가 trainable하므로 최종 28개 slot은 본 학습 중 미리 cache하지 않는다.
- 모든 block의 Teacher K/V를 이미지별로 무조건 저장하지 않는다. 저장량이 너무
  크므로 post-LLM condition을 cache하고 필요한 block/probe만 projection한다.
- Teacher를 사용하는 optimizer step에서는 28개 block 모두의 same-Q internal
  Teacher를 한 forward 안에서 계산한다. v1에서는 대표 block subset으로 바꾸지
  않는다.
- BF16 autocast, fused AdamW, pinned/prefetched loader와 GPU-resident 반복 cache를
  기존 production loader와 동일하게 사용한다.
- Style attention은 28 token만 읽으므로 text 512-token attention 대비 attention
  matmul 증가는 작다. 주 trainable parameter는 block별 full-rank K/V다.

## 14. 필수 로깅과 검증

### 14.1 함수 정확성

- `s=0` 및 no-reference가 frozen Anima와 수치적으로 같은지
- Q가 block당 한 번만 계산되는지
- Native O bias가 한 번만 적용되는지
- 신규 Style K/V가 모두 `bias=False`인지
- Text/style softmax가 실제로 분리되는지
- Style token이 512 zero token으로 padding되지 않는지
- Invalid reference가 Set Attention에서 mask되는지
- Teacher와 Student internal residual이 같은 Q를 사용하는지
- Student text cache와 validation prompt에 작가명/token id가 없는지

### 14.2 학습 지표

- Flow loss와 paired flow improvement
- Teacher residual cosine 및 normalized Huber
- Teacher-direction projected ratio `p / ||T||`
- Orthogonal/Teacher RMS ratio
- Block별 style/text residual RMS
- Block 간 residual 상쇄 비율
- Correct-vs-wrong advantage와 ranking accuracy
- Within-artist centered cosine / between-artist centered cosine
- Artist retrieval top-1/margin
- 84-token reconstruction cosine/Huber
- Style attention entropy, top-1 probability, slot별 attention mass/coverage
- Reference-view difference와 common-output ratio
- Prompt mode별 flow/Teacher/artist 지표
- Reference 수 `1/2/4/8`별 validation 지표
- Loss별 weighted contribution과 주요 parameter group gradient norm

Attention-map diversity만으로 성공을 판단하지 않는다. Native O 뒤의 실제
residual과 최종 velocity에서 기능적 다양성을 측정한다.

### 14.3 정성 샘플

- 같은 prompt/seed에서 reference만 바꾸는 controlled validation artist 비교
- 작가별로 서로 다른 held-out caption/seed를 쓰는 natural panel도 별도 생성
- Reference 1/2/4/8장 비교
- 전역 strength `0/0.5/1/2`의 단조성 확인
- Train artist와 validation artist를 분리해 표시
- Fixed TestSample 1--7 비교
- No-style frozen Anima baseline을 항상 같은 시트에 포함
- Reference 원본은 crop하지 않고 padding하여 표시
- Student와 validation 생성 prompt에 작가명이 없는지 sheet metadata에 표시
- 현재 최선의 소형 Style Tokenizer checkpoint를 같은 prompt/reference 조건의
  비교 baseline으로 포함

Style이 강해지면서 내용ㆍ구도가 함께 무너지는지 별도로 평가한다.

### 14.4 실행 주기와 resume

- Validation metric: 매 `250` optimizer step
- 8개 train/validation artist panel: 매 `500` step
- Fixed TestSample/reference sheet와 소형 Style Tokenizer 비교: 매 `1,000` step
- Resume checkpoint: 매 `500` step과 best validation 갱신 시
- Checkpoint에는 trainable model, optimizer, scheduler, global step,
  Python/NumPy/Torch/CUDA RNG와 sampler state를 저장한다. AMP scaler를 실제로
  사용하는 dtype/config인 경우에만 그 상태도 저장한다.
- Frozen Anima와 Resampler weight는 checkpoint에 복제하지 않고 model id, revision,
  config hash만 저장한다. 최종 배포 manifest에는 해당 의존성을 명시한다.

W&B는 `train/flow`, `train/teacher`, `train/artist`, `train/reconstruction`,
`val/functional`, `val/artist`, `model/activation`, `system/perf` namespace만 사용한다.
동일 의미의 legacy metric을 중복 기록하지 않는다.

## 15. 주요 위험과 대응

### Slot 의미 불일치

Canonical query와 slot identity만으로 완전한 의미 정렬이 보장되지는 않는다.
Soft type preference, Q/K에만 넣는 slot identity, same-slot aggregation 뒤 1층
mixer와 slot usage 진단으로 대응한다. Identity를 style value에 직접 더해 공통
출력을 만드는 방식은 사용하지 않는다.

### Detail reconstruction의 content leakage

Reconstruction을 약하게 두고 후반 감소시키며, same-artist cross-content target과
multi-reference에서 style 공통 성분을 선택하도록 한다.

### Artist-tag Teacher의 편향

Artist tag는 순수한 시각 화풍만 표현하지 않는다. Synthetic bootstrap과 internal
alignment에 사용하되 Human domain의 최종 정답으로 간주하지 않는다.

### 신규 full-rank K/V의 표본 효율

약 117M의 신규 K/V를 28개 block마다 독립적으로 학습하므로 dense internal
Teacher 없이 flow만으로는 수렴이 느릴 수 있다. 필요하면 후속 ablation에서
신규 shared base + block별 low-rank delta를 비교하되, 첫 기준 모델은 합의한
block별 full-rank K/V로 만든다.

### 큰 무의미한 출력

Raw magnitude 하한을 금지하고 Teacher 방향 투영 band와 orthogonal penalty를
동시에 사용한다.

### 작은 공통 출력

고정 alpha만 믿지 않고 projection floor, same-artist consistency, cyclic
correct-vs-wrong, centered artist retrieval을 함께 사용한다.

## 16. 첫 구현의 완료 조건

1. No-style에서 frozen Anima 출력이 보존된다.
2. Exact-self와 synthetic Teacher pilot에서 internal residual cosine과 projection
   ratio가 함께 상승한다.
3. 출력 크기 증가가 orthogonal residual 증가로만 나타나지 않는다.
4. Correct reference가 cyclic wrong reference보다 일관되게 우세하다.
5. 같은 prompt/seed에서 validation artist별 시각적 차이가 생기며 하나의 공통
   방향으로 붕괴하지 않는다.
6. Strength `0 -> 0.5 -> 1 -> 2`가 대체로 단조롭게 작용하고 이미지 구조가
   유지된다.
7. 단일 Reference 성능이 multi-reference 학습 때문에 퇴화하지 않는다.
8. Synthetic와 Human held-out validation을 분리해 측정했을 때 paired flow
   improvement의 bootstrap 95% confidence interval 하한이 0보다 높다.
9. Correct-vs-wrong accuracy의 confidence interval 하한이 random `0.5`보다 높고,
   향상이 한두 artist에만 집중되지 않는다.

이 조건을 통과한 뒤에만 Resampler 일부 joint fine-tuning이나 cross-slot mixer
2층, block별 alpha 학습을 후속 실험으로 검토한다.
