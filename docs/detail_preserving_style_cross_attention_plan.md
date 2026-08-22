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

### 7.2 Shared-base 초기화 계약 수정(v7)

Shared-base 구현은 메모리와 계산량을 줄이기 위한 구조일 뿐, native text K/V를
재사용하기 위한 구조가 아니다. v6 구현은 medoid block 3/12/18/26의 native text
K/V weight를 네 base에 복사하여 본 문서 2절의 확정 원칙을 위반했다.

Style Reader 출력과 Anima post-LLM text context는 차원만 `1024`로 같을 뿐 의미적
basis와 분포가 다르다. `k_norm`과 `v_norm`은 크기를 안정화하지만 두 표현 공간을
정렬하지 않는다. 따라서 v7부터는 다음 계약을 사용한다.

- 네 shared full-rank Style K/V base는 서로 독립적으로 Xavier uniform 초기화한다.
- medoid와 block cluster는 base 공유 범위를 정하는 데만 사용한다.
- 블록별 rank-64 K/V delta는 down Xavier, up zero 초기화를 유지한다.
- Anima의 native Q, full-rank O, K/V norm 및 attention backend만 재사용한다.
- 초기 `alpha_b(t)` calibration은 새 Xavier branch의 출력 크기를 native effect
  규모에 맞추는 초기화일 뿐, K/V 방향이나 학습 중 RMS를 고정하지 않는다.

v6 체크포인트는 native-text-K/V 초기화 ablation으로만 보존하며 v7의 초기값으로
재사용하지 않는다.

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

## 17. v17 방향 우선 재정렬 실험

v16의 2,500-step 검증에서는 artist-centered 출력 RMS가 native teacher와 거의
같아졌지만 held-out flow direction cosine은 약 `0.035`, 선택 block의 post-gate
cosine 중앙값은 약 `0.008`에 머물렀다. 따라서 v17은 출력량을 더 강제하지 않고
**올바른 작가 방향을 먼저 학습한 뒤 크기를 여는 것**을 목표로 한다. Shared
K/V base + block별 low-rank delta 구조와 frozen Anima/Resampler는 이번 비교에서
바꾸지 않는다.

### 17.1 Post-gate 방향과 크기 분리

선택 block의 `gate_cross x O(style attention)`과 native artist-tag teacher를
artist-centered한 뒤 다음 두 항을 별도로 계산한다.

```math
L_{pg-dir}=1-cos(center(S_b), center(T_b))
```

```math
c_b = <S_b,T_b> / ||T_b||^2
```

`L_pg-dir`은 정규화된 출력에서 계산하여 작은 calibrated alpha가 방향 gradient를
약화하지 않게 한다. Post-gate magnitude band는 방향 bootstrap 단계에서는 끄고,
성능 기준을 통과한 다음에만 `c_b`가 설정 범위에 들어오도록 활성화한다. 기존의
고차원 post-gate Huber 회귀는 기본값에서 끈다.

### 17.2 Final residual 목표 단순화

Native centered final-velocity residual의 full-resolution Huber weight는 `0.20`에서
`0.05`로 1/4 축소한다. 대신 latent spatial grid를 average-pooling한 `2x2`와
`4x4` 저주파 residual을 각각 teacher RMS로 정규화해 평균한 보조 목표를 둔다.
세부 노이즈를 복제하는 대신 반복 가능한 저주파 작가 효과를 보존하는 것이 목적이다.

### 17.3 Teacher-direction cyclic ranking

일반 flow MSE 기반 correct-vs-wrong ranking을 주 ranking으로 사용하지 않는다.
Controlled teacher batch의 동일한 `x_t`, timestep, content, Student Q에서 student
artist effect와 native teacher effect를 center하고 cyclic-shifted teacher를 hard
negative로 사용한다.

```math
L_{rank}=relu(m - cos(S_i,T_i) + cos(S_i,T_{i+1}))
```

Wrong 경로를 일부러 망가뜨리는 별도 목표 없이 correct 방향만 움직인다. 기존
flow ranking은 v17 기본 설정에서 비활성화한다.

### 17.4 결합 common-output / artist-energy 제약

Controlled batch의 student effect를 `S_i=C+D_i`, `C=mean_i(S_i)`로 분해한다.
Frozen native teacher scale만 분모로 사용하여 공통 성분 상한과 artist-centered
성분 하한을 한 번에 계산한다.

```math
L_{joint}=relu(||C||/s_T-r_C)^2
          + w_D relu(r_D-RMS_i(D_i)/s_T)^2
```

Common 성분만 처벌해 zero-output으로 도망가지 못하도록 두 항은 항상 함께
활성화한다. Teacher controlled batch와 일반 human main batch의 same-Q probe에
같은 정의를 사용한다.

### 17.5 성능 기반 curriculum

Reference 수, target 포함률과 teacher 주기는 절대 step만으로 전환하지 않는다.
Loader는 최대 8개 held-out reference를 효율적으로 prefetch하고, 현재 stage가
허용한 reference 수만 mask한다. 검증 때 다음 지표를 평가해 연속 두 번 통과했을
때만 다음 stage로 전환하며 stage와 연속 통과 횟수는 training state에 저장한다.

- 선택 block post-gate cosine 중앙값
- native final projection coefficient
- heldout correct-vs-wrong advantage
- validation common-output ratio
- artist-centered student/native RMS 비율
- 후속 stage에서는 heldout paired improvement

Stage 0은 exact-self 1-reference와 매-step controlled teacher를 유지하고 post-gate
magnitude를 끈다. 첫 기준을 통과하면 1/2/4-reference와 target probability `0.65`,
post-gate magnitude를 연다. 더 엄격한 heldout 기준을 통과한 뒤에만 1/2/4/8
reference, target probability `0`, teacher 매 2-step 단계로 넘어간다. 기준을
통과하지 못하면 학습 step이 늘어도 자동으로 난도를 높이지 않는다.

### 17.6 v17 선택 기준

`val/functional/panel`의 서로 다른 prompt/seed 다양성은 성공 근거로 사용하지
않는다. 같은 prompt/seed의 fixed-reference에서 common effect와 artist-centered
effect를 분리하고, direction/ranking 검증과 함께 판단한다. Global strength 증가는
방향 정렬을 통과한 뒤의 inference 조절로만 사용한다.

## 18. v18 16-artist centered objective

v17 1,500-step gradient 진단에서 controlled Teacher gradient가 main flow보다
`8.6--8.8x` 컸고, projection band와 orthogonal cap의 cosine은 `-0.55`였다.
또한 cyclic/common hinge는 쉬운 batch-4에서 gradient가 정확히 0이 되었고,
block별 post-gate 목표는 공통 Reader에서 충돌했다. v18은 다음처럼 단순화한다.

- 동일 `x_t`, timestep, content, Student Q를 공유하는 16명 controlled batch를
  4명 microbatch로 처리한다. Student residual은 final artist mean으로 center한다.
- Final centered residual에 `1-cos(S_i,T_i)`를 직접 적용하고, 양의 native
  projection에는 약한 magnitude band만 둔다. 기존 projection/artist-energy/
  orthogonal 세 항은 제거한다.
- Cyclic negative 하나 대신 16명 native teacher 전체를 negative로 쓰는 InfoNCE를
  적용한다. Frozen teacher만 bank로 사용하므로 다른 student graph를 보존하지 않는다.
- Common-output은 16명 전체 student mean과 frozen native scale로 계산한다. Hard
  hinge 대신 softplus를 사용하고, 첫 no-grad pass로 얻은 mean의 정확한 gradient를
  두 번째 microbatch pass에 선형 surrogate로 전달한다.
- Residual 회귀는 `2x` low-frequency를 주 보조 항으로 쓰고 full-resolution은
  약하게 유지하며 `4x` 항은 제거한다.
- Controlled 증류 합계에는 단일 `teacher_global_weight=0.1`을 적용한다. Post-gate
  증류는 adapter K/V, block delta, base mixing에만 역전파하고 Reader에는 전달하지
  않는다.
- Validation의 `self` 경로는 이름만 유지하되 실제 active performance stage와 같은
  reference 수와 target 포함률을 사용한다. 고정 exact-self는
  `exact_self_probe`로 별도 기록한다.
- Curriculum 전환은 post-gate cosine을 사용하지 않는다. Final centered cosine,
  positive native projection, 16-artist common-output ratio가 연속 3회 기준을 통과할
  때만 다음 stage로 이동한다. Post-gate cosine은 진단용으로만 계속 기록한다.

Main flow와 reconstruction은 기존대로 유지한다. 기존 main-batch artist-effect,
prototype, common/magnitude 보조 항은 v18에서 꺼 중복 gradient를 제거한다.

## 19. v19 Reader 정보보존 선행학습

v18은 1,500스텝까지 final centered cosine과 known-artist InfoNCE가 상승했지만,
같은 prompt/seed의 미지 fixed reference 출력은 오히려 서로 비슷해졌다. 실제
gradient 진단에서는 controlled teacher 합계가 main flow보다 약 `2.6x` 컸으며,
특히 InfoNCE가 Reader의 가장 큰 gradient였다. 반면 무작위 초기화된 86M Reader의
reconstruction gradient는 그보다 훨씬 작았다. 또한 기존 decoder는 각 reference의
`84 -> 28 -> 84`만 지도하고 reference set-attention과 cross-slot mixer는 지도하지
않았다.

따라서 v19은 Anima를 로드하기 전에 Reader만 다음 두 목표로 4,000스텝
선행학습한다.

- 각 reference의 정규화된 84개 Dual-query token을 해당 28개 canonical token에서
  복원한다.
- 같은 작가의 네 reference를 set-attention/mixer로 합친 최종 28개 token에서,
  네 입력의 평균 84-token을 복원한다. 이 항이 multi-reference 경로까지 직접
  학습한다.

본학습은 새 adapter와 함께 처음부터 시작하되 선행학습 Reader를 불러온다. Reader
LR은 `2e-5`, reconstruction weight는 전 구간 `0.05`로 유지해 정보보존을 잊지
않게 한다. Known-artist InfoNCE는 `0.05`로 낮추고, 관측된 공통 출력 붕괴를 직접
억제하는 16-artist common-output 항은 `0.15`로 높인다. 나머지 v18 final-centered
teacher, 저주파 residual, 성능 기반 curriculum은 유지한다. 선택 기준은 fixed
reference 간 차이, pooled reconstruction validation, final centered cosine과
common-output ratio를 함께 사용한다.

## 20. v19 2k performance-gate correction

Reader 선행학습을 적용한 본학습은 2,000스텝에서 exact-self paired
improvement `+0.00655`, final centered cosine `0.334`, 16-artist InfoNCE
accuracy `0.602`, common-output ratio `0.541`을 달성했다. 반면 native-axis
projection coefficient는 magnitude floor ramp가 완료된 후에도 `0.227`에서
정체했다. 출력의 centered RMS는 native의 `0.927`배이므로 이는 출력
에너지 부족이 아니라, lossy reference에서 고차원 native residual 축을
완전히 회귀하도록 요구한 기준이 과도한 문제였다.

따라서 exact-self stage의 목표는 절대 native effect의 40% 회귀가 아니라,
학습 가능한 방향성과 작가 분리가 안정적으로 확립되었는지로 한다.

- Stage 0은 최소 2,000스텝 이후 final cosine `>=0.30`, native projection
  `>=0.20`, common-output ratio `<=0.62`를 2회 연속 통과하면 종료한다.
- Stage 1은 target-included 1/2/4-reference를 충분히 보도록 최소
  4,000스텝까지 유지한다. 그 뒤 final cosine `>=0.30`, projection
  `>=0.20`, common-output `<=0.65`를 3회 연속 통과하면 완전히
  target-excluded 1/2/4/8-reference stage로 이동한다.
- Magnitude/direction/InfoNCE 가중치는 변경하지 않는다. 1,500→2,000
  구간에서 final cosine과 InfoNCE가 계속 상승했고, 샘플 효과 RMS도
  축소되지 않았기 때문이다.

2,000스텝 checkpoint와 optimizer/RNG state를 그대로 resume하며, 전환
후에는 heldout paired improvement, correct-vs-wrong advantage, fixed-reference
작가 간 차이를 주요 선택 기준으로 본다.

## 21. v20 전체 작가 dual-domain teacher

v19은 4,500스텝에서 완전 target-excluded stage로 정상 전환되고 5,000스텝
heldout paired improvement도 양수였지만, 같은 prompt/seed의 일곱 fixed
reference가 거의 같은 기본 Anima 화풍을 만들었다. Teacher InfoNCE가 70%를
넘어도 실제 화풍이 분리되지 않았으므로 내부 retrieval 수치만으로 성공을
판정할 수 없다. 직접 원인은 controlled native teacher가 synthetic cache와
교차하는 1,800명만 보았고, 나머지 human 작가는 noisy main flow만 받았다는
것이다.

v20은 모델 구조와 Reader 선행학습은 유지하고 지도 범위를 바로잡는다.

- Human frozen-Resampler cache는 native bank의 train 작가 4,000명 전체를
  controlled teacher 대상으로 사용한다.
- 기존 Anima synthetic cache 1,800명은 별도 domain으로 유지한다. 한 reference
  set에서 두 domain을 섞지 않고 human:synth teacher batch를 `3:1`로 교대한다.
- 완전 heldout stage에서도 controlled teacher를 매 스텝 유지한다.
- Final centered all-wrong InfoNCE는 `0.05 -> 0.10`, 16-artist common-output은
  `0.15 -> 0.20`으로 올리고 common threshold는 `0.45`로 낮춘다.
- 일반 human main batch에도 네 작가가 같은 `x_t`, timestep, prompt를 쓰는
  common-output + artist-energy 항을 4스텝마다 `0.02` 가중치로 적용한다.
- 출력 디렉터리와 W&B run을 v19와 분리하고 adapter는 다시 무작위 초기화한다.
  Reader만 검증된 4k reconstruction checkpoint에서 시작한다.

성공 기준은 heldout flow 수치가 아니라 fixed-reference 1x 시트에서 서로 다른
화풍이 육안으로 확인되고, pixel common component와 final controlled common-output이
함께 감소하는 것이다. 이 기준을 만족하지 못하면 shared-K/V 용량보다 먼저
teacher-to-human 전달 경로와 final functional objective를 다시 진단한다.

## 22. v21 Synthetic-only teacher와 main-flow 출력 하한

v20의 Human reference를 native `@artist` residual에 직접 정렬하는 가정은
폐기한다. Anima의 작가 태그 효과는 약하고 실제 Human reference의 화풍과
일치한다는 보장이 없으므로, 작가 태그 증류는 그 태그로 직접 생성한 Synthetic
Anima reference에만 적용한다. v20의 Human:Synthetic `3:1` teacher domain,
Human 기준 alpha calibration, main Human native-scale common auxiliary 및 강화된
teacher weight는 모두 v19 값으로 복원한다.

v21은 v19의 pretrained Reader를 재사용해 초기 정보 병목을 통제하되 이를 새
스타일 지도 신호로 간주하지 않는다. 지속 reconstruction weight는 `0.05`에서
`0.01`로 낮춰 forgetting 방지만 맡긴다. 이 초기값을 제거하면 출력 크기/LR
실험과 무작위 86M Reader 재학습이 혼재하므로 이번 비교에서는 유지한다.

핵심 실험 변수는 Human main flow의 약한 출력 shortcut이다. 같은 noisy latent,
timestep, prompt에서 Adapter 출력과 frozen Anima 출력을 비교하고,
`target_velocity - base_velocity` 방향으로 투영된 Adapter 최종 velocity 변화에만
하한을 둔다. 전체 RMS만 키우는 것으로는 만족할 수 없으며, unrelated orthogonal
성분은 main flow MSE에 그대로 불리하다. 투영 하한은 1~500스텝에
`0.05 -> 0.20`으로 올리고 네 스텝마다 `0.50` 가중치로 적용한다. 과대 출력을
막기 위해 desired residual RMS의 `0.75`를 약한 상한으로 둔다.

- Exact-self stage: projection scale `1.0`
- Target-mixed stage: projection scale `0.5`
- Fully heldout stage: projection scale `0.25`

학습률은 v19 대비 두 배로 올린다.

- Reader `2e-5 -> 4e-5`
- Shared K/V `5e-5 -> 1e-4`
- Block delta `1e-4 -> 2e-4`
- Base mixing `2e-5 -> 4e-5`

Reader, adapter, optimizer는 모두 새 v21 run에서 처음부터 시작하되 Reader 가중치만
기존 reconstruction checkpoint로 초기화한다. v19 checkpoint는 재사용하지 않는다.
선택 기준은 projection loss 자체가 아니라 exact-self/heldout paired improvement,
final style delta 크기, common-output, fixed-reference 1x 시각적 작가 차이와 이미지
안정성을 함께 사용한다.

## 23. v22 전체 final-delta RMS 강제 bootstrap

v21 500~720스텝에서 final style delta는 desired residual RMS의 약 `0.12`에
머물렀고, 실제 desired 방향 projection coefficient는 약 `0.006`이었다. 기존
projection floor는 네 스텝마다만 적용되어 스텝당 실효 기여가 main flow의 약
4%였고, 모델은 하한을 만족시키지 않은 채 작은 출력 패널티를 감수했다.

v22는 방향이 맞는 성분에만 하한을 주던 main projection objective를 끈다. 대신
모든 main-flow 스텝과 모든 performance stage에서 다음 실제 출력 비율을 직접
`1.0`으로 맞춘다.

`RMS(v_style - v_frozen) / RMS(v_target - v_frozen)`

이 loss는 방향을 보지 않으므로 직교 residual도 크기가 같으면 만족한다. 가중치
`0.50`, Smooth-L1 beta `0.10`을 첫 스텝부터 적용해 전체 크기를 먼저 확보한다.
별도의 normalized-direction loss는 추가하지 않는다. 기존 raw flow MSE가 충분히
큰 style delta를 target residual 쪽으로 회전시키도록 하여, magnitude 강제와 방향
학습의 역할을 명확히 분리한다. Synthetic-only native artist teacher, Reader 초기값,
LR 및 나머지 v21 구성은 유지하고 adapter와 optimizer는 새로 시작한다.

## 24. v23 non-reconstructive repeatable artist effect

v22는 500스텝에서 self/heldout 모두 style delta RMS를 desired residual의 약
`0.96`까지 키웠지만 방향 cosine은 `0.02` 안팎, orthogonal ratio는 약 `0.96`,
paired improvement는 약 `-0.89`였다. 전체 desired residual에는 작가 화풍 외에
개별 content, noise/timestep 성분과 Frozen Anima 오차가 포함되므로, lossy style
reference가 알 수 없는 성분까지 큰 출력으로 만들게 한 것이 실패 원인이다.

v23은 DEADiff의 same-style/different-content 학습과 contrastive style learning을
따라 다음처럼 바꾼다.

- Full final-delta RMS 강제 loss를 완전히 제거하고 표준 rectified-flow MSE를 주
  목표로 복원한다.
- 각 작가에서 target을 포함하지 않는 heldout reference를 두 개의 disjoint view로
  나눈다. 두 view는 동일한 `x_t`, timestep, prompt, seed와 Student Q에서 평가한다.
- 첫 view의 final Anima effect는 detach하고 두 번째 view가 그 artist-centered
  low/mid-frequency effect를 재현하게 한다. Batch artist mean은 비교 전에 제거한다.
- Batch의 다른 모든 작가를 negative로 쓰는 symmetric InfoNCE와 동일 작가
  repeatability를 함께 사용한다. 단일 cyclic wrong만 쓰지 않는다.
- Repeatable teacher 방향 projection에는 약한 magnitude band만 둔다. Full desired
  residual 크기나 직교 성분에는 하한을 주지 않는다.
- Frozen Resampler 정보 보존 reconstruction은 `0.03`, token prototype은 `0.03`의
  약한 보조 항으로 유지한다. Functional effect는 `0.15`, common-output은 `0.02`,
  repeatable magnitude band는 `0.05`를 사용한다.

Raw flow tensor의 cross-reference 일치는 같은 controlled condition 안에서만
정의한다. 서로 다른 content/noise/timestep 사이에서는 raw tensor를 같게 만들지
않고 pooled functional effect, retrieval, ICC와 생성 샘플로 일반화를 평가한다.
주요 선택 지표는 heldout repeatable ratio/ICC, all-artist retrieval, common-output,
fixed-reference 시각적 작가 차이와 이미지 안정성이다. Paired flow improvement는
보조 지표로만 사용한다.

## 25. v24 teacher-free exact-self flow baseline

v23 500스텝에서 synthetic teacher와 repeatable-artist 내부 지표는 개선됐지만,
heldout flow와 실제 생성 이미지의 스타일 효과는 개선되지 않았다. 목표 간 충돌과
약한 teacher 효과를 구조 자체의 학습 가능성과 분리하기 위해 1,000스텝의 최소
baseline을 새로 시작한다.

- Frozen Anima만 재사용하고 Reader, shared K/V, block delta, base mixer는 모두
  무작위 초기화한다. 이전 Reader·adapter 체크포인트는 사용하지 않는다.
- 모든 학습 행은 target의 cached 84-token 표현 한 장만 reference로 사용한다.
- 최적화 목적은 표준 rectified-flow MSE 하나뿐이다. Teacher bank와 native
  timestep weighting, reconstruction, prototype, contrastive/ranking, magnitude,
  common-output 및 post-gate 증류는 계산하지 않는다.
- Teacher calibration도 사용하지 않는다. 블록별 alpha는 timestep과 무관한
  `0.10`, global gain은 `1.0`으로 고정한다.
- Prompt mode는 Full 30%, Tag Dropout 40%, Short 20%, Empty 10%다. Empty가 아닌
  행은 50% 확률로 quality prefix를 사용하고, Empty는 그대로 빈 prompt를 쓴다.
- 250스텝마다 train/validation exact-self panel을, 500스텝마다 고정된 validation
  이미지 일곱 장의 exact-self panel을 생성한다. 모든 시트의 reference는 표시된
  target 자체이며 heldout reference를 섞지 않는다.

이 실험은 스타일 일반화를 평가하지 않는다. Exact-self에서도 이미지를 따라가는
효과가 생기지 않으면 teacher나 보조 loss가 아니라 Reader→K/V→Anima 주입 경로와
고정 alpha가 병목이다. Exact-self가 성공하면 이후에만 같은 작가의 다른 reference로
일반화하는 목적을 하나씩 추가한다.

## 26. v25 최소 보조목표와 단계적 self 제거

v24는 1,000스텝 동안 최종 delta를 줄여 flow 손상을 완화했지만, validation
exact-self의 paired improvement는 `-0.0617`, 방향 cosine은 `0.0300`에 머물렀다.
따라서 복잡한 과거 규제를 다시 켜지 않고 다음 네 보조 신호만 추가한다.

- 1~100스텝에는 synthetic Anima reference에 한정해 28개 전 블록의
  `gate_cross × O` 출력을 native artist-tag 출력 방향에 맞춘다. 블록 loss는 합이
  아니라 평균이며, Reader에는 이 local teacher gradient를 주지 않는다.
- 최종 Anima velocity residual의 크기 하한은 동일 `x_t`, timestep, prompt와 Q를
  공유하는 controlled artist batch에서만 계산한다. Artist batch mean을 제거한 뒤
  frozen native median scale의 `0.25 → 0.50` 범위로 하한을 올리므로 공통 출력은
  이 loss를 만족할 수 없다. 상한은 `1.25`다.
- Reader reconstruction은 전 구간 `0.03`으로 둔다.
- 251스텝부터 같은 작가의 서로 겹치지 않는 두 heldout reference view를 같은
  controlled probe에서 평가하고, batch의 모든 다른 작가를 negative로 쓰는
  functional InfoNCE를 `0.05`까지 ramp한다. Token prototype은 사용하지 않는다.

Reference curriculum은 `1~250: target 한 장만`, `251~1000: 1~4장 중 target 필수`,
`1001~2000: target 포함률 1→0 선형 감소, 1~8장`이다. Empty prompt는 과거 계약대로
항상 target 한 장만 reference로 사용한다. 표준 rectified-flow MSE가 주 loss이며,
prototype, cyclic ranking, 별도 common-output, raw desired-residual magnitude loss는
모두 끈다. Alpha는 v24와 동일하게 모든 블록 `0.10`, global gain `1.0`으로 시작해
loss 구성과 reference curriculum만 비교한다.

## 27. v36 reference-fidelity continuation

v35에서는 750→1,000스텝 사이 target 포함률이 `0.67→0.50`으로 내려가면서
style-output ratio, artist retrieval, ICC와 cross-reference repeatability가 함께
감소했다. 같은 작가의 서로 다른 작품이 반드시 같은 화풍이라는 가정도 실제
데이터와 맞지 않으므로, v36은 v35 750스텝의 모델·Adam 상태에서 다음처럼 잇는다.

- 같은 작가의 disjoint-reference functional objective 총 가중치는 `0.10→0.05`,
  repeatability 내부 가중치는 `0.75→0.10`으로 낮춘다.
- Repeatability는 계속 1로 끌어올리지 않는다. Centered low/mid-frequency effect의
  agreement가 `0.30`보다 낮을 때만 hinge penalty를 주고, 그 이상에서는 서로 다른
  작품의 reference-specific 성분을 보존한다.
- 750스텝의 target 포함률을 `0.80`으로 다시 올린 뒤 8,000스텝까지 천천히 0으로
  낮춘다. 1,500스텝까지 reference 수는 1장 70%, 2장 25%, 4장 5%로 구성한다.
- Prompt는 Full 20%, Tag Dropout 40%, Short 25%, Empty 15%이며 Empty는 항상
  exact target 한 장을 사용한다. Reader reconstruction은 `0.01→0.005`의 약한
  보존 항으로 유지한다.
- Synthetic native-artist teacher는 8스텝마다 유지하되 Reader gradient만 0.1배로
  줄인다. Teacher는 Anima-facing K/V를 정렬하고, Reader는 human flow를 통해
  reference별 content·style 세부 정보를 보존하게 한다.
- LR decay는 6,000스텝부터 시작한다. Common-output/artist-energy 제약은 유지하지만
  disjoint-view magnitude weight는 `0.03→0.01`로 낮춘다.

우선 평가는 paired flow 하나가 아니라 고정 reference 시각 결과, style-output
ratio, artist retrieval/ICC, common-output ratio, 같은 작가에서 reference를 바꿨을
때의 출력 다양성을 함께 사용한다.
