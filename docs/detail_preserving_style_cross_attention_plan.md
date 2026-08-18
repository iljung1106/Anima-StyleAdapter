# Detail-preserving typed-slot Style Cross-Attention 계획

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

이 문서는 설계 계획이며 아직 구현 완료를 의미하지 않는다.

## 2. 확정한 설계 원칙

1. Anima의 native Q와 full-rank O는 동결하여 재사용한다.
2. Text와 style은 동일한 Q를 쓰되 서로 다른 K/V와 별도 softmax를 사용한다.
3. Style K/V는 native text K/V 복사본이 아니라 블록별 신규 full-rank
   Xavier 초기화 행렬로 만든다.
4. Anima의 native timestep-conditioned `gate_cross(t)`는 그대로 재사용한다.
5. 블록별 style 강도는 우선 학습하지 않고 고정한다. 자유롭게 0으로 수렴할
   수 있는 gate를 두지 않는다.
6. No-style은 learned null token이 아니라 style branch의 정확한 우회다.
7. 출력 크기 하한은 raw RMS가 아니라 유효한 Teacher 방향으로의 투영에 둔다.
8. Reference 순서는 의미가 없으며, 여러 Reference는 같은 canonical slot끼리
   Set Attention으로 합친다.
9. Spatial slot을 고정 위치에 묶기보다 이미지 안의 선ㆍ채색ㆍ눈ㆍ머리카락ㆍ
   질감 같은 세부 정보를 content-adaptive하게 읽게 한다.

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
                  cross-slot Transformer, 우선 1 block
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

Reference 하나당 `28 x 1024` learned latent slot을 사용한다.

- 모든 slot은 고유한 learned identity를 가진다.
- 출력 slot은 고정 spatial grid가 아니다.
- 각 slot은 필요하면 spatial/global/summary를 모두 읽을 수 있다.
- 초기 soft type preference는 `16 spatial / 8 global / 4 summary`로 둔다.
- 이 preference는 hard mask가 아닌 학습 가능한 attention-logit bias다.
- 매 reader block의 residual 이후 slot identity를 다시 더해 slot permutation과
  공통 출력 수렴을 억제한다.

Reader는 pre-norm Perceiver-style cross-attention 2 block을 기본값으로 한다.
과도한 깊이는 초기 실험에서 사용하지 않는다.

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
- Frozen Resampler가 낸 84개 token만 타입별 cosine + 약한 Huber로 복원한다.
- Reconstruction은 전체 loss의 작은 보조 항으로 시작하고 후반에는 감소시킨다.
- Decoder는 추론 모델에 포함하지 않는다.

이는 detail 보존에 유리하지만 content leakage도 키울 수 있으므로 주 loss로
사용하지 않는다.

## 5. Multi-reference aggregation

Reference별 출력은 `[B, R, 28, 1024]`다. Reference를 token 축으로 평평하게
섞지 않고 각 canonical slot index에 대해 R축 Set Attention을 수행한다.

```text
z_s = SetPool_s({z_1,s, z_2,s, ..., z_R,s})
```

- Reference order embedding은 사용하지 않는다.
- 유효 Reference mask를 항상 적용한다.
- 단일 Reference도 별도 bypass 없이 동일한 경로를 통과한다.
- Slot마다 Reference 신뢰도가 다를 수 있어야 한다.
- Reference 수가 늘어나도 출력 token 수는 항상 28개다.

Set Attention 뒤에는 pre-norm cross-slot Transformer를 우선 1 block만 둔다.
이는 같은 slot 정렬의 작은 오류를 보정하고 spatial/global/summary 정보를 최종
스타일로 조합한다. Mixing 전후에 slot identity를 다시 더한다. 2 block은 1 block
결과에서 정보 혼합 부족이 확인될 때만 비교한다.

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
`1024 -> 2048` full-rank Linear로 새로 만들고 Xavier 초기화한다. Anima의 native
`k_norm`, `v_norm`, attention backend와 head layout을 그대로 적용한다.

MLP 전 residual은 다음과 같다.

```math
x_b <- x_b + gate^cross_b(t) [ O_b(A^text_b) + s alpha_b O_b(A^style_b) ]
```

- `s`: 사용자가 조절하는 전역 style strength
- `alpha_b`: optimizer에 넣지 않는 고정 block별 계수
- `gate_cross(t)`: frozen Anima의 native channel/timestep gate

구현상 native O에 bias가 있으면 text/style에 O를 각각 호출하여 bias를 두 번
더하면 안 된다. `O(A_text + s alpha A_style)`로 한 번 투영하거나 style 쪽에는
native O weight만 bias 없이 적용한다. Q도 한 번만 계산하며 text residual을 더한
후 다시 계산하지 않는다.

28개 신규 full-rank K/V는 약 117M parameter다. Reader와 Set Aggregator를 합친
전체 trainable 규모는 약 140--160M을 예상한다. Anima와 Resampler는 동결한다.

## 7. 고정 block strength와 calibration

28개 block에 임의의 같은 alpha를 적용하지 않는다. Synthetic artist-tag Teacher의
block별 style/text residual RMS profile을 측정한다.

```math
m_b = median_(artist,content,t) ||T_b|| / (||A^text_b|| + eps)
```

이를 이용해 초기 style residual이 block마다 과도하거나 지나치게 작지 않도록
고정 `alpha_b` vector를 정한다. Teacher 효과가 거의 없는 block은 작게 두거나
비활성화할 수 있다. Alpha는 첫 실험 동안 학습하지 않으며 전역 strength `s`만
추론 인자로 노출한다.

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
필요할 때만 3-prediction 방식을 별도 평가한다.

## 9. Frozen Anima internal Teacher

Synthetic 데이터에서는 Anima가 이미 알고 있는 `@artist` 반응을 직접 Teacher로
사용할 수 있다. 가장 중요한 계약은 Teacher와 Student를 같은 Q에서 비교하는
것이다.

Student trajectory의 현재 block hidden에서 Q를 한 번 계산하고, 그 Q에 대해
content-only와 `content + @artist` native text context를 각각 적용한다.

```math
T_b(Q_b) = gate_b(t) O_b[
  Attn(Q_b, K_b^(content+artist), V_b^(content+artist))
  - Attn(Q_b, K_b^content, V_b^content)
]
```

```math
S_b(Q_b) = gate_b(t) O_b Attn(Q_b, K_b^style, V_b^style)
```

Raw text K/V는 회귀하지 않는다. 동일한 attention 결과를 만드는 K/V는 유일하지
않기 때문이다. 다음을 Teacher target으로 사용한다.

- 선택된 block의 attended/output residual 방향과 절대 크기
- 전체 Anima의 최종 velocity residual
- 같은 artist가 다른 content에서 만드는 centered residual의 일관성

Native Teacher는 synthetic Anima image에만 직접 적용한다. Human reference에는
정확한 artist-tag Teacher가 없으므로 일반 flow loss, same-artist consistency와
target-inclusive bootstrap을 사용한다. Human과 synthetic reference를 같은 이미지
분포라고 강제하지 않는다.

Teacher reference와 target은 가능한 한 content와 seed를 교차한다.

```text
reference: artist A, content 1, seed 1
target:    artist A, content 6, seed 4
teacher:   content 6 + @artist A, seed 4
```

같은 content/seed만 반복하면 style 대신 이미지 복사를 학습할 수 있다.

## 10. 작은ㆍ공통ㆍ무의미한 출력 붕괴 방지

Teacher residual을 `T`, Student style residual을 `S`라 한다. Raw output RMS에
하한을 두지 않고 Teacher 방향으로의 투영을 측정한다.

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

초기에는 방향 정렬을 먼저 만들고 projection floor를 천천히 높인다.

- 초기 0--250 step: floor를 거의 사용하지 않고 방향/flow 정렬 중심
- 250--1,000 step: Teacher 크기의 약 `0.1 -> 0.5`로 lower ratio ramp
- 이후 신뢰 가능한 synthetic Teacher: validation에 따라 `0.5 -> 1.0` 검토
- Upper ratio는 lower보다 충분히 넓게 두어 강제 saturation을 피한다.

공통된 nonzero 출력도 실패이므로 동일 `x_t`, text, timestep에서 다음을 추가한다.

- 같은 artist의 서로 다른 reference residual consistency
- 다른 artist에 대한 centered residual contrastive
- Correct Teacher가 cyclic wrong Teacher보다 가까운 양방향 ranking
- Batch 공통 residual을 제거한 뒤 artist retrieval/margin

Wrong reference를 강제로 망가뜨리는 단방향 loss는 쓰지 않는다. Cyclic batch의
모든 artist가 동시에 자신의 Teacher에 맞도록 구성한다.

## 11. 전체 loss 구성

초기 구현의 구조는 다음과 같다. 정확한 수치는 smoke/gradient contribution을
측정한 뒤 확정하며, 한 항의 raw scalar 크기만 보고 맞추지 않는다.

```text
L = L_flow
  + lambda_internal * L_internal_teacher
  + lambda_direction * L_teacher_direction
  + lambda_band * (L_floor + L_upper)
  + lambda_orth * L_orth
  + lambda_rank * L_correct_vs_wrong
  + lambda_consistency * L_same_artist_consistency
  + lambda_reconstruction * L_84_token_reconstruction
```

- `L_flow`: 실제 rectified-flow target에 대한 기본 loss
- `L_internal_teacher`: 같은 Q에서의 block별 native artist residual
- `L_teacher_direction`: final velocity/internal residual cosine 또는 normalized Huber
- `L_floor/L_upper`: Teacher 방향의 절대 작용 크기 band
- `L_orth`: 의미 없는 직교 출력 억제
- `L_correct_vs_wrong`: 공통 artist-independent 출력 억제
- `L_same_artist_consistency`: content가 달라도 유지되는 작가 효과
- `L_84_token_reconstruction`: detail 보존용 약한 training-only 항

Hard token RMS normalization, raw output norm floor와 모든 block의 강제 동일 출력은
사용하지 않는다.

## 12. 첫 학습 curriculum

### Phase A: 신규 K/V와 slot 정렬 bootstrap

- Resampler와 Anima는 동결한다.
- Reader, Set Aggregator, Mixer, 신규 full-rank K/V를 처음부터 optimizer에 넣는다.
- 자유 gate warmup은 두지 않는다.
- Reference 1장 비중을 가장 높게 둔다.
- Synthetic internal Teacher를 초기에는 거의 매 step 사용한다.
- Exact-self와 target-inclusive sample은 초기 방향 형성에만 제한적으로 사용한다.
- Projection floor는 낮은 값에서 ramp한다.

### Phase B: same-artist cross-content와 multi-reference

- Reference와 target content/seed를 교차한다.
- Reference 1--4장으로 확대하되 1장과 2장을 더 자주 뽑는다.
- Same-slot Set Attention을 실제로 학습한다.
- Correct-vs-wrong cyclic ranking과 same-artist consistency를 ramp한다.
- Internal Teacher cadence는 매 step에서 2--4 step마다로 낮춘다.

### Phase C: target-excluded generalization

- Reference 1--8장, target 포함률을 0으로 내린다.
- Human flow와 synthetic Teacher를 함께 사용한다.
- Detail reconstruction은 낮추되 완전히 제거할지는 held-out content leakage를 보고
  결정한다.
- Resampler는 첫 실험 전체에서 동결한다. Joint fine-tuning은 별도 후속 실험이다.

구체적인 step 경계는 첫 2k pilot의 validation과 샘플을 보고 확정한다. 신규
117M K/V를 Human flow만으로 처음부터 정렬하는 실험은 하지 않는다.

## 13. 데이터ㆍ캐시ㆍ효율 계약

- Frozen Resampler의 `84 x 1024` cache를 재사용한다.
- Target latent와 post-LLM text condition cache를 재사용한다.
- Reader가 trainable하므로 최종 28개 slot은 본 학습 중 미리 cache하지 않는다.
- 모든 block의 Teacher K/V를 이미지별로 무조건 저장하지 않는다. 저장량이 너무
  크므로 post-LLM condition을 cache하고 필요한 block/probe만 projection한다.
- Block internal Teacher는 초기에 모든 block 또는 대표 block을 batch화하고,
  이후에는 block subset을 순환 샘플링할 수 있다.
- BF16 autocast, fused AdamW, pinned/prefetched loader와 GPU-resident 반복 cache를
  기존 production loader와 동일하게 사용한다.
- Style attention은 28 token만 읽으므로 text 512-token attention 대비 attention
  matmul 증가는 작다. 주 trainable parameter는 block별 full-rank K/V다.

## 14. 필수 로깅과 검증

### 14.1 함수 정확성

- `s=0` 및 no-reference가 frozen Anima와 수치적으로 같은지
- Q가 block당 한 번만 계산되는지
- Native O bias가 한 번만 적용되는지
- Text/style softmax가 실제로 분리되는지
- Style token이 512 zero token으로 padding되지 않는지
- Invalid reference가 Set Attention에서 mask되는지

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

Attention-map diversity만으로 성공을 판단하지 않는다. Native O 뒤의 실제
residual과 최종 velocity에서 기능적 다양성을 측정한다.

### 14.3 정성 샘플

- 같은 prompt/seed에서 여러 validation artist 비교
- Reference 1/2/4/8장 비교
- 전역 strength `0/0.5/1/2`의 단조성 확인
- Train artist와 validation artist를 분리해 표시
- Fixed TestSample 1--7 비교
- No-style frozen Anima baseline을 항상 같은 시트에 포함
- Reference 원본은 crop하지 않고 padding하여 표시

Style이 강해지면서 내용ㆍ구도가 함께 무너지는지 별도로 평가한다.

## 15. 주요 위험과 대응

### Slot 의미 불일치

Shared query와 slot identity만으로 완전한 의미 정렬이 보장되지는 않는다.
Soft type preference, slot identity 재주입, same-slot aggregation 뒤 1층 mixer와
slot usage 진단으로 대응한다.

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

이 조건을 통과한 뒤에만 Resampler 일부 joint fine-tuning이나 cross-slot mixer
2층, block별 alpha 학습을 후속 실험으로 검토한다.
