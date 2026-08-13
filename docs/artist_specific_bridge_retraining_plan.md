# Artist-Specific Resampler 및 Anima Bridge 재학습 계획

## 1. 목적과 현재 진단

이 문서는 다음 학습 사이클에서 수행할 Resampler 재학습, frozen-token cache 갱신,
artist-specific Bridge/Connector 정렬, online Frozen-Anima 학습의 실행 순서와 통과 조건을
정의한다.

현재 시스템은 Style Adapter의 출력 크기를 키우는 데에는 성공했지만 reference artist에 맞는
방향을 안정적으로 만들지 못했다. 기존 offline K/V/O bootstrap은 validation zero-output
improvement 약 `0.59`를 기록했으나 correct/wrong-reference gap은 거의 `0`이었다. 이는 큰
connector가 reference를 읽기보다 여러 작가 태그에 공통적인 평균 effect를 출력하는 shortcut을
선택했음을 뜻한다.

현재 online exact-self 실행의 750-step 고정 validation도 다음과 같았다.

- held-out reference paired-flow improvement: `-2.69%`
- exact-self reference paired-flow improvement: `-2.47%`
- style output/base velocity RMS ratio: 약 `4.7%`

따라서 현재 문제는 style 출력이 너무 작은 것이 아니다. 시각적으로 보이는 변화는 만들지만 그
방향이 target flow를 평균적으로 악화한다. 해결 순서는 표현의 artist 정보, native Anima effect
정렬, online flow 최적화를 분리하여 검증하는 방식으로 바꾼다.

## 2. 고정 모델 경계

기본 입력과 모델 규모는 우선 유지한다.

- Vision features: C-RADIO L18 spatial + L24 spatial + L24 native SigLIP CLS
- Per-reference Resampler: 내부 폭 `1536`, 3 blocks, 출력 `128 x 1024`, 약 120M
- Minimal Set Aggregator: single-reference 단계에서는 완전 우회
- Anima: 28 blocks, frozen backbone
- Style connector: pretrained Anima K/V/O base + block별 rank-128 delta
- Reference-specific path: query-conditioned centered artist head

단순한 모델 증설이나 총 step 연장은 평균 artist-effect shortcut을 더 잘 맞추게 할 수 있으므로,
아래 단계의 통과 조건이 실패했을 때 첫 해결책으로 사용하지 않는다.

## 3. Stage R — Per-reference Resampler 재학습

### 3.1 목표

Resampler가 reconstruction에 필요한 위치·질감 정보를 유지하면서, 같은 작가의 다른 content에서
공통적인 artist 정보를 현재보다 명확히 남기게 한다. 목표는 작가 분류 정확도의 단독 최대화가
아니라 reconstruction, unseen-artist retrieval, downstream reference-effect discrimination의
Pareto 개선이다.

### 3.2 Loss

```text
L_R = L_reconstruction
    + lambda_joint * L_joint_prototype
    + lambda_slot  * L_slot_prototype
    + lambda_div   * L_slot_diversity
```

초기 탐색 범위는 다음과 같다.

- reconstruction: `1.0`
- joint prototype: `0.06`, `0.09`, `0.12`
- slot prototype: `0.001~0.003`
- slot diversity: 기존 `0.0075` 기준 유지
- prototype ramp: 전체 학습의 초기 `20~30%`

Joint prototype은 전체 128개 token을 사용하는 attention descriptor 또는 다음 mean/std
descriptor에 적용한다.

```text
descriptor(z) = normalize(concat(mean(LN(z)), std(LN(z))))
```

Slot prototype은 매우 약하게 유지한다. 이를 크게 만들면 128개 slot이 모두 같은 artist ID를
중복 저장하고 image-specific 정보가 사라질 수 있다.

### 3.3 학습량과 평가

최대 `10k~15k steps`를 허용하되 validation sweet spot의 checkpoint를 선택한다.

- 250 steps마다: 고정 episodic validation loss
- 500 steps마다: 1/2/4/8-reference unseen-artist Top-1/MRR와 L18/L24 reconstruction
- 1,000 steps마다: frozen mini reference-effect probe의 correct/wrong gap
- 500 steps마다 checkpoint 저장

선택 지표는 다음 세 항목의 Pareto 성능이다.

1. unseen-artist retrieval
2. L18/L24 reconstruction
3. frozen mini connector에서의 correct/wrong reference gap

Retrieval이 높아도 reconstruction이나 downstream effect gap이 악화되면 선택하지 않는다.

## 4. Stage C — Frozen Resampler token cache 재생성

선택한 Resampler checkpoint를 동결하고 cache를 다시 만든다. 이전 Resampler의 token cache와
혼용하지 않는다.

### 4.1 우선 생성

- synthetic teacher의 유효 이미지 약 7,872장
- shape: `128 x 1024`
- dtype: BF16
- 예상 용량: 약 2.1GB

이 cache로 Bridge 단계가 통과한 뒤에만 실제 Danbooru 약 150k장의 전체 cache를 만든다.
전체 cache의 예상 용량은 약 36.6GiB다.

### 4.2 Manifest 계약

각 cache manifest에 다음을 기록한다.

- Resampler checkpoint 경로와 SHA256
- Resampler architecture/version
- C-RADIO source cache version
- image ID와 artist/style ID
- source split
- token shape와 dtype

학습 시작 시 manifest의 checkpoint hash가 설정과 다르면 즉시 실패시킨다.

## 5. Stage A0 — Reference-specific 경로 선행 정렬

### 5.1 학습 경계

동결한다.

- C-RADIO와 Resampler
- Minimal Set Aggregator
- 대형 shared/group connector
- block별 K/V/O low-rank delta
- Anima backbone

우선 학습한다.

- Query-conditioned Centered Artist-Specific Head와 그 내부 full-rank `style_kv` Bridge

공통효과 connector의 `style_context_proj`는 A0에서 동결한다. Centered Head 내부의
`style_kv: 1024 -> 1024`가 reference-specific Anima 좌표 정렬을 담당한다. 작은 `1e-4`
공통 Bridge를 Head의 LayerNorm 앞에서 함께 학습하면 scale은 제거되면서 gradient만 증폭되는
퇴화가 발생하므로, 공통 Bridge는 A1에서 full effect와 함께 연다.

이 단계에서는 full/common artist effect를 목표로 삼지 않는다. Reference-independent 평균 effect로
loss를 줄이는 경로를 구조적으로 차단한다.

### 5.2 동일 조건의 centered target

작가별 native teacher effect를 다음과 같이 정의한다.

```text
Delta O_a = Attn(Q, K(content + @artist_a), V(content + @artist_a)) W_O
          - Attn(Q, K(content),             V(content))             W_O

E_a = Delta O_a - mean_artist(Delta O)
```

Center를 계산하는 artist batch는 반드시 다음 조건을 공유한다.

- 동일 content prompt
- 동일 timestep
- 동일 Anima block
- 동일 Q와 query 위치
- 가능하면 동일 noisy-latent trajectory 조건

그래야 centered target에 content, timestep, query 차이가 섞이지 않는다.

### 5.3 Hard negative와 all-pairs 학습

하나의 batch는 동일 content에 대한 여러 작가로 구성한다. 예를 들어 32작가 batch에서 각 target
Q에 32개 candidate reference를 모두 적용하여 `32 x 32` all-pairs 결과를 만든다. 대각선만
정답이다.

주 hard negative는 동일 content 조건의 다른 작가다. Content와 Q가 고정되므로 reference의
content shortcut으로 matching 문제를 풀 수 없다. 필요하면 style descriptor상 가까운 작가를
추가 hard negative로 사용한다.

### 5.4 Loss

```text
L_A0 = 1.0 * normalized centered-residual Huber
     + lambda_dir   * centered direction/cosine
     + lambda_match * all-pairs artist-effect InfoNCE
     + lambda_rms   * log-RMS magnitude
```

Raw K/V token MSE는 사용하지 않는다. Teacher의 512 text token과 student의 128 style slot은
token-wise 대응하지 않으므로 실제 Q에 대한 attention output을 직접 지도한다.

### 5.5 통과 조건

다음 조건을 모두 충족해야 Stage A1로 이동한다.

- validation centered residual cosine이 안정적으로 양수
- correct/wrong cosine gap이 양수
- correct/wrong improvement gap이 양수
- held-out content에서 위 gap 유지
- unseen validation artist에서 위 gap 유지
- 특정 소수 block만 성능을 담당하지 않음

평균 full-effect improvement는 이 단계의 선택 기준이 아니다. Meta-test 작가는 checkpoint 선택에
쓰지 않고 validation-best 확정 후 한 번만 평가한다.

## 6. Stage A1 — Full artist effect 추가

Stage A0 best에서 시작한다.

1. Centered Head는 먼저 동결하거나 매우 낮은 LR로 유지한다.
2. Shared connector를 열고 full native artist effect를 학습한다.
3. centered residual과 all-pairs discrimination loss는 계속 유지한다.
4. 그 뒤 group connector를 연다.
5. block별 K/V delta를 연다.
6. 필요할 때만 O delta를 마지막으로 연다.

효과를 다음처럼 역할 분리하여 학습한다.

```text
Delta O_artist = common artist-like effect + reference-specific centered effect
```

대형 connector가 common effect만 출력하는 shortcut으로 돌아가지 못하도록 centered loss와
correct/wrong gap을 checkpoint 선택에 계속 포함한다. O delta는 출력 방향을 직접 회전시킬 수
있으므로 가장 늦게 개방한다.

Centered Head는 우선 최종 경로에 유지한다. 전체 단계가 통과한 뒤 성능 손실 없이 단일 connector로
증류할 수 있을 때만 제거를 검토한다.

## 7. Stage O — Online native residual distillation

Stage A1이 통과한 뒤 Frozen Anima 전체를 연결한다. 일반 target flow MSE만으로 시작하지 않는다.

```text
Delta v_teacher = F(x_t, t, content + @artist) - F(x_t, t, content)
Delta v_student = F_style(x_t, t, content, z_ref) - F(x_t, t, content)
```

초기 online loss는 다음을 포함한다.

- teacher RMS로 정규화한 velocity-residual Huber
- velocity-residual direction cosine
- log-RMS magnitude
- offline block attention-effect anchor
- 낮은 비중의 target rectified-flow MSE

초기에는 native artist residual과 block anchor를 강하게 두고 target flow MSE는 약하게 둔다.
Validation에서 native residual 정렬과 correct/wrong gap이 유지될 때 target flow 비중을 점차
높인다. Offline loss를 갑자기 제거하지 않는다.

일반 target residual에는 style 외에도 pose, 구도, caption 누락, texture, Frozen Anima의 일반
예측 오차가 섞인다. 따라서 이를 처음부터 높은 비중으로 사용해 약 394M 연결부 전체를 열지
않는다.

## 8. Stage B — 제한된 Resampler 공동학습

Online native residual이 통과한 뒤에만 Resampler를 연다.

1. 최종 `1536 -> 1024` projection
2. 마지막 encoder block
3. 필요할 때만 마지막 두 block

순서로 개방한다. Resampler LR은 connector LR의 `5~10%`에서 시작한다. Frozen Stage-R token에
대한 cosine anchor와 약한 reconstruction을 유지하며 prototype은 끄거나 매우 약하게 둔다.
목표는 새로운 작가 classifier를 만드는 것이 아니라 같은 작가의 다른 content reference가
native artist effect를 더 쉽게 만들게 하는 것이다.

## 9. Multi-reference 전환

Single-reference held-out 성능이 통과한 뒤에만 Minimal Set Aggregator를 연다.

- reference 수: 1/2/4/8
- target-excluded same-artist reference
- reference 순서 무작위화
- reference dropout
- 중복 reference 금지

Reference 수가 늘어날수록 teacher residual 정렬과 생성 품질이 실제로 개선되는지 측정한다.
Aggregator를 연 뒤 1-reference 성능이 떨어지면 다음 단계로 넘어가지 않는다.

## 10. Null 및 conditioning 처리 규칙

서로 다른 null 개념을 분리한다.

1. **Post-LLM trailing-zero positions**: Anima text cross-attention은 text padding mask를 받지
   않으므로 항상 512개 conditioning 위치 전체를 사용한다. Trailing zero 위치도 softmax
   normalization에 포함되므로 절대 trim하지 않는다.
2. **Empty-prompt text conditioning**: Text CFG의 unconditional branch로 별도 cache하며 역시
   512개 위치를 유지한다.
3. **Null style**: Style CFG 및 style dropout용이다. 최종 정의는 실제 style branch bypass로
   하거나, learnable null token을 사용할 경우 충분한 style dropout으로 반드시 학습한다.

Learnable null-style token이 학습되지 않은 상태에서 이를 Style CFG의 기준점으로 사용하지
않는다. Paired-flow validation의 baseline은 항상 실제 style branch bypass로 계산한다.

## 11. 정성 및 정량 평가

### 11.1 현재 1,000-step 모델 기준선

현재 실패 계열의 1,000-step checkpoint는 삭제하지 않고 다음 비교의 기준선으로 보존한다.

- 동일 positive/negative prompt
- 동일 post-LLM conditioning
- 동일 초기 noise와 seed
- 동일 1MP 해상도와 30-step schedule
- Text CFG 4 고정
- Style CFG 1과 4 각각
- 바꾸는 것은 reference artist뿐

한 비교 sheet에 다음을 포함한다.

- style bypass
- null style
- 서로 다른 작가 5~8명
- 각 작가의 exact-self reference
- 각 작가의 target-excluded held-out references

모든 작가 출력이 비슷하면 common-effect shortcut, self만 다르면 image/content-copy shortcut,
same-artist held-out reference끼리 일관되고 작가 간 출력이 달라지면 artist-specific mapping으로
판정한다.

### 11.2 각 단계의 공통 지표

- teacher/student normalized residual loss와 cosine
- zero/bypass 대비 improvement
- correct/wrong cosine 및 improvement gap
- output/teacher RMS ratio
- timestep 및 28-block별 지표
- seen/unseen artist와 seen/held-out content 분리
- 동일 prompt/seed의 다중 작가 contact sheet

## 12. 중단 및 진행 규칙

- Stage R은 retrieval 단독 최고점으로 선택하지 않는다.
- Stage A0 correct/wrong gap이 양수가 아니면 full effect를 열지 않는다.
- Stage A1 unseen-artist gap이 유지되지 않으면 online으로 가지 않는다.
- Online paired-flow가 감소하면서 offline artist alignment도 감소하면 즉시 해당 checkpoint 이후
  학습을 중단한다.
- 출력 크기만 증가하고 방향 정렬이 악화되면 magnitude나 LR을 더 올리지 않는다.
- Meta-test는 validation 기반 선택이 끝난 뒤 한 번만 사용한다.
- 각 단계는 새 output directory와 W&B run ID를 사용하고 이전 checkpoint를 덮어쓰지 않는다.

## 13. 실행 순서 요약

```text
R   Resampler reconstruction + moderated artist supervision
    -> Pareto-best checkpoint

C   New frozen Resampler token cache

A0  Bridge + Centered Artist-Specific Head
    -> same-content/same-Q centered all-pairs training

A1  Shared common effect
    -> group connector
    -> K/V delta
    -> optional O delta

O   Online native artist velocity-residual distillation
    + persistent offline block anchor
    + gradual target flow MSE

B   Partial Resampler joint tuning

M   Target-excluded 1/2/4/8-reference Minimal Aggregator training

D   Synthetic-to-real Danbooru curriculum
```

이 순서를 통해 실패 원인을 한 번에 하나씩 분리한다. Resampler가 정보를 보존하는지, 연결부가
reference artist를 실제로 사용하는지, online flow가 그 정렬을 유지하면서 생성 품질을
개선하는지를 각 단계의 독립적인 통과 조건으로 확인한다.
