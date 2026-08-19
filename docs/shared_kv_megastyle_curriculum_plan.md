# Shared K/V 및 MegaStyle 상시 혼합 계획

## 목적

기존 detail-preserving Style Cross-Attention은 28개 block마다 독립적인
`1024 -> 2048` full-rank K/V를 두어 약 117.4M개의 K/V 파라미터를 학습했다.
관찰된 주 실패는 방향이 맞지 않는 style residual을 작은 `alpha_b`로 축소해
손실을 피하고, target이 빠진 뒤에는 이미 낮아진 LR 때문에 다시 정렬하지 못한
것이다. 다음 학습은 표현 공유, dense 초기 지도, metric-gated curriculum 및
일정한 외부 스타일 데이터 노출을 사용한다.

## Style branch 강도

`alpha_b`는 절대 강도가 아니라 block 사이의 상대 profile만 나타낸다.

```text
alpha_b = global_gain * relative_block_gain_b
median(relative_block_gain) = 1
relative_block_gain in [0.5, 2.0]
```

어떤 block도 0으로 비활성화하지 않는다. Anima가 projected K/V에 RMSNorm을
적용하므로 K/V weight 전체 scale을 바꾸는 방법으로 출력 크기를 조절하지 않는다.
절대 크기는 post-attention `global_gain`, teacher-aligned magnitude loss 및 추론
style strength가 담당한다. 초기 `global_gain`은 지나치게 작은 0.02 계열을 쓰지
않고 0.25--0.5 또는 안정적인 native 초기화에서는 1을 사용한다.

## Shared K/V

K와 V는 각각 네 개의 full-rank Shared Base와 block별 rank-64 delta로 구성한다.

```text
W_b = sum_c softmax(mix_b)[c] * W_shared_c + B_b A_b
```

block mixing은 측정된 cluster에 거의 one-hot으로 초기화하고 LoRA up 행렬은
zero-init한다. Q, O 및 `gate_cross(t)`는 block별 native Anima 모듈을 그대로
사용한다. 현재 차원에서 Shared Base 4개와 rank-64 delta는 약 27.8M K/V
파라미터로, 독립 full-rank 구조보다 약 76% 작다.

네 개 base는 block 번호를 임의로 구간화하지 않고 다음 복합 유사도로 고른다.

- 35%: 동일 Q에서 측정한 centered artist teacher residual의 linear CKA
- 25%: native Q activation의 linear CKA
- 15%: native K 입력 부분공간 유사도
- 15%: native V 입력 부분공간 유사도
- 10%: artist/content/timestep별 teacher residual RMS profile 상관

복합 거리에 4-medoids를 적용한다. 각 cluster의 medoid native K/V가 Shared Base
초기값 후보이며, 최종 JSON에 cluster, medoid, 상대 block gain을 모두 기록한다.

## Metric-gated curriculum과 LR

초기에는 exact-self와 dense centered residual teacher로 방향과 절대 크기를 함께
정렬한다. exact-self만 사용해 복사 shortcut에 머물지 않도록 target-excluded
same-style episode도 소량 유지한다.

target 포함률은 고정 step에서 자동으로 내리지 않는다. target-excluded heldout에서
다음 조건을 네 번 연속 통과할 때에만 다음 단계로 이동한다.

- paired flow improvement의 bootstrap confidence interval 하한이 0보다 큼
- correct-vs-wrong 성능이 chance보다 유의미하게 높음
- residual 방향과 teacher 대비 절대 RMS가 허용 범위 안에 있음

그 뒤 target 포함률을 `1.0 -> 0.5`, 다시 조건을 통과하면 `0.5 -> 0`으로
낮춘다. 악화 시 현재 확률을 유지한다. LR은 warmup 뒤 이 전체 정렬/annealing
구간에서 plateau를 유지하고, target 포함률이 0인 본학습이 안정된 뒤에만 cosine
decay를 시작한다. 새 LoRA delta group을 여는 경우 해당 group만 짧게 re-warmup한다.

## MegaStyle-1.4M 상시 혼합

MegaStyle은 style ID 단위로 split하고, 같은 style의 서로 다른 content ID에서
reference와 target을 선택한다. style description은 grouping과 점검에만 쓰며 student
text input에는 넣지 않는다. target text는 content caption과 Anima용 태그 캐시를
사용한다.

학습 microbatch의 15%를 모든 curriculum 단계에서 MegaStyle로 고정한다. 20개
microbatch마다 정확히 3개를 MegaStyle로 선택하는 결정적 저불일치 스케줄을
사용한다. 나머지 85%는 기존 Danbooru/Anima 학습 loader다. dense artist-tag teacher
step은 이 비율과 별도로 계속 실행하므로 MegaStyle style ID가 artist teacher bank에
존재할 필요가 없다.

검증은 기존 anime heldout과 MegaStyle style-ID heldout을 분리해 보고한다. 원본
Parquet 다운로드 후 C-RADIO, frozen Dual-query Resampler token, Qwen VAE latent,
multimode text cache를 별도 디렉터리에 생성해야 실제 혼합 학습을 시작할 수 있다.

MegaStyle 라이선스는 연구·교육 용도로 제한되므로 이 데이터를 사용한 checkpoint는
상업용 계열과 분리한다.
