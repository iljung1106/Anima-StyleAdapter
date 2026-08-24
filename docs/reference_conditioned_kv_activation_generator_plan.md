# Reference-conditioned K/V Activation Generator 학습 계획

## 목표와 추론 계약

레퍼런스 이미지로부터 Anima 28개 블록의 native text K/V에 더할
`delta K/delta V` activation을 직접 생성한다. 작가 태그 Teacher는 사용하지
않으며 LoRA는 오프라인 Teacher로만 사용한다.

추론 경로는 다음으로 고정한다.

`references -> frozen visual features/Resampler -> style memory -> text-conditioned K/V residual generator -> native text attention`

추론 시 artist ID, LoRA bank, 검색 결과 또는 mixture coefficient를 입력하지
않는다. Native Q, text softmax, O와 `gate_cross`는 그대로 사용하고 별도
style-attention 및 learned Common K/V 경로는 두지 않는다. Strength 0은 원본
Anima와 정확히 같고, strength는 reference-conditioned residual에만 적용한다.

## Teacher 데이터

- 먼저 보존된 rank-16 K/V-only 64명과 512장 synthetic reference로 구조를
  검증한다. Human reference도 같은 작가에 대해 별도 domain으로 사용한다.
- 검증 후 완성된 320명 K/V-only factor bank와 Human reference로 확대한다.
- Reference 생성용 content prompt는 소수의 고정 template를 반복하지 않고,
  기존 Anima caption manifest의 서로 다른 이미지 caption에서 무작위로
  표본화한다. Artist tag는 제거하고 caption ID, seed와 선택된 Teacher
  mixture를 manifest에 함께 기록해 정확히 재현 가능하게 한다.
- K/V 증류에 쓰는 text context도 같은 caption pool에서 무작위로 뽑되,
  reference를 생성한 caption에만 고정하지 않는다. 이를 통해 학생이 한
  LoRA/mixture를 특정 content prompt와 결합해 외우는 것을 막는다.
- Teacher 분포는 단일 25%, 합이 1인 positive pair/triple 25%, 양수 합이
  1.0--1.5인 amplified mixture 25%, signed mixture 25%를 기본값으로 한다.
- Signed coefficient는 개별 `[-0.5, 1.5]`, negative mass `<= 0.5`로 제한하고,
  실제 K/V/attention residual RMS가 단일 Teacher 중앙값의 0.2--1.5배를
  벗어나는 조합은 재표본화한다.
- 임의 가중치는 reference만으로 식별할 수 없으므로 pair/triple/amplified/
  signed Teacher는 반드시 실제 merged K/V-LoRA로 이미지를 생성한다. 학생은
  이 materialized image만 보며 계수는 받지 않는다.

## 학생 구조

- Frozen Resampler의 typed reference token을 얕은 Reader와 2-layer Set
  Transformer로 합쳐 32x1024 style memory를 만든다.
- 각 블록에서 512x1024 text context가 style memory를 cross-attention으로
  읽고, 독립적인 block output head가 2048차원 `delta K`와 2048차원
  `delta V`를 출력한다.
- K/V residual은 native projection 출력에 더한 뒤 원래 `k_norm/v_norm`을
  통과시킨다. 28개 block output head는 공유하지 않는다.
- Generator 내부 폭은 우선 256으로 두고 전체 학생을 약 50--90M 범위로
  제한한다. 마지막 K/V head는 작은 nonzero Xavier scale로 초기화한다.

## 학습 단계

### A. Dense K/V bootstrap

Frozen Anima 전체를 실행하지 않고 cached text context와 LoRA factor로
Teacher `delta K/delta V`를 GPU에서 즉시 계산한다. 매 step 4--7개 블록을
순환해 모든 블록을 균등 노출한다. Resampler와 Reader는 동결하고 Generator만
학습한다.

주 loss는 pre-normalization K/V Huber이며, block별 Teacher RMS로 정규화한
방향 loss를 약하게 함께 사용한다.

### B. Native-attention functional alignment

같은 Q와 frozen native O에서 학생과 Teacher의 attention output을 맞춘다.
K/V loss는 계속 유지한다. 이 단계부터 Reader의 마지막 pooling/mixer만
Generator LR의 0.1배로 열 수 있다.

이번 실험은 Phase B에서 종료한다. Frozen Anima 전체의 final velocity/flow
loss는 사용하지 않는다. 초기 권장 상대 비중은 `K/V activation 1.0 :
attention output 0.3`이다. Prototype, common-output, native artist-tag 및
복잡한 magnitude 규제도 사용하지 않는다.

## 검증과 진행 기준

- 64명 pilot에서 train/heldout prompt별 K/V cosine, normalized error,
  attention-output cosine을 측정한다.
- 동일 prompt/seed에서 8개 작가의 결과가 서로 구별되고, 1/2/4 reference
  증가가 평균적으로 악화되지 않는지 확인한다.
- 단일, positive mixture, amplified, signed를 별도 표로 기록한다.
- 64명 pilot이 기능적으로 수렴한 경우에만 320명 전체와 더 많은
  materialized mixture로 확대한다.
- 최종 acceptance는 Teacher 회귀값만이 아니라 artist-disjoint raw reference의
  시각적 스타일 차이와 원본 content 보존을 함께 기준으로 한다.
