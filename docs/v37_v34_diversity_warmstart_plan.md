# v37b: v34 다양성 보존 warm-start 실험

## 목적

`detail-style-v34-anti-common-functional` 500스텝은 동일 prompt/seed의
고정 레퍼런스 평가에서 1x 작가 간 다양성 비율 `0.897`을 기록했다.
Artist-only는 `0.909`, Common-only는 `0`이므로 차이는 실제 Artist 경로에서
발생했다. v37b는 이 가중치를 출발점으로 삼되, 장기 학습에서 같은 작가의
서로 다른 작품과 서로 다른 작가의 출력이 합쳐지는 목적함수를 제거한다.

## 초기화

- Reader와 Adapter는 v34 step 500에서 불러온다.
- optimizer, scheduler, RNG, global step은 새로 시작한다.
- 구조와 파라미터 형상은 v34와 동일하게 유지한다.

## 기능적 작가 손실

- 서로 다른 작품을 같은 작가라는 이유만으로 계속 수축시키는 symmetric
  cross-view InfoNCE를 사용하지 않는다.
- 동일 `x_t`, timestep, prompt, frozen-Anima Q에서 만든 최종 centered
  velocity effect에 대해 matching reference가 배치의 모든 wrong artist보다
  margin `0.10`만큼 우세하도록 hinge ranking을 적용한다.
- 같은 작가의 disjoint view에는 cosine `0.25`의 약한 하한만 요구한다.
  하한을 만족하면 추가로 가까워지게 하는 gradient는 없다.
- Reader reconstruction은 `0.005`로 유지하여 reference별 정보를 보존한다.

## Common과 Artist 분리

최종 style effect는 다음과 같이 정의한다.

\[
\Delta v = C(Q,t) + s_{artist} A(Q,t,R)
\]

- Common K/V는 계속 학습하지만 LR은 `3e-5`로 낮춘다.
- 일반 flow에서 Common gradient는 `0.10`배만 전달한다.
- native teacher step에서는 Common-only forward로 native cross-artist mean의
  방향을 회귀하고 RMS를 `0.90~1.10` band에 둔다.
- Artist teacher forward는 Common을 우회한 centered native residual만
  학습한다. 두 teacher objective는 서로의 경로에 역전파하지 않는다.
- 기존의 total-output common suppression은 끈다. 의도된 native Common을
  0으로 만드는 목적과 충돌하기 때문이다.

## 추론 강도

- 사용자 strength는 Artist residual에만 곱한다.
- Common은 항상 1x native scaffold를 유지한다.
- text CFG는 기존처럼 text에만 적용한다.
- 어댑터 전체 비활성화는 별도의 bypass/master control을 사용한다.

## 관찰 기준

250스텝마다 수치 검증과 고정 레퍼런스 1x/2x 시트를 만든다. 특히 다음을
v34 step 500 기준과 비교한다.

- fixed-reference pairwise RMS 및 diversity ratio
- Artist-only diversity
- Common/native RMS와 방향 정렬
- final centered all-wrong margin violation
- corresponding-view cosine과 view-difference ratio
- heldout flow/output 지표 및 시각적 파손 여부

우선 2,000스텝까지 다양성이 유지되는지 판단하고, 유효할 때만 8,000스텝
커리큘럼을 계속한다.
