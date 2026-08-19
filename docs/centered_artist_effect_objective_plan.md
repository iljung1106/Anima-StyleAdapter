# Centered artist-effect objective

## 목적

기존 single-target flow MSE는 그대로 유지한다. 추가 목표는 reference의
개별 내용이나 모든 작가에 공통인 출력이 아니라, 같은 작가의 서로 다른
그림에서 반복되는 실제 Anima style residual을 학습·측정하는 것이다.

## Functional artist effect

동일한 `x_t`, text context, noise, timestep에서 두 reference view를 사용한다.

- view A: target 이미지 한 장. Functional teacher로만 사용하며 detach한다.
- view B: target과 겹치지 않는 같은 작가의 heldout reference. Student 경로다.
- `style effect = styled velocity - frozen Anima base velocity`
- 각 view에서 batch artist mean을 빼 공통 출력을 제거한다.
- 2x/4x average-pooled latent residual을 사용해 고주파·이미지별 잡음을 줄인다.
- symmetric artist InfoNCE와 signed repeatable-effect ratio를 함께 최적화한다.

Functional loss는 250 step부터 시작해 1,000 step에 weight `0.02`가 되며,
비용을 제한하기 위해 4 step마다 실행한다. Target view는 detach하므로 추가로
필요한 heldout student pass는 primary loss를 먼저 backward한 뒤 순차 실행한다.

## Common output와 artist magnitude

Centered objective는 공통 성분에 불변이므로, raw heldout residual의 작가 평균
RMS 비율에 별도 hinge loss를 건다. Batch 4의 독립 방향 기준이 약 `0.5`임을
고려해 threshold를 250→2,000 step 동안 `0.90→0.65`로 낮춘다. Weight는
250→1,000 step 동안 `0.04`로 ramp한다. 분모 RMS는 detach하여 무관한
residual energy를 부풀리는 해법을 막는다.

공통 성분만 줄여 전체 출력이 0으로 수축하지 않도록, centered heldout
residual을 detached exact-target teacher 방향에 투영한 절대 계수를 사용한다.
투영 하한은 250→2,000 step 동안 `0.35→0.70`, 상한은 `1.25`이며 상한
penalty는 하한의 `0.25`배다. Weight는 250→1,000 step 동안 `0.05`로
ramp한다. 전체 RMS가 아니라 teacher 방향 성분에만 하한을 적용하므로
직교 노이즈나 VAE texture로 크기만 채울 수 없다.

## Episodic artist prototype

28개 최종 token을 spatial 16개, global 8개, summary 4개로 나누어 각 type의
평균을 이어 붙인다. 별도 학습형 분류 head나 작가별 parameter table은 두지
않는다. Target view embedding을 같은 작가의 heldout view prototype과
분류하는 symmetric episodic contrastive loss를 사용한다.

Prototype loss는 250→1,000 step 동안 weight `0.01`로 ramp하며 2 step마다
실행한다. 이 loss는 reader/tokenizer에 작용하고, functional loss는 reader와
Style K/V 양쪽에 작용한다.

## 전체 비중

- flow MSE: `1.0`
- reconstruction: `0.01`
- correct-vs-wrong flow ranking: 최대 `0.05` (250→750)
- centered functional artist effect: 최대 `0.02` (250→1,000, 매 4 step)
- raw common-output hinge: 최대 `0.04` (250→1,000, 매 4 step)
- centered artist-magnitude band: 최대 `0.05` (250→1,000, 매 4 step)
- episodic artist prototype: 최대 `0.01` (250→1,000, 매 2 step)
- native centered teacher objective: 기존 주기와 비중 유지

모든 학습 batch는 작가 label이 명확한 Anima/Danbooru corpus에서만 가져온다.
MegaStyle 혼합은 v10부터 비활성화하며 기존 캐시는 삭제하지 않는다.

## Validation

Validation 작가와 heldout 이미지에서 같은 matched flow probe를 사용해 다음을
기록한다. 네 timestep(`0.2, 0.45, 0.7, 0.9`)에서 평가한다.

- same/wrong artist centered cosine 및 gap
- cross-reference artist retrieval top-1과 random 기준
- signed repeatable-effect ratio
- functional ICC: between-artist variance / (between + within variance)
- common-output ratio와 view-difference ratio
- type-wise token prototype cosine gap 및 retrieval top-1

모델 선택에서는 기존 flow validation과 시각 샘플을 유지하면서,
`functional_artist_icc`, cosine gap, heldout retrieval이 함께 상승하고
common-output ratio가 악화되지 않는 구간을 우선한다.
