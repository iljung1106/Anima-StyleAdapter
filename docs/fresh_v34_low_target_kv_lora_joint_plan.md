# Fresh v34 low-target + K/V-LoRA joint experiment

## 목적

v34의 검증된 `Reader -> Common/Artist Style Cross-Attention` 구조만 유지하고,
v31/v34 체크포인트를 재사용하지 않은 채 시각 레퍼런스로 K/V-only LoRA의 기능적
효과를 일반화하는지 검증한다. 런타임에는 LoRA ID, 사전, 검색 및 LoRA 가중치가
필요하지 않다.

## 초기화와 구조

- Frozen dual-query Resampler 캐시는 재사용하지만 Reader와 Style Adapter는 새로
  초기화한다.
- 84개 타입 토큰을 typed Reader가 reference당 읽어 28개 canonical style token을
  만든다.
- Common과 Artist 경로를 분리하고, Artist는 4개 Xavier K/V shared base와 block별
  rank-64 delta를 사용한다.
- Frozen Anima의 Q/O와 `gate_cross`는 재사용한다.
- Common은 첫 500 optimizer step만 학습한 뒤 동결한다.

## 공동 학습

매 optimizer step에서 다음 두 목표를 모두 역전파한다.

1. 실제 human target의 rectified-flow MSE
2. 동일한 실제 Anima Q에서 측정해 캐시한 단일 K/V-only LoRA의 centered
   post-attention functional effect

LoRA 혼합 teacher와 native artist-tag teacher는 이번 1차 실험에서 제외한다.
LoRA reference는 human/synthetic 도메인을 교대로 쓰며 Artist ID를 모델 입력으로
주지 않는다.

## 낮은 target 포함률

- step 1부터 일반 prompt 배치의 target 포함 확률은 0.20이다.
- 이 확률은 step 4,000까지 0으로 감소한다.
- `empty` prompt 10%는 content 조건이 없으므로 single exact-self를 강제한다.
- reference 수는 1/2/4장을 각각 60/30/10%로 사용한다.

따라서 초반 전체 effective target 포함률은 약 28%이고, 후반에는 empty-prompt의
10%만 남는다. 처음부터 이미지 복사에 의존하지 않으면서 content-free 학습 계약은
유지한다.

## 실행 및 선택

- 총 8,000 step이고 체크포인트는 250 step마다 저장한다.
- 500 step마다 fixed-reference 샘플과 train 4명/validation 4명의
  target/reference/frozen-Anima/
  styled 결과를 묶은 `val/functional/panel`을 생성한다.
- 1,000 step마다 heldout controlled few-shot의 1/4-reference만 한 장으로
  비교한다. 별도 quick/diverse sheet는 생성하지 않는다.
- 주 선택 기준은 heldout 작가 분리, single-reference 재현, reference 수 증가 시
  개선, 그리고 frozen Anima의 형상 보존이다. Training teacher 회귀값만으로 모델을
선택하지 않는다.
