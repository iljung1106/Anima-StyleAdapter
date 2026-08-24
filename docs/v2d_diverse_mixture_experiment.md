# v2d Diverse-Mixture Experiment

## 목적

검증된 `direct-reference-kv-distillation-v2d`의 모델, pretrained Reader,
Common/Artist 분리 및 매 스텝 full-Anima functional objective를 유지한다.
변경 변수는 full-LoRA Teacher의 혼합 분포뿐이다.

## Teacher 혼합

- 기반 Teacher: 기존 256개 rank-16 full LoRA
- positive pair 116개, positive triple 76개
- amplified pair/triple 116개, 계수 합 `1.0--1.7`
- signed pair/triple 76개
- signed는 대수적 합 1.0을 유지하고 `sum(abs(weight)) <= 1.7`로 제한한다.
- 3-way signed는 양수 두 개와 음수 한 개로 구성한다.

각 혼합은 실제 merged-LoRA로 네 장의 reference를 생성한다. Student 입력에는
LoRA ID, 혼합 계수 또는 component reference가 들어가지 않으며, materialized
image를 Frozen Resampler로 인코딩한 토큰만 들어간다.
각 mixture는 준비된 artist-free content bank에서 서로 다른 네 prompt를
결정론적으로 무작위 선택해, 모든 스타일이 같은 composition template를
반복하는 content shortcut을 막는다.

## 학습

- 0--1,500 step: single full-LoRA와 native artist teacher를 1:1로 사용
- 1,500 step 이후: native artist, single LoRA, mixture LoRA를 1:1:1로 사용
- mixture 내부 비율: pair 30%, triple 20%, amplified 30%, signed 20%
- Reader는 v2d와 같이 `detail_style_reader_pretrain_v1`을 동결한다.
- fresh Common/Artist K/V와 기존 강도 calibration을 사용한다.
- teacher-decomposed final-velocity, centered direction/magnitude, functional
  InfoNCE 및 common-ratio objective는 v2d와 동일하다.

## 판정

single-reference 품질을 훼손하지 않으면서 positive triple, amplified 및 signed
Teacher의 centered cosine/InfoNCE와 실제 고정 샘플 다양성이 개선되어야 한다.
실제-Q block loss, K/V operator 회귀, 512-artist 확대는 이번 실험에 넣지 않는다.
