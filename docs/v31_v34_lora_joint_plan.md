# v31/v34 + K/V-LoRA 보조 증류 실험

## 목적

이 실험은 새로운 LoRA 전용 학습법이 아니다. 원본 v31의 Common/Artist 분리와
원본 v34의 human-flow 및 reference 커리큘럼을 보존하고, K/V-only LoRA의
functional effect를 추가 teacher로 사용한다.

## 초기값과 원본 신호

- v31 `step-3500`의 Reader와 Style Adapter 가중치에서 시작하고 optimizer는 새로 만든다.
- v34와 동일하게 0--250 step은 exact single-reference, 250--1000 step은 target 포함
  1/2/4-reference, 1000--2000 step은 target 포함률을 0으로 줄이며 1/2/4/8-reference를 쓴다.
- 매 optimizer step의 human rectified-flow objective를 유지한다.
- disjoint heldout-view functional objective, controlled artist objective, main/common-output
  penalty, synthetic Anima artist-tag teacher를 원본 v34의 빈도와 가중치로 유지한다.
- 샘플 panel과 fixed-reference sheet는 각각 500 step마다 생성한다.

## 추가되는 LoRA teacher

- LoRA functional teacher는 원본 loss를 대체하지 않고 같은 optimizer step에 추가 backward된다.
- 매 2 step마다 실행하며 전체 backward scale은 `0.25`이다.
- 0--500 step은 individual K/V-only LoRA만 사용한다.
- 이후 teacher 종류는 `single, single, pair, triple` 순환으로 individual teacher를 더 자주 쓴다.
- reference 도메인은 human과 LoRA-generated synthetic 이미지를 번갈아 사용한다.
- 같은 frozen Anima의 content, timestep, noisy latent, text context에서 Student effect와
  teacher effect를 비교한다. Artist-centered Huber/direction/magnitude와 all-wrong InfoNCE를
  주 신호로 사용하고, reference-independent common 회귀는 약하게 둔다.

## 비교 해석

출력 폴더와 W&B run은 `detail-style-v34-lora-joint-v1`로 분리한다. 따라서 원본 v34와
비교할 때 달라지는 주된 독립변수는 LoRA functional auxiliary이며, 중지된
`fresh_v34_low_target_kv_lora_joint` 실험은 이 비교의 초기값으로 사용하지 않는다.
