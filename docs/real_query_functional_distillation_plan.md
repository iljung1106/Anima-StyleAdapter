# Real-Q Functional K/V Distillation

## 목적

무작위 Gaussian Q에서 K/V-LoRA를 구분하던 기존 capacity probe를 실제
Frozen Anima 생성 경로에 맞춘다. 학생은 reference image만 받아 각 블록의
저랭크 K/V operator를 만들며, artist ID나 LoRA 계수는 입력받지 않는다.

## 데이터

- Teacher: 완료된 320개 rank-16 K/V-only LoRA
- Human reference: 기존 frozen Dual-query Resampler token cache
- Synthetic reference: 각 Teacher LoRA로 8장씩 새로 생성하고 같은 Resampler로 캐시
- 두 reference domain은 한 set으로 섞지 않고 batch 단위로 번갈아 학습한다.

## 실제-Q 캐시

기존 256-content × 4-timestep base trajectory에서 64 content를 균등 선택한다.
Frozen Anima를 실제 `x_t`, timestep, 512-token prompt context로 실행하고 각
블록 `q_norm` 뒤의 query 64개를 저장한다. Q, text context, noisy latent,
timestep 및 base velocity는 항상 같은 source row를 사용한다.

## 학습

1. 같은 실제 Q와 text context에 Teacher 및 Student K/V를 적용한다.
2. native K/V normalization, attention softmax, pretrained O 뒤의 effect를 계산한다.
3. 동일 조건의 8-artist controlled batch 평균을 제거하고 centered Huber,
   cosine, magnitude 및 relation loss를 사용한다.
4. pre-normalization K/V 회귀는 weight 0.05의 보조 신호로만 둔다.
5. Reader는 1,500 step까지 완전 동결한다. 이후 마지막 mixer만 LR `5e-6`로 연다.
6. 500 step부터 8 step마다 2-artist full Frozen Anima forward를 실행한다.
   Teacher/Student가 앞 블록을 변경해 만든 후속 Q까지 포함한 final velocity
   centered effect를 맞춘다.

## 검증

- 마지막 32개 Teacher artist는 학습에서 제외한다.
- 실제-Q content 64개 중 마지막 8개도 학습에서 제외한다.
- Human과 Synthetic heldout reference를 각각 기록한다.
- 1,000 step부터 고정 외부 reference 패널을 생성해 수치와 시각적 의미가
  함께 개선되는지를 확인한다.

단순 common-output penalty, rank 확대 또는 strength 증가는 실제-Q 의미 정렬을
대체하지 않는다. 실제-Q 검증과 고정 샘플이 모두 개선될 때만 규모를 늘린다.

## 실행 결과 (2026-08-25)

- 실제-Q bank: 64 content × 4 timestep × 28 block × 64 query, 약 1.88 GB
- Synthetic reference: 320 artist × 8 image = 2,560장
- Synthetic frozen-Resampler cache: 2,560개, 약 443 MB
- 본 학습: 4,000 step, 288 train / 32 heldout artist, 약 436초
- Reader는 1,500 step까지 동결했고 이후 마지막 mixer만 열었다.
- sparse full-Anima final-velocity 증류는 500 step부터 8 step마다 실행했다.

heldout Human/Synthetic 평균 centered functional cosine과 16-way relation
accuracy는 모두 2,750 step에서 최고였다.

| Step | Mean centered cosine | Mean relation accuracy |
|---:|---:|---:|
| 1,500 | 0.05264 | 0.14397 |
| 2,250 | 0.05829 | 0.14174 |
| **2,750** | **0.06880** | **0.15960** |
| 3,500 | 0.06223 | 0.13616 |
| 4,000 | 0.05957 | 0.14397 |

16-way 무작위 정확도는 0.0625이므로 실제-Q 조건의 작가 구분 신호는
학습되었다. 그러나 외부 고정 reference 생성에서는 2,750 step도 reference별
화풍 차이가 약했다. 4,000 step은 latent artist-centered effect RMS가 더 컸지만
얼굴·명암·구도와 같은 일반적 변형이 주로 증가했고 heldout teacher 정렬은
퇴행했다. 따라서 이 실험의 보존 후보는 2,750 step이며, 단순 장기 학습이나
strength 증가는 다음 단계로 채택하지 않는다.

다음 실험은 Reader 표현이 teacher operator를 예측할 수 있는지를 별도 probe로
분리해 확인하고, synthetic reference에 실제 Teacher LoRA의 시각적 효과가
충분히 드러나는지 선별한 뒤 진행해야 한다. 현재 결과만으로는 실제-Q 사용이
필요조건임은 확인했지만 reference 의미 정렬 문제를 해결했다고 볼 수 없다.
