# Direct Reference Whole-Model Functional 10k 학습 계획

## 목표와 추론 구조

Styled reference만으로 Anima의 native text K/V residual을 생성한다. 추론에는
teacher, 작가 ID, base reference, mixture coefficient가 필요하지 않다.

```text
reference image(s)
  -> frozen visual Resampler tokens
  -> trainable Reader / style aggregator
  -> style memory

current text context + style memory + block index
  -> reference-conditioned direct K/V generator
  -> raw delta-K / delta-V
  -> frozen Anima native K/V projection에 가산
  -> k_norm / v_norm / attention / residual blocks / final velocity
```

- Generator와 optimizer는 새로 초기화한다.
- Reader만 기존 visual bootstrap 체크포인트에서 초기화하고 end-to-end로 연다.
- Common 분기, batch/population mean subtraction, `Reader(styled)-Reader(base)`를
  사용하지 않는다.
- sample-wise style normalization을 사용하지 않아 reference/mixture 강도를
  보존한다.
- block별 context projection과 output head는 유지하되, 모든 style key에 같은
  값을 더해 softmax에서 상쇄되는 기존 `block_embedding`은 제거한다.

## Teacher 데이터

```text
K/V-only LoRA teachers: artist_kv_lora_teachers_rank16_320_b2_v3
final-velocity cache:
  kv_lora_functional_teacher_bank_rank16_320_mixtures_compact16x4_v1
```

캐시는 다음 576개 teacher 효과를 같은 base/noisy/text/timestep 조건에서
전체 frozen Anima에 실행한 결과로 저장한다.

- single 320
- pair 64
- triple 64
- amplified 64
- signed 64
- 각 target당 16 content x 4 timestep

Mixture는 새로 무작위 생성하지 않고 기존 materialized mixture reference와
동일한 style ID와 signed/amplified weight를 사용한다.

## 손실

### 초기 block bootstrap

초반에만 raw teacher delta와 local native-attention effect를 사용해 K/V 좌표계와
초기 scale을 잡는다. 두 loss 모두 step 2,000에서 정확히 0이 된다.

```text
block_weight(step):
  0-500     = 1
  500-2000  = linear 1 -> 0
  2000+     = 0
```

### Whole-model teacher functional loss

```text
teacher_effect = cached_teacher_prediction - cached_base_prediction
student_effect = student_prediction - cached_base_prediction
```

주 손실은 final velocity effect의 normalized Huber, cosine direction, RMS ratio로
구성한다. 공통방향 억제는 mean을 빼지 않고 teacher geometry를 기준으로 한다.

```text
L_common = mean ReLU(
  cosine(student_effect_i, student_effect_j)
  - cosine(teacher_effect_i, teacher_effect_j)
)
```

따라서 teacher가 실제로 공유하는 방향은 허용하고 student가 그보다 더 한
방향으로 붕괴하는 경우만 벌점으로 준다.

### Human functional flow loss

```text
base    = FrozenAnima(no adapter)
correct = FrozenAnima(correct reference)
wrong   = FrozenAnima(shuffled reference)

desired_delta = target_velocity - base
student_delta = correct - base
```

- residual Huber와 cosine으로 `student_delta`를 `desired_delta`에 맞춘다.
- 절대 MSE margin 대신 base 대비 상대 개선율로 correct/wrong ranking을 계산한다.

```text
gain = (base_mse - styled_mse) / clamp(base_mse)
L_rank = ReLU(margin + wrong_gain - correct_gain)
```

## 10k 커리큘럼

| Step | 주 목표 | Update 구성 |
|---:|---|---|
| 0-500 | K/V 좌표계 초기화 | block teacher 중심, whole-model teacher 시작 |
| 500-2,000 | 국소 목표 제거 | block `1 -> 0`, whole-model teacher `0 -> 1` |
| 2,000-5,000 | 누적 기능 재현 | cached whole-model teacher 100% |
| 5,000-10,000 | 실제 이미지 적응 | human flow 75%, whole-model teacher 25% |

2,000 이후 raw block delta와 local functional-attention loss는 optimizer에 전혀
사용하지 않는다. 기록이 필요하면 gradient 없는 진단 지표로만 계산한다.

## Batch와 최적화

- cached whole-model teacher: physical batch 4 우선, OOM/여유 부족 시 2
- human flow correct/wrong: physical batch 2
- gradient accumulation: 2 microbatches
- Generator 기준 LR: `2e-6`
- Reader LR: Generator의 `0.15-0.20배`
- optimizer와 scheduler state는 신규 생성
- 100-step smoke에서 unclipped norm을 수집하고 p99 이상 이상치만 clipping

Pairwise common loss는 accumulation된 effective batch가 아니라 실제 physical
batch에서 계산된다. 따라서 teacher 단계는 batch 4를 우선 검증한다.

## 검증 및 산출물

- 수치 validation: 250 step
- checkpoint + fixed reference + 기존 8-panel: 2k, 5k, 10k
- validation reference, prompt, seed, timestep은 모든 checkpoint에서 고정
- 1x와 2x strength를 모두 생성

필수 지표:

- teacher/student final-effect cosine, normalized error, RMS ratio
- teacher-relative excess common-direction loss와 pairwise cosine
- artist variance 및 common-direction occupancy(진단 전용)
- base/correct/wrong MSE와 relative gain
- correct-minus-wrong gain
- single/pair/triple/amplified/signed별 성능
- 1/2/4/8 reference count별 일관성과 성능

10k의 최종 판단은 block cosine이 아니라 fixed/panel의 reference 재현,
`correct_gain > wrong_gain`, 작가 간 출력 다양성, mixture 강도 보존으로 한다.
