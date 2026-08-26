# 320-Teacher Direct Reference K/V Delta 학습 계획

## 목표와 추론 계약

Styled reference만으로 현재 문장의 블록별 native text K/V residual을 직접 생성한다.

```text
styled references + text context + block index
    -> predicted raw delta-K / delta-V [B, 2, 512, 2048]
    -> native K/V에 가산
    -> 기존 k_norm / v_norm / attention / O 경로
```

- 추론과 학습 입력에 base reference를 사용하지 않는다.
- `Reader(styled) - Reader(base)`와 같은 입력 차분을 사용하지 않는다.
- 별도 Common 분기, population mean subtraction, centered teacher target을 사용하지 않는다.
- 목표는 K/V-only LoRA가 만드는 **full raw teacher ΔK/ΔV**이다.

## 사용할 데이터

- 320개 K/V-only LoRA factor bank
  - `artist_kv_lora_teachers_rank16_320_b2_v3`
- 작가당 8장, 총 2,560장의 styled reference 및 frozen visual-token cache
  - `kv_lora_teacher_references_rank16_320x8_realq_v2`
- 256개 cached text context
- 기존 64-teacher 기반 pair/triple/amplified/signed mixture reference와 mixture 명세

64개 mixture 구성 작가는 반드시 training split에 포함한다. 나머지 작가에서 192개를 더해 총 256개를 학습하고, 사용되지 않은 64개 작가를 held-out artist 평가에 사용한다. Reference 이미지의 content와 teacher target을 계산할 text context는 항상 독립적으로 샘플링한다.

## 모델

1. Frozen Resampler visual-token cache는 그대로 사용한다.
2. Reader/style aggregator는 cached token을 받아 multi-reference style memory를 만든다.
3. Reader는 고정 materialization하지 않고 generator와 함께 end-to-end 미세조정한다.
4. 각 블록의 text token이 style memory를 조회해 raw ΔK/ΔV를 생성한다.
5. 블록별 text-context projection과 ΔK/ΔV output head는 유지한다.

Generator 앞의 sample-wise style `LayerNorm`은 제거한다. 안정화가 필요하면 training bank에서 한 번 계산한 고정 channel 통계로만 정규화하여 sample 간 RMS/강도 차이를 보존한다.

## Teacher와 손실

각 block에서 저장된 LoRA factor로 teacher를 즉시 계산한다.

```text
teacher_delta = (text_context @ down[artist, block]) @ up[artist, block]
```

Mixture target은 component별 full delta의 정확한 가중합이다. 주 손실은 다음으로 구성한다.

- normalized Huber/MSE: raw ΔK/ΔV 값 정합
- cosine loss: K와 V의 방향 정합
- RMS-ratio loss: teacher 대비 residual 강도 정합
- native-attention loss: 동일 native Q에서 `k_norm/v_norm/attention/O` 결과 정합
- reference consistency: 같은 작가의 서로 다른 1/2/4/8-reference view가 같은 target을 예측하도록 정합

Population-centered loss와 Common 억제 loss는 사용하지 않는다. 다만 공통방향 붕괴 여부는 작가 간 예측 분산과 wrong-artist 지표로 계속 관찰한다.

## 학습 구성

- Batch 구성: 320-teacher single 50%, 기존 mixture 50%
- Mixture 내부: pair/triple/amplified/signed를 균등 샘플링
- Reference 수: 1/2/4/8장을 혼합하고 일부 reference dropout 적용
- Block: 매 step 4개 block을 순환 샘플링
- Generator LR을 기준으로 Reader LR은 `0.2배` 사용
- Raw ΔK/ΔV 손실은 처음부터 적용하고 native-attention loss만 초반 500 step 동안 ramp
- 100-step smoke에서 unclipped gradient norm 분포를 기록한 뒤 99~99.5 percentile 이상의 이상치만 clipping
- 첫 본 학습은 4,000 step, 250 step마다 수치 검증, 500 step마다 실제 생성 panel 저장

## 검증과 통과 기준

Train artist와 64개 held-out artist를 분리해 다음을 single 및 mixture 종류별로 기록한다.

- raw ΔK/ΔV cosine, normalized error, student/teacher RMS ratio
- native-attention output cosine/error
- correct-reference 대 wrong-reference margin
- 1/2/4/8 reference count별 성능과 예측 일관성
- 작가 간 predicted residual 분산 및 공통방향 점유율
- 고정 prompt/seed의 val/functional/panel과 1×/2× strength sample

100-step smoke에서 tensor shape, loss 감소, Reader gradient, mixture 강도 보존을 확인한 뒤 4,000-step 본 학습을 시작한다. 500-step 간격으로 held-out K/V 지표와 실제 생성 품질이 함께 개선되지 않으면 장기 학습을 계속하지 않고 Reader 병목과 reference identifiability를 먼저 재검토한다.
