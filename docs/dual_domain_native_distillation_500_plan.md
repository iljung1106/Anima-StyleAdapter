# 500-artist dual-domain native-effect distillation

## 목적

10k 모델에서 96명 human reference로 centered Anima `@artist` flow effect를
회귀했을 때 정성적 스타일 재현과 artist-effect 분리가 개선됐다. 다음 실험은
작가 수를 500명으로 늘리고, human reference와 Anima synthetic reference를
서로 독립된 입력 domain으로 증류한다. 기존 Style Tokenizer 체크포인트는
사용하지 않고 compact tokenizer를 무작위 초기화해 2,000 step 학습한다.

## 데이터와 분할

- 작가 500명: human cache에서 이미지가 충분한 train 작가를 결정적으로 선택
- teacher 분할: train 450 / validation 25 / meta-test 25
- human reference: 기존 전체 Dual-query Resampler token cache
- synthetic reference: 작가당 8 content × 2 seed = 16장
- synthetic `@artist`에는 내부 `human:` style namespace를 넣지 않고 raw 작가명만 사용
- validation/meta-test 50명은 일반 flow train loader에서도 제외

## Native teacher bank

각 작가, 8개 공통 content, 8개 timestep에서 동일 latent·noise·seed를 사용한다.

\[
d_a = F(x_t,t,c+@a)-F(x_t,t,c),\qquad
\tau_a=d_a-\operatorname{mean}_b d_b
\]

Synthetic content-control의 512×512 latent를 probe로 재사용하며 teacher tensor는
FP16으로 한 번만 계산해 캐시한다.

## 독립 domain 증류

Human과 synthetic은 서로 비교하거나 같게 만드는 loss를 사용하지 않는다.
각 domain이 별도의 reference batch에서 같은 방법으로 detached teacher effect를
회귀한다.

\[
L_D=L_{Huber}((s_D-\tau)/RMS(\tau))
 +0.1(1-\cos(s_D,\tau))
 +0.05L_{Huber}(\log RMS(s_D),\log RMS(\tau))
\]

`D`는 human 또는 synthetic이다. 각 domain weight는 250 step 동안 0에서 0.05로
증가한다. Cross-domain token/residual consistency는 두지 않는다.

## 동시 flow 학습

매 optimizer step마다 다음 gradient를 합산한다.

1. 전체 human corpus target-excluded reference의 rectified-flow MSE:
   batch 4 × accumulation 4
2. human teacher: 4 artist × 4 references
3. synthetic teacher: 4 artist × 4 references

Reference 수는 1/2/3/4장을 50/30/15/5%로 사용한다. Frozen Anima와 frozen
Dual-query Resampler는 업데이트하지 않으며 약 9.48M compact Style Tokenizer만
학습한다.

## 최적화와 관찰

- fused AdamW, LR 1e-4, 100-step warmup, cosine decay, grad clip 1.0
- Synthetic 생성은 GPU DCT 기반 SPEED, GPU-resident text condition,
  batch 8/16/24/32 실측 자동 선택, pinned D2H double buffer를 사용
- Native teacher는 기존 synthetic text cache를 재사용하고 artist batch를
  8/16/24/32 중 실측 선택하며, centered effect를 GPU에서 계산
- 2 TiB host RAM에 Human/Synthetic reference-token shard를 한 번만 상주시켜
  NFS random-read 반복을 제거
- Human/Anima teacher loss는 독립적으로 계산하되 같은 shape의 forward만
  batch 8로 결합하여 H100 kernel occupancy를 높임
- 250 step마다 일반 validation과 human/synthetic teacher validation
- teacher validation은 고정 16개 content/timestep probe를 평균
- 250 step마다 checkpoint, 500 step마다 panel, 1,000 step마다 fixed-reference
- W&B에는 두 domain의 direct/direction/magnitude loss, cosine, projection,
  orthogonal ratio, student/teacher RMS를 분리 기록

성공 기준은 정성적 스타일 재현 증가와 함께 두 domain teacher cosine/projection이
상승하고, 일반 heldout flow와 same-artist reference 안정성이 붕괴하지 않는 것이다.
