# C-RADIO tap / per-reference Resampler 실험

## 목적

Train 작가 중 1,000명에서 작가당 10장을 고정 표집하여, C-RADIO spatial tap과 SigLIP global 표현의 선택이 다음 두 성능에 미치는 영향을 비교한다.

- 16-token 병목에서의 고정 C-RADIO 특징 복원
- 학습에 보지 않은 작가의 1/2/4/8-shot prototype 분류

## 데이터와 통제 조건

- 작가 분할: meta-train 800명 / meta-validation 100명 / meta-test 100명
- 입력: 종횡비를 유지하되 `max_side=512`, `max_pixels=512²`, 각 변은 16의 배수
- 캐시: spatial block 8/12/16/20/24, 각 레이어 SigLIP CLS, 최종 SigLIP visual embedding
- 모든 변형은 동일한 512차원·16-token per-reference Resampler와 동일한 decoder를 사용한다.
- 공통 decoder 목표는 block 8/16/24 spatial 특징이다. 작가 loss는 공통 latent 전체가 아니라 pooled 256차원 projection에 적용한다.

## 비교 순서

1. Spatial only: `[24]`, `[12,24]`, `[20,24]`, `[8,16,24]`, `[12,20,24]`
2. 주요 후보에 block 24 native SigLIP CLS 추가
3. native CLS가 유효한지 확인하기 위해 최종 SigLIP visual embedding과 비교

선택 기준은 validation 작가의 prototype Top-1/MRR과 block별 reconstruction cosine의 Pareto 성능이다. 고정 학습 작가 classifier는 사용하지 않으며, 매 step 같은 작가의 이미지가 support/query 역할을 바꾸는 episodic prototype loss를 사용한다.

## 실행

```bash
HF_HOME=/workspace/.cache/huggingface \
  .venv/bin/anima-data --config configs/anima500k-human.yaml tap-experiment
```

특징 추출과 학습은 각각 `tap-extract`, `tap-train`으로 분리 실행할 수도 있다. 특징 shard와 학습 checkpoint는 중단 후 재개되며, 완료된 variant는 `force: false`일 때 자동으로 건너뛴다. 결과는 `data/anima500k-human/tap_resampler_experiment/evaluation.json`에 기록한다.

Validation 결과로 후보를 확정한 뒤 `selected_variant`를 설정하고, 보존한 meta-test 작가는 한 번만 평가한다.

```bash
.venv/bin/anima-data --config configs/anima500k-human.yaml tap-test
```

최종 결과는 `tap_resampler_experiment/final_test.json`에 기록한다.
