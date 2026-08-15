# Dual-query Style Tokenizer 학습 계획

## 목적

사전학습한 Dual-query Resampler를 이미지별 고정 encoder로 사용하고, 여러 reference에서 공통 스타일을 추출하는 작은 Set Transformer를 학습한다. 최종 출력은 `32×1024` 토큰이며 Anima의 post-LLM `512×1024` context에서 실제 텍스트 뒤의 빈 위치에 삽입한다. 별도 style K/V branch는 이 순수 토큰 경로가 검증된 뒤에만 고려한다.

## 고정 입력과 캐시

선택한 Resampler checkpoint는 이미지마다 `80×1024` query token과 `4×1024` artist-summary token을 만든다. 선택적으로 `512`차원 descriptor도 평가용으로 저장한다. Reconstruction decoder는 cache 생성과 추론에 사용하지 않는다.

전체 데이터 캐시는 BF16 contiguous shard로 저장한다. shard는 512~1024장을 묶고, image ID와 artist ID를 별도 index에 기록한다. 학습 loader는 shard 단위 RAM cache, background prefetch, reusable pinned buffer와 비동기 H2D copy를 사용한다. Frozen Resampler는 Context bootstrap의 optimizer와 checkpoint에 포함하지 않는다.

## Style Tokenizer

- 각 reference: `80×1024`, 또는 artist-summary를 포함한 `84×1024`
- 입력 reference: 1~8장, 순서 embedding 없음
- 32개 learned output query가 전체 reference set을 1층 cross-attention으로 읽음
- 2층 cross-slot Transformer로 slot 사이 정보를 교환
- 최종 출력: `32×1024`
- no-style은 토큰을 주입하지 않은 Frozen Anima 원본 경로
- style token은 positive text context에 함께 들어가므로 text와 동일한 CFG를 받음

## Artist-summary A/B

동일한 Resampler checkpoint, tokenizer 초기화, episode 순서와 optimizer 설정으로 다음 두 조건을 짧게 비교한다.

1. query-only: reference별 80개 query token만 사용
2. query+summary: 80개 query와 4개 artist-summary token을 함께 사용

held-out paired-flow improvement, correct-vs-wrong 차이와 고정 prompt/seed 샘플을 기준으로 한 조건을 선택한다. Style Tokenizer의 output-token artist contrastive loss는 두 조건에서 동일하며 summary 전달 여부만 바꾼다. 선택된 조건은 본학습에서 1/2/4/8-reference 지표와 common-output 계열 지표를 계속 추적한다.

## Anima 학습 커리큘럼

### 0~2,000 step: exact-self Context bootstrap

- reference 한 장은 항상 target
- Resampler와 Anima는 동결, Style Tokenizer만 학습
- rectified-flow MSE가 주 loss
- 500 step마다 고정 train 작가 4명과 held-out 작가 4명의 이미지를 batch 생성
- 1,000 step마다 고정 외부 reference 샘플 sheet 생성

### 2,000~8,000 step: 같은 작가 multi-reference

- reference 1~4장
- target 포함 확률 `1.0 → 0.5`
- 적은 reference 수를 더 자주 선택: 기본 확률 `1/2/3/4장 = 0.45/0.30/0.15/0.10`
- target이 포함될 때만 exact-self 보조항을 적용하고 포함률과 함께 감소
- 같은 작가의 서로 다른 reference subset에 약한 consistency loss 적용

### 8,000~20,000 step: target-excluded style transfer

- reference 1~8장
- target 포함 확률 `0.5 → 0`
- 기본 reference 수 확률 `1~8장 = 0.30/0.24/0.16/0.10/0.07/0.05/0.04/0.04`
- correct-reference 개선이 확인된 뒤에만 wrong-reference ranking을 점진적으로 도입
- 마지막 구간은 target과 겹치지 않는 reference만 사용

### 후반 joint alignment

순수 토큰 경로가 생성 샘플에서 작동한 뒤 Resampler 상위 1~2 block만 작은 LR로 연다. 이때 캐시 경로와 online feature 경로를 혼합하고 semantic/VAE reconstruction 및 artist loss는 사전학습 가중치의 `1/5~1/10`만 유지한다.

## 선택 및 기록

Resampler는 마지막 checkpoint를 자동 채택하지 않고 held-out artist 분류, positive-hard-negative gap과 semantic/VAE reconstruction의 Pareto로 선택한다. Style Tokenizer는 validation flow loss 하나가 아니라 paired improvement, correct/wrong/null reference, target 포함/제외, 1/2/4/8-reference consistency, token RMS/variance와 정성 샘플을 함께 기록한다. Checkpoint는 500 step마다, validation은 250 step마다 저장·수행하며 W&B run은 중단 후 동일 ID로 resume한다.
