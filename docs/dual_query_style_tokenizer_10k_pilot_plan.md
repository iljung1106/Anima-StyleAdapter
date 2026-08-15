# Dual-query Style Tokenizer 10k Pilot 계획

## 목적

Artist-summary token을 사용하는 Dual-query Style Tokenizer를 전체 train split에서
10,000 optimizer step 학습한다. Frozen Anima의 기본 생성 품질을 유지하면서
올바른 reference가 만드는 작가별 velocity 방향과 충분한 효과 크기를 학습하고,
작가와 무관한 공통 출력 및 직교 방향의 큰 변형을 억제하는 것이 목표다.

## 고정 구조

- Per-reference encoder: 학습 완료된 Dual-query Resampler, 동결
- Reference 표현: `80×1024` query + `4×1024` artist-summary
- Set Style Tokenizer: artist-summary **ON**, 출력 `32×1024`
- 주입: Anima post-LLM `512×1024` context의 실제 text 뒤 빈 위치
- Frozen Anima의 native Q/K/V/O와 shared text/style CFG 사용
- 학습 대상: Set Style Tokenizer만

## 10k 커리큘럼

| Step | Reference | 목적 |
|---:|---|---|
| 0–500 | exact target 1장 | native context 정렬과 초기 방향 형성 |
| 500–1,000 | target-excluded 같은 작가 1–2장 | exact-self shortcut 제거 |
| 1,000–4,000 | target-excluded 같은 작가 1–4장 | correct-vs-wrong ranking 학습 |
| 4,000–10,000 | target-excluded 같은 작가 1–8장 | 실제 multi-reference 일반화 |

Reference 수는 1장을 가장 자주, 2장을 그다음으로 뽑는다. 3장 이상은
aggregation 능력을 유지할 만큼만 노출한다. Step 500 이후 target 이미지는
reference에 포함하지 않는다.

## Loss

주 loss는 Frozen Anima rectified-flow MSE다.

### 항상 유지

- flow MSE: weight `1.0`
- artist token contrastive: weight `0.01`, 2 step마다 계산
- slot diversity: weight `0.001`
- subset consistency: weight `0.002–0.005`
- normalized residual Huber:
  - exact-self `0.05`
  - target-excluded `0.01–0.02`

### Target-aligned coefficient floor

단순 출력 RMS가 아니라 `delta = v_style - v_base`가
`residual = v_target - v_base` 방향으로 갖는 투영 계수에 하한을 둔다.

- step 0–500: `c_min 0.02 → 0.15`, weight `0.20–0.25`
- step 500–10,000: `c_min 0.03 → 0.06`, weight `0.05–0.10`
- 학습 종료까지 유지하되 target-excluded 구간에는 더 낮은 하한을 사용한다.

### Correct-vs-wrong cyclic flow ranking

- step 1,000부터 활성화
- 배치 안의 다른 작가 reference를 cyclic donor로 사용
- wrong prediction은 detach하여 wrong 출력을 고의로 망치는 shortcut 차단
- weight `0 → 0.00075–0.001`, 1,000 step 동안 ramp
- 4 step마다 batch row 2개에서 계산
- correct/wrong direction ranking과 약한 centered-direction 항을 함께 사용

### Bounded aligned-effect 보조항

원본 대비 raw 출력 변형량을 직접 보상하지 않는다. 4 step마다 target 방향의
양의 투영량에 하한과 상한을 두고, 직교 성분이 커지면 패널티를 준다.
Target-excluded 구간의 초기 범위는 `r_min 0.04–0.06`, `r_max 0.20`으로 둔다.

### Common-output penalty

- step 0–1,000: 측정만 수행
- step 1,500부터 활성화
- 8 step마다 작은 controlled microbatch에서 velocity delta의 공통 성분 계산
- `ReLU(common_output_ratio - threshold)^2` 형태의 hinge loss
- weight `0.001–0.005`
- denominator는 detach하고, flow alignment/ranking과 함께 사용하여 작가별
  무작위 노이즈를 만드는 shortcut을 억제한다.

Controlled probe에 동일 noisy latent·prompt·timestep을 사용하는 것은 내부
지표/loss 계산에만 한정한다. 주기적 정성 샘플의 작가별 target 프롬프트와
시드는 기존 방식대로 유지한다.

## 학습과 기록

- optimizer: AdamW, peak LR `1e-4`, warmup 250–500 step, cosine decay,
  minimum LR `1e-5`
- batch: 4, gradient accumulation 4
- validation: 250 step마다
- checkpoint 및 train 4명/validation 4명 패널: 500 step마다
- 고정 외부 reference sheet: 1,000 step마다
- 1/2/4/8-reference 평가: 1,000 step마다
- W&B에 각 raw loss, weighted loss, gradient norm, LR과 curriculum 상태 기록

필수 지표:

- self/heldout/wrong paired-flow improvement와 positive fraction
- correct-vs-wrong paired advantage
- aligned coefficient와 floor violation
- direction cosine, delta/desired RMS, orthogonal/desired RMS
- style-output ratio와 style-token RMS
- common-output ratio
- within/between artist centered cosine와 artist retrieval
- reference subset consistency

## 선택 및 중단 기준

다음 조건을 함께 만족하는 checkpoint를 선택한다.

1. heldout paired improvement가 양수이고 여러 validation에서 유지된다.
2. correct-vs-wrong advantage가 양수이며 증가한다.
3. 출력 효과가 지나치게 작아지지 않고 orthogonal ratio가 폭증하지 않는다.
4. common-output ratio가 감소하거나 허용 범위 안에 머문다.
5. 1-reference 성능을 유지하면서 multi-reference가 품질을 개선한다.
6. 정성 샘플에서 content 보존, 작가별 차이, 이미지 안정성이 함께 확인된다.

연속된 validation에서 heldout 성능과 correct-vs-wrong advantage가 함께
악화되거나, common/orthogonal 성분 증가와 생성 붕괴가 나타나면 중단하고
마지막 안정 checkpoint로 되돌린다. 이 10k run은 최종 모델 확정이 아니라
flow-level 작가 구분과 비붕괴 학습 가능성을 검증하는 pilot으로 취급한다.
