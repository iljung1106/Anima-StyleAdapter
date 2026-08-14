# Anima native-context StyleTokenizer 실험

## 목적

현재 `128×1024` Per-reference Resampler 표현을 유지하면서, 별도의 style
attention과 학습 K/V 없이 Anima가 이미 사용하는 post-LLM context 공간에
스타일을 정렬할 수 있는지 검증한다.

## 구조

1. 현재 Resampler의 캐시된 `128×1024` 토큰을 입력한다.
2. slot attention pooling으로 reference별 1024차원 표현을 만든다.
3. reference attention pooling으로 여러 reference를 순서 불변으로 합친다.
4. `1024 → 512 → 16×1024` StyleTokenizer가 compact style token을 만든다.
5. 각 캡션의 실제 `token_length` 직후 zero-padding 16자리를 이 토큰으로
   대체한다. 전체 context 길이는 `512×1024`로 유지한다.
6. frozen Anima의 원래 28개 block K/V/O와 단일 text/style softmax를 그대로
   사용한다.

학습 파라미터는 약 9.48M이며 Anima, Resampler, Anima K/V/O는 동결한다.

## Null style과 CFG

learned null token과 style dropout은 사용하지 않는다. StyleTokenizer를
삽입하지 않은 원래 캐시 context가 frozen Anima의 정확한 no-style 조건이다.
스타일 토큰은 positive context의 일부이므로 기본 추론과 정성 샘플은 표준
shared CFG를 쓴다.

`velocity = null + CFG * (full(text + style) - null)`

따라서 기본 패널에는 별도 style CFG를 표시하지 않는다. 별도 style 강도를
실험할 때만 `guidance_mode: separate`를 명시하고 text/style delta를 각각
스케일한다. 이 선택적 3-forward 방식의 style delta는
`full(text + style) - text_only`이다.

## 첫 실험

- 출력: 16개 토큰, 각 1024차원
- 데이터: train 작가마다 대표 target 한 장을 순환하는 exact-self
- batch 4, gradient accumulation 4, 4,000 optimizer steps
- AdamW, LR `1e-4`, 200-step warmup, cosine decay
- loss: 표준 rectified-flow MSE만 사용
- validation: train exact-self, artist-disjoint validation exact-self,
  artist-disjoint same-artist heldout reference
- 250 step마다 검증, 500 step마다 768×768/30-step 정성 샘플

첫 실험에서 paired flow improvement와 reference 의존성이 확인된 뒤에만
target-excluded multi-reference curriculum이나 Resampler 상단 공동학습을 연다.

## Phase B: target-excluded 일반화

현재 production run은 Phase A 체크포인트를 재사용하지 않고 fresh
initialization에서 시작한다.

- 전체 train 이미지 풀을 사용하고 target 노출은 작가별로 거의 균등하게
  샘플링한다.
- reference는 같은 작가의 다른 그림 1–8장이고 target을 포함하지 않는다.
- frozen Resampler, frozen Anima, 16×1024 native-context 삽입 구조는 유지한다.
- AdamW LR `1e-4`, 200-step warmup, cosine decay, 8,000 step을 사용한다.
- frozen-Anima rectified-flow MSE가 주 손실이다.
- 같은 작가의 비중첩 reference 집합 두 개에서 얻은 전체 16개 slot을 직접
  supervised contrastive로 정렬한다. 다른 batch 작가는 negative이며 이
  손실은 첫 1,000 step 동안 weight `0.00075`까지 ramp한다.
- step 1,500부터 같은 target, prompt, noise, timestep에서 correct와 wrong
  artist의 flow 방향을 비교하는 bounded auxiliary loss를 weight `0.00075`까지
  ramp한다. 두 이미지에 대해 4 optimizer step마다 한 번 계산한다.
- wrong-artist 예측은 이 손실에서 stop-gradient한다. 따라서 wrong reference를
  의도적으로 파괴적인 방향으로 보내는 shortcut은 허용하지 않는다.
- 공통 flow 성분의 centering은 이 보조 비교 안에서만 사용하며 실제 style
  residual에서는 제거하지 않는다.
- 2-step real-cache gradient 진단에서 두 최대 보조 weight는 각각 flow
  gradient norm의 약 9%였다. direction은 4 step마다만 계산하므로 평균 영향은
  약 2--3%이며, flow MSE가 계속 최적화를 지배한다.
- 500 step마다 self와 heldout 생성 패널을 보며 target content 복사 감소,
  스타일 변화, 붕괴 여부를 함께 판단한다.

최종 체크포인트는 heldout paired improvement와 direction cosine이 높고,
correct-vs-wrong advantage가 양수이며, 정성 샘플에서 콘텐츠 누출이 낮은
Pareto 후보로 선택한다. 단일 noisy validation 지점만으로 고르지 않고 후보를
더 큰 고정 validation 표본으로 재평가한다.

완료 후 `style-tokenizer-select`는 마지막 체크포인트를 반드시 포함한
step 1,500--8,000의 상위 8개 후보를 64개 고정 validation batch로
재평가한다. 선택 점수는
`heldout LCB95 + 0.25*self LCB95 + 0.5*(heldout-wrong)`이며, 선택된 후보는
reference 1/2/4/8장 각각에 대해 추가로 heldout/wrong 검증한다.
`style-tokenizer-export`는 선택 checkpoint에서 optimizer를 제거한
`style_tokenizer.safetensors`, SHA-256·모델/입출력 계약·검증 지표를 담은
`manifest.json`, 사용 설명을 담은 `README.md`를 생성하고 strict round-trip
load를 확인한다.

## Phase B production 결과 (2026-08-15)

Fresh initialization, peak LR `1e-4` 설정으로 전체 8,000 optimizer step을
완료했다. step 3,930에서 496 token보다 긴 caption이 16개 style slot을 남기지
않아 중단된 문제는, 고정 512-token context의 마지막 16자리를 항상 style
token에 예약하도록 수정한 뒤 step 3,750 state에서 재개했다. 수정 후 같은
step을 통과했으며 최종 summary에는 정확히 8,000 step이 기록됐다. 전체
resume 로그에는 Traceback, OOM, NaN이 없고 token RMS는 마지막까지 약
`0.1486`으로 유지됐다.

빠른 250-step validation으로 뽑은 8개 후보를 64개 고정 batch로 다시 평가한
결과는 다음과 같다. 개선율은 frozen Anima의 flow MSE 대비 상대 개선율이며
퍼센트로 표시한다.

| step | self 개선 | heldout 개선 | wrong-artist 개선 | correct-wrong 우위 | 선택 점수 |
|---:|---:|---:|---:|---:|---:|
| 5,000 | +0.359% | +0.429% | +0.064% | +0.365%p | 0.005490 |
| 5,250 | +0.370% | +0.446% | +0.038% | +0.408%p | 0.005775 |
| 6,500 | +0.418% | +0.453% | +0.142% | +0.311%p | 0.005721 |
| 7,000 | +0.389% | +0.441% | +0.070% | +0.372%p | 0.005733 |
| 7,250 | +0.397% | +0.477% | +0.065% | +0.412%p | 0.006275 |
| 7,500 | +0.410% | +0.468% | +0.079% | +0.389%p | 0.006198 |
| **7,750** | **+0.401%** | **+0.518%** | **+0.081%** | **+0.436%p** | **0.006750** |
| 8,000 | +0.359% | +0.516% | +0.094% | +0.422%p | 0.006540 |

선택 규칙에 따라 step 7,750을 최종 모델로 선택했다. 이 후보의 self와
heldout positive fraction은 각각 74.2%, 70.3%였고 wrong-artist는 54.3%였다.
95% CI half-width는 self 0.117%p, heldout 0.132%p, wrong-artist 0.168%p다.
즉 절대 flow 개선은 작지만, 더 큰 고정 표본에서도 올바른 reference가 잘못된
작가보다 일관되게 유리하다.

선택 모델의 reference 수별 32-batch 검증은 다음과 같다.

| references | heldout 개선 | wrong-artist 개선 | correct-wrong 우위 | heldout positive fraction |
|---:|---:|---:|---:|---:|
| 1 | +0.175% | -0.073% | +0.248%p | 68.0% |
| 2 | +0.495% | +0.104% | +0.391%p | 75.0% |
| 4 | +0.486% | +0.133% | +0.353%p | 72.7% |
| 8 | +0.554% | +0.255% | +0.298%p | 72.7% |

독립 batch 추정이라 2/4/8 사이의 작은 순위 차이는 유의하다고 볼 수 없지만,
multi-reference가 single-reference보다 heldout 개선을 크게 높이는 방향은
확인된다. 작가 구분 우위는 2-reference가 가장 높고, 절대 heldout 개선은
8-reference가 가장 높았다.

step 7,750의 train/validation self/heldout 4개 768x768, 30-step 패널을
확인했다. 순수 노이즈, 공통 출력, 색면 붕괴는 없었고 모든 조건에서 선화,
채색, 명암, 형태가 reference에 따라 실제로 바뀌었다. heldout 패널은 인물,
의상, 배경 구성을 대체로 유지했으며 self 패널은 예상대로 내용 복사 성향이
더 강했다. 따라서 이 모델은 다음 Anima adapter 학습의 style-token source로
사용할 수 있지만, StyleTokenizer 단독 출력만으로 완전한 화풍 재현이 끝난
것으로 해석하지 않는다.

배포 산출물은 `style_tokenizer.safetensors` 37.9 MB와 입출력 계약 및 선택
지표를 담은 `manifest.json`, `README.md`다. strict state-dict round-trip과
관련 테스트 10개를 통과했다. weight SHA-256은
`0ce346f6225bec0e8773697ae14c8bd0e87bda44ff2cee60dc24e4608c937888`이다.
null style은 learned token이 아니라 style token을 삽입하지 않은 원래 frozen
Anima context이므로 정확한 base path를 유지한다.

## LR 10배 분기 실험

Phase A의 step 1,500 체크포인트를 별도 출력 디렉터리의
`training_state.pt`로 복사하여 optimizer moments, RNG, data position을 모두
보존한다. 원래 Phase A 디렉터리는 변경하지 않는다. 분기 run은
`LR=1e-3`을 고정해 step 1,501–2,000만 실행하며 50 step마다 검증하고 250
step마다 같은 정성 패널을 만든다. heldout improvement와 direction cosine이
빠르게 상승하지 않거나 wrong-artist와 구분되지 않거나 샘플이 붕괴하면 높은
LR을 채택하지 않고 원래 Phase A를 재개한다.

이 분기는 fine-tuning 안정성 실험이며 LR 학습 속도의 공정 비교는 아니다.
별도의 scratch 분기는 원래 Phase A와 동일한 seed, 초기화, 데이터 순서,
200-step warmup, cosine decay, 총 4,000 step을 사용하고 peak/min LR만 정확히
10배인 `1e-3/1e-4`로 둔다. 두 run은 동일 step의 고정 validation과 500-step
정성 패널로 비교한다.
