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

Phase A의 validation Pareto 최적 체크포인트를 `checkpoints/selected.pt`로
고정한 뒤 새 optimizer로 이어서 학습한다.

- target은 전체 train 이미지 풀에서 선택한다.
- reference는 같은 작가의 다른 그림 1–8장이고 target을 포함하지 않는다.
- frozen Resampler, frozen Anima, 16×1024 native-context 삽입 구조는 유지한다.
- AdamW LR `5e-5`, 200-step warmup, cosine decay, 최대 8,000 step을 사용한다.
- self/heldout뿐 아니라 같은 batch의 작가 reference를 순환시킨 wrong-artist
  조건을 함께 검증한다. `heldout paired improvement - wrong-artist paired
  improvement`가 양수여야 reference-specific style을 학습했다고 본다.
- 500 step마다 self와 heldout 생성 패널을 보며 target content 복사 감소,
  스타일 변화, 붕괴 여부를 함께 판단한다.

최종 체크포인트는 heldout paired improvement와 direction cosine이 높고,
correct-vs-wrong advantage가 양수이며, 정성 샘플에서 콘텐츠 누출이 낮은
Pareto 후보로 선택한다. 단일 noisy validation 지점만으로 고르지 않고 후보를
더 큰 고정 validation 표본으로 재평가한다.

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
