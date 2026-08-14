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
추론 시 style CFG는 다음 차이를 사용한다.

`full(text + style tokens) - text_only(original cached context)`

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
