# Query StyleTokenizer v2 joint training

## 목적

기존 StyleTokenizer의 `128 tokens -> 1 descriptor -> 16 tokens` 병목을
제거한다. 새 모델은 각 레퍼런스에서 32개 query가 128개 Resampler token
전체를 cross-attention하고, 마지막까지 `32 x 1024` 표현을 유지한다. 이
표현은 사전학습으로 따로 고정하지 않고 frozen Anima의 flow loss와 함께
처음부터 학습한다.

## 구조

1. 공유 per-reference decoder 3층: 32 learned queries가 `128 x 1024`
   Resampler token을 읽는다.
2. 동일한 slot끼리 레퍼런스 축 attention pooling을 수행한다. 따라서
   레퍼런스 순서에는 불변이고 slot 의미는 유지된다.
3. cross-slot Transformer 2층이 32개 slot 사이의 관계를 정리한다.
4. 출력 `32 x 1024`는 별도 legacy bridge 없이 Anima context로 직접
   들어간다.
5. Anima의 동일한 Q로 text/style attention을 따로 계산하고 합친 뒤,
   frozen native full-rank O를 한 번 사용한다. Style K/V는 native K/V
   복사본과 trainable rank-32 delta로 구성한다.

Tokenizer는 약 76.0M parameter이며, 추론에서 쓰지 않는 1층 재구성
decoder를 포함한다. Frozen Anima와 기존 128-token Resampler cache를
사용한다.

## Loss

주 loss는 rectified-flow velocity MSE다. 보조 loss는 다음처럼 제한한다.

- artist contrastive: target과 같은 작가의 target-excluded reference에서
  나온 aligned slot은 가깝게, 배치의 다른 작가는 멀게 한다. 가중치
  `0.01`, 2 update마다 적용한다.
- per-reference reconstruction: aggregation 전 32 slot에서 선택한 한
  reference의 정규화된 128 Resampler token을 복원한다. 가중치는
  `0.02 -> 0.005`로 8k step 동안 감소한다. 정보 보존을 돕는 약한
  regularizer이며 최종 aggregate를 이미지별 세부 정보로 오염시키지 않는다.
- slot diversity: 서로 다른 slot의 cosine 중복을 약하게 억제한다.
  가중치 `0.001`이다.

별도 null token은 쓰지 않는다. Style branch를 제거하면 frozen Anima의
정확한 base path가 되기 때문에, style CFG의 unconditional branch는
adapter context를 비우는 방식으로 정의한다.

## 20k curriculum

- 0–2k: reference 1장 = exact target, 전체 train image pool에서 교대한다.
- 2k–8k: reference 1–4장, target 포함률 `1.0 -> 0.5`.
- 8k–20k: reference 1–8장, target 포함률 `0.5 -> 0.0`.

K/V delta와 tokenizer는 첫 step부터 같이 학습한다. Style/text branch RMS를
실제 Anima block에서 측정해 초기 비율을 0.25로 보정하고 alpha는 이후
동결한다. 따라서 alpha를 0으로 줄이는 shortcut은 사용할 수 없다.

Validation은 250 step마다 validation artist에 대해 exact-self,
target-excluded heldout, wrong-artist를 같은 noise/timestep으로 비교한다.
Checkpoint는 500 step마다 저장한다.

Checkpoint 직후에는 고정된 train 작가 4명과 validation 작가 4명을
`768 x 768`, 30-step으로 batch 생성한다. 각 sheet는 동일한 target prompt,
초기 noise, target-excluded heldout reference를 유지하면서 frozen Anima
base와 StyleTokenizer 출력을 나란히 표시한다. Text CFG는 4, 별도로 분리된
style CFG는 1이며 W&B에도 8개 panel을 업로드한다.
