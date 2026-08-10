# Anima 본학습 입력 캐시

본 Style Adapter 학습에서는 Anima, Qwen3 text encoder, Anima LLM Adapter와 Qwen-Image VAE를 동결한다. 매 step에서 이 모델들을 다시 실행하지 않도록 다음 두 cache를 만든다.

## Text conditioning

Anima의 실제 text 경로는 `Qwen3-0.6B hidden state → 6-layer LLM Adapter → 1,024-d cross-attention condition`이다. T5 encoder를 쓰는 것이 아니라 T5 tokenizer ID를 LLM Adapter의 target query로 사용한다. 따라서 동결된 LLM Adapter까지 통과한 최종 condition을 저장한다.

- 이미지당 `full`, `general_dropout` 두 caption variant
- rating, count, character tag는 dropout variant에서도 보존
- general tag 15%를 image ID와 seed로 결정적으로 제거
- 유효 T5 token까지만 FP16으로 이어 붙이는 packed storage
- 512 caption items당 한 safetensors shard
- caption dropout과 Style/Text CFG용 null condition 별도 저장
- Anima model commit, sd-scripts commit, tokenizer 길이와 variant 설정을 cache signature에 포함

각 shard의 `conditioning`은 `[sum(sequence_length), 1024]`, `offsets`는 각 item의 시작 위치다. Manifest에는 `(image id, variant, shard, token offset, token length)`가 기록된다. 이미지별 NPZ를 만들지 않으므로 NFS metadata I/O와 max-length padding 저장 낭비를 피한다.

## VAE latent

공식 Qwen-Image weight를 sd-scripts의 image-only 2D VAE 구현으로 변환하여 사용한다. Posterior mean과 공식 latent mean/std normalization을 적용한 16-channel FP16 latent를 저장한다.

- 원본 비율을 유지하며 1MP와 long side 1,536 이하로만 축소
- 64px bucket 정렬 후 center crop
- 일반 이미지는 확대하지 않음
- 짧은 변이 256px 미만인 12장만 최소 bucket까지 확대
- manifest width/height를 사용해 decode 전에 shape 정렬
- NFS reader와 Pillow decoder 분리
- 재사용 pinned host buffer 두 개와 단일 GPU queue
- GPU 계산과 다음 이미지 read/decode, shard write를 중첩
- 동일 latent shape 512장 단위 safetensors shard

## 실행

`sd-scripts`는 config에 기록된 commit으로 checkout해야 한다.

```bash
cd /workspace
git clone https://github.com/kohya-ss/sd-scripts.git
git -C /workspace/sd-scripts checkout 37a1cbbc5725ed2a3575506e7bd2001c9908ac92

cd /workspace/Anima-StyleAdapter
HF_HOME=/workspace/.cache/huggingface \
  .venv/bin/anima-data --config configs/anima500k-human.yaml anima-text-cache

HF_HOME=/workspace/.cache/huggingface \
  .venv/bin/anima-data --config configs/anima500k-human.yaml anima-latent-cache

.venv/bin/anima-data --config configs/anima500k-human.yaml anima-cache-validate
```

`anima-cache`는 두 단계를 순차 실행한다. 각 단계는 `manifests/part-*.parquet`을 기준으로 완료 shard를 재사용하므로 중단 후 같은 명령으로 resume할 수 있다. 기존 cache와 model/config signature가 다르면 덮어쓰지 않고 오류를 낸다.

H100 SXM 실측에서 text batch 64는 16.64 items/s였고 128로 키워도 16.58 items/s로 개선되지 않았다. VAE batch 8은 29.08 images/s였고 16은 28.89 images/s였다. 따라서 production 기본값은 text 64, VAE 8로 고정한다.

VAE activation memory는 해상도에 따라 증가한다. 최대 batch 8에 더해 batch당 target pixel budget을 4,194,304로 제한하여 512²에서는 8, 704×768에서는 7, 1024²에서는 4로 자동 조절한다. 이 제한은 H100 80GB에서 후반 고해상도 bucket의 OOM을 막기 위한 것이다.

## 본학습 loader 계약

- Text: 선택한 variant row의 `token_offset:token_offset+token_length`를 읽고 batch 최대 길이까지만 padding한다.
- Caption dropout: `null_conditioning.safetensors`의 `caption_dropout_null`로 교체한다.
- Target: aspect bucket이 같은 latent row를 묶는다.
- Reference: target ID와 다른 1~8개 image ID의 기존 C-RADIO cache를 읽고 동결 Resampler에 전달한다.

공식 계약 근거는 [sd-scripts Anima training guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md), [`strategy_anima.py`](https://github.com/kohya-ss/sd-scripts/blob/main/library/strategy_anima.py), [`anima_models.py`](https://github.com/kohya-ss/sd-scripts/blob/main/library/anima_models.py)다.
