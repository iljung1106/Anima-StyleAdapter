# Anima Style Adapter data pipeline

Danbooru 메타데이터에서 작가·이미지 후보를 고르고, 실제 파일을 검증해 내려받은 뒤 near-duplicate를 제거하고 WD EVA02 Tagger 결과를 sharded Parquet으로 만드는 파이프라인이다.

상세 연구 계획은 [docs/anima_style_adapter_training_plan.md](docs/anima_style_adapter_training_plan.md)에 있다.

현재 1차 학습 데이터는 `ij/anima-style-embedding-500k-full-face`의 **human full 이미지**만 사용한다. `synthetic/`과 `.face.webp`는 다운로드·추출 대상에서 명시적으로 제외한다. 5,000개 style identity의 50장씩을 dedup 후보로 사용하고, 전역 exact duplicate와 동일 style identity 내부 near-duplicate를 제거한 뒤 작가당 30장을 본 학습 manifest에 넣는다.

## 설치

Python 3.10 이상을 사용한다. 데이터 준비만 할 때와 GPU 태깅/C-RADIO 캐시까지 할 때를 나누어 설치할 수 있다.

```bash
python -m venv .venv
.venv/bin/pip install -e .
# GPU 이미지에서 적절한 CUDA PyTorch를 먼저 설치한 다음:
.venv/bin/pip install -e ".[tagger,features]"
```

Windows PowerShell에서는 `.venv/bin` 대신 `.venv/Scripts`를 사용한다.

## 실행

소규모 데이터 준비 smoke run:

```bash
anima-data --config configs/smoke.yaml prepare
```

단계별 실행:

```bash
anima-data --config configs/anima500k-human.yaml anima500k-download
anima-data --config configs/anima500k-human.yaml anima500k-extract
anima-data --config configs/anima500k-human.yaml dedup
anima-data --config configs/production.yaml select
anima-data --config configs/production.yaml download
anima-data --config configs/production.yaml dedup
anima-data --config configs/production.yaml tag
anima-data --config configs/production.yaml caption
anima-data --config configs/production.yaml features
```

`all`은 여섯 단계를 연속 실행한다. `prepare`는 모델 추론 전의 select/download/dedup까지만 실행한다. 모델 캐시는 Tagger 약 1.28 GB에 더해 C-RADIO와 SigLIP2-g가 로컬 smoke 환경에서 약 6.1 GB를 사용하므로 persistent volume에 둔다.

## 대규모 실행 방식

- 원격 Parquet은 Hugging Face filesystem의 range read로 필요한 컬럼만 두 번 순회한다.
- RunPod volume에 메타데이터를 미리 둔 경우 `metadata.local_glob`을 설정하면 동일 코드가 로컬 shard를 읽는다.
- 1차 pass는 작가별 count와 날짜 범위만 보관한다.
- 2차 pass는 최신성 가중으로 고른 작가 pool에 대해서만 작가당 `per_artist_scan_cap`개의 bounded reservoir를 보관한다.
- 다운로드 파일은 Danbooru ID modulo 1000으로 directory sharding하며, MD5가 맞는 기존 파일은 재사용한다.
- 태깅 결과는 `tags/part-*.parquet`으로 나뉘고 기존 shard의 ID를 건너뛰므로 중단 후 재실행할 수 있다.
- Anima caption은 rating, 인물 수, 캐릭터/의상 캐릭터, 일반 태그 순으로 만들며 `content_caption`에는 rating을 제외한다.
- C-RADIO 입력은 큰 이미지만 축소한 뒤 원 비율에 가장 가까운 16 배수로 최소 중앙 crop한다. 해상도별 bucket batch를 사용한다.
- feature cache는 backbone spatial token과 SigLIP2-g image/text 및 정규화된 `image - content_scale * text` residual을 원래 차원으로 보존한다. 투영·결합은 학습 가능한 `build_style_feature_combiner`가 담당하므로 cache를 다시 만들지 않고 adapter 차원을 바꿀 수 있다.
- 태거 weight와 label CSV는 ephemeral home cache가 아니라 output volume의 `model_cache/`에 고정된다.

전체 작업에서는 약 6.2 GB의 metadata Parquet을 persistent volume에 먼저 내려받아 두 번의 scan을 로컬에서 수행하는 편이 빠르다.

```bash
hf download Shio-Koube/Danbooru-2026-parquet-metadata \
  --repo-type dataset --include "*.parquet" \
  --local-dir metadata/danbooru-2026
```

그 뒤 별도 production 설정에서 다음과 같이 바꾼다.

```yaml
metadata:
  local_glob: "metadata/danbooru-2026/*.parquet"
  include_files: []
```

## 주요 산출물

```text
data/production/
  artist_stats.parquet
  candidate_manifest.parquet
  download_manifest.parquet
  dedup_manifest.parquet
  final_manifest.parquet
  images/000..999/
  model_cache/
  tags/part-00000.parquet ...
  captions/part-00000.parquet ...
  cradio_model_cache/
  cradio_features/
    part-00000.safetensors ...
    manifests/part-00000.parquet ...
  *_summary.json
```

`final_manifest.parquet`은 작가별 최종 이미지 목록이다. 태거 출력은 raw dense vector 대신 설정한 score floor 이상의 `(tag, probability)`를 최대 `sparse_top_k`개 저장한다. 선택 threshold를 바꾸는 실험은 이 sparse cache의 floor 이상 범위에서 모델 재실행 없이 가능하다.

## C-RADIO feature 계약

- RADIO 코드 revision: `c0f37017930e9dda53f93424cf4bf39fc51f287e`
- backbone: `c-radio_v4-so400m`, adaptor: `siglip2-g`
- 입력: RGB `[0, 1]`, 높이·너비 모두 16의 배수
- 텐서 키: `{danbooru_id}.backbone_spatial`, `.backbone_summary`, `.siglip_visual`, `.siglip_text`, `.siglip_residual`
- manifest에는 원본/resize/crop 크기, token·feature 차원, caption hash와 모델 revision을 기록한다.

`configs/smoke.yaml`의 256 px CPU 설정은 API와 저장 계약을 확인하기 위한 것이다. 실제 캐시에서는 `configs/production.yaml`의 CUDA 설정과 더 큰 해상도를 사용한다.

## 태거 계약

구현은 canary Space의 실제 추론 코드를 기준으로 한다.

- architecture: `eva02_large_patch14_448`
- 흰색 정사각 padding 후 448×448 bicubic resize
- RGB에서 BGR로 변환 후 `[-1, 1]` 정규화
- model card에 명시된 recentered global cutoff `0.6025`
- model revision과 label CSV revision을 설정에 고정

## 운영 전 조정할 값

`configs/production.yaml`의 날짜 창, 최신성 감쇠, 후보 배수와 pHash/dHash 임계값은 첫 분포 집계 및 수동 duplicate audit 후 확정한다. 특히 near-duplicate 임계값은 임의의 일반값을 그대로 production 기준으로 사용하지 않는다.

## 테스트

```bash
pytest
```

테스트는 날짜 창·결정적 표집, perceptual hash, 태거 전처리 계약처럼 대량 실행 실패 위험이 큰 부분에 집중한다.
외부 StyleNet controlled layer benchmark의 protocol과 실행법은
[`docs/stylenet_layer_benchmark.md`](docs/stylenet_layer_benchmark.md)에 기록되어 있다.
