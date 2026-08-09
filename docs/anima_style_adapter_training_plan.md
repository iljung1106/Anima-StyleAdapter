# Anima Style Adapter: 1차 데이터·표현 학습 계획

- 문서 상태: 사전 계획 / 실험 전
- 작성일: 2026-08-09
- 1차 목표 규모: 작가 5,000명 × 작가당 최종 이미지 30장 = 150,000장

## 1. 목표와 범위

Danbooru 메타데이터에서 충분한 이미지가 있는 작가를 선별하고, 업로드 시점이 비교적 가까운 이미지 30장을 작가별로 구성한다. 이미지에는 WD EVA02 Tagger를 적용하고, C-RADIOv4-SO400M의 공간 특징과 SigLIP2-g adaptor에서 얻은 content-subtracted 특징을 입력으로 삼아 Style Resampler를 학습한다.

Resampler는 하나의 공통 경로에서 다음 두 목적을 동시에 학습한다.

1. 같은 작가의 이미지가 유사한 스타일 공간에 놓이도록 하는 작가 비교대조·프로토타입 학습
2. 입력 이미지의 유용한 시각 특징과 구조가 잠재표현에 남도록 하는 encoder-decoder 재구성 학습

재구성 decoder는 표현 학습에만 사용하며, 실제 Style Transfer 추론에서는 제거한다. Resampler 표현이 검증된 뒤 Anima의 28개 block에 연결할 Style Adapter K/V 구조와 전체 크기를 확정한다.

## 2. 현재 확정한 사항

| 항목 | 결정 |
|---|---|
| 메타데이터 | `Shio-Koube/Danbooru-2026-parquet-metadata` |
| 1차 데이터 규모 | 5,000 artists × 30 images = 150,000 images |
| 시간 정보 | 원본 publication 시점이 아니라 Danbooru의 `created_at`을 사용 |
| 작가별 시간 제약 | 한 작가의 표본이 일정한 업로드 날짜 범위 안에 있도록 제한 |
| 최신성 | 최근 작가·최근 이미지에 더 높은 표집 가중치 부여 |
| 중복 제거 | 정확 중복 및 거의 중복 이미지만 빠르게 제거 |
| 스타일 필터링 | 이미지 임베딩에 의한 스타일 유사도 검사·군집화는 1차 학습에서 하지 않음 |
| 태거 | `ashen-sensored/wd-eva02-tagger-2026-canary` |
| 비전 인코더 | `nvidia/C-RADIOv4-SO400M` |
| 시각 입력 | 16의 배수인 비정사각 해상도와 다양한 종횡비 사용 |
| 특징 입력 | SigLIP2-g content-subtracted 특징 + C-RADIO backbone 공간 특징 |
| 레이어 선택 | 실제 데이터 수집 뒤 probe 실험으로 결정 |
| 표현 학습 | 단일 Resampler 경로에 작가 임베딩 loss와 재구성 loss를 함께 적용 |
| 추론 | 재구성 decoder는 사용하지 않음 |
| 작가 split | 기본안 95% train / 2.5% validation / 2.5% test |
| 학습 작가의 이미지 역할 | 고정 reference/target이 아니라 step마다 동적으로 교대 |

## 3. 단계별 파이프라인

```mermaid
flowchart LR
    A["Danbooru Parquet metadata"] --> B["Metadata filtering"]
    B --> C["Artist/date-window sampling"]
    C --> D["Candidate image download"]
    D --> E["Exact + near-duplicate removal"]
    E --> F["Final 5,000 × 30 manifest"]
    F --> G["WD EVA02 tagging"]
    F --> H["Aspect-preserving preprocessing"]
    G --> I["Content caption"]
    H --> J["C-RADIO backbone features"]
    H --> K["SigLIP2-g visual features"]
    I --> L["SigLIP2-g text features"]
    K --> M["Content subtraction"]
    L --> M
    J --> N["Layer/feature probes"]
    M --> N
    N --> O["Resampler size and structure decision"]
    O --> P["Joint artist embedding + reconstruction training"]
    P --> Q["Anima 28-block K/V adapter training"]
```

### 3.1 남은 작업의 간단한 순서

1. RunPod에서 5,000명 × 30장을 수집하고 태깅·caption까지 생성한다.
2. 소규모 probe subset으로 C-RADIO 중간 레이어를 비교한 뒤 사용할 특징만 확정한다.
3. 확정된 특징을 전체 데이터에 캐싱하고, 단일 경로의 Style Encoder와 학습 전용 decoder를 훈련한다.
4. 1~8장 set 입력을 처리하는 multi-reference cross-attention을 훈련한다.
5. Anima의 실제 28개 block 형상을 확인하여 block별 Style K/V를 연결하고 생성 loss로 훈련한다.
6. seen/unseen artist와 reference 수별 평가 후 구조와 loss를 조정한다.

## 4. 메타데이터 스캔과 작가 선별

원본 데이터셋은 약 1,060만 행이므로 전체를 메모리에 올리지 않는다. Polars lazy scan, DuckDB 또는 PyArrow dataset scanner로 필요한 컬럼만 읽고, 메타데이터 단계의 저비용 필터를 먼저 수행한다.

필요한 주요 컬럼은 다음과 같다.

- 식별·시간: `id`, `created_at`, `md5`
- 작가: `tag_string_artist`, `tag_count_artist`
- 중복 단서: `tag_string_meta`, `parent_id`, `source`, `pixiv_id`
- 다운로드: `file_url`, `large_file_url`, `original_url`, `file_ext`, `file_size`
- 이미지: `image_width`, `image_height`
- 상태: `is_deleted`, `is_banned`, `is_flagged`, `is_pending`
- 태그: `tag_string_general`, `tag_string_character`, `tag_string_copyright`, `tag_string_meta`

초기 기본안은 다운로드 가능한 정상 이미지와 단일 작가 태그가 있는 행을 우선하는 것이다. 다중 작가 작품, rating, 지원 확장자, 삭제·flag 상태의 정확한 포함 정책은 데이터 분포를 집계한 뒤 설정 파일에 명시한다.

### 4.1 작가별 시간 창

`created_at`은 작품 공개일이 아닌 Danbooru 업로드일이다. 따라서 이것을 화풍 시기의 완전한 증거로 간주하지 않고, 이용 가능한 시간 근사치로만 사용한다.

작가별 후보를 `created_at`순으로 정렬한 뒤, 최종 30장과 중복 제거 여유분을 확보할 수 있는 연속 시간 창을 찾는다. 가능한 창이 여러 개라면 더 최근인 창을 우선한다. 정확한 `date_window_days`는 분포를 확인해 결정한다. 고정된 창 하나로 충분한 작가가 크게 줄어들 경우에는 다음 순서로 완화한다.

1. 기본 시간 창 안에서 충분한 작가 선별
2. 부족한 작가에 한해 창을 단계적으로 확대
3. 확대된 실제 창 길이를 manifest에 기록

### 4.2 최신성 가중 표집

최신 이미지가 다수를 차지하도록 작가 선택과 작가 내부 이미지 선택에 별도의 최신성 가중치를 둘 수 있다. 초기 후보식은 다음과 같으며, 감쇠 상수는 데이터 분포를 보고 정한다.

```text
w_artist(a) = exp(-(T_snapshot - t_recent(a)) / tau_artist)
w_image(i|a) = exp(-(t_recent(a) - t_i) / tau_image)
```

- `T_snapshot`: 데이터셋 기준 시점
- `t_recent(a)`: 해당 작가 후보 중 가장 최근 업로드 시점
- `t_i`: 이미지 업로드 시점
- `tau_artist`, `tau_image`: 아직 미정인 감쇠 상수

동일한 설정과 seed로 결과를 재생성할 수 있어야 한다. 작가당 처음부터 30장만 내려받지 않고, 다운로드 실패와 near-duplicate 제거를 감안한 후보 배수를 먼저 뽑는다. 후보 배수는 소규모 측정 후 결정한다.

### 4.3 작가 split과 작가 내부 이미지 split

작가 일반화와 같은 작가의 미관측 이미지 일반화를 혼동하지 않도록 두 축을 모두 분리한다.

**작가 단위 split**의 기본안은 5,000명 중 4,750명(95%)을 train, 125명을 validation, 125명을 test로 고정하는 것이다. Validation/test 작가는 Style Encoder와 Anima K/V 학습에 전혀 사용하지 않는다.

**이미지 단위 split**도 각 작가 안에서 별도로 둔다.

- Train 작가: 기본안은 30장 중 27장을 동적 학습 pool, 3장을 seen-artist held-out target으로 둔다.
- Validation/test 작가: 최대 8장을 고정 reference pool, 나머지 22장을 target pool로 두고 1/2/4/8장 reference 조건을 같은 target에 비교한다.
- 정확한 장수는 manifest 통계를 본 뒤 설정으로 조정하되 split seed와 이미지 ID는 고정한다.

Train 작가의 27장 학습 pool 안에서는 reference와 target을 고정하지 않는다. 매 step 한 이미지를 target으로 뽑고 나머지 이미지 중 1~8장을 reference로 뽑으므로, 모든 이미지가 학습 중 reference와 target 역할을 번갈아 맡는다. 일반 단계에서는 target과 reference가 서로 다른 이미지여야 하며 near-duplicate도 함께 배제한다. 초기 self-reference curriculum을 사용할 때만 설정된 확률로 target을 reference 집합에 의도적으로 포함하고, 그 비율을 점차 0에 가깝게 낮춘다.

고정 held-out 이미지는 이러한 역할 교대에 넣지 않는다. 이는 seen-artist 평가에서 학습에 사용되지 않은 target을 보장하기 위한 것이다. Unseen validation/test 작가의 reference와 target 역시 학습에 들어가지 않는다.

## 5. 빠른 중복 및 near-duplicate 제거

1차 데이터셋에서는 스타일 유사도를 검사하지 않는다. 중복 제거는 다음의 저비용 계층으로 제한한다.

### 5.1 메타데이터 단계

1. 동일 `md5`는 전역 exact duplicate로 묶는다.
2. `tag_string_meta`의 `duplicate`, `pixel-perfect_duplicate` 표식을 중복 단서로 사용한다.
3. 정규화된 동일 source URL을 보조 단서로 사용할 수 있다.
4. 동일 `pixiv_id`만으로 제거하지 않는다. 다중 페이지 게시물은 서로 다른 이미지일 수 있기 때문이다.

### 5.2 다운로드 뒤 이미지 단계

최종 후보 풀에 대해서만 작은 정규화 thumbnail의 64-bit pHash/dHash 또는 wHash를 계산한다. 모든 이미지 쌍을 비교하지 않고 hash prefix bucket, BK-tree 또는 LSH로 가까운 후보만 찾는다. 경계 사례에 한해 작은 thumbnail SSIM으로 확인할 수 있다.

첫 실험의 perceptual near-duplicate 검사는 기본적으로 작가 내부에서 수행한다. 전역 near-duplicate 제거는 재업로드나 작가 태그 오류를 잡을 수 있지만 공동 작업·재가공 이미지를 잘못 제거할 위험이 있으므로 별도 실험 항목으로 둔다.

Hash 거리와 SSIM 임계값은 소규모 수동 감사 표본으로 보정한다. 제거 시에는 `removed_id`, `kept_id`, 제거 이유, hash 거리 및 확인 점수를 기록한다. 대표 이미지는 최신성, 해상도, 다운로드 상태를 이용한 고정 규칙으로 선택하되 정확한 우선순위는 데이터 조사 후 확정한다.

## 6. 다운로드와 이미지 전처리

C-RADIOv4-SO400M은 patch size 16이며, 16 단위의 비정사각 해상도를 처리할 수 있다. 따라서 정사각형 강제 변환을 피하고 원본 종횡비를 최대한 보존한다.

전처리 원칙은 다음과 같다.

1. EXIF orientation을 적용하고 입력 색상 및 alpha 처리 방식을 고정한다.
2. 이미지가 token/pixel 예산보다 클 때만 종횡비를 유지해 축소한다.
3. 축소 후 각 변을 16의 배수인 지원 해상도에 맞춘다.
4. padding을 기본값으로 삼지 않고, 원본 비율과 가장 가까운 크기가 되도록 최소 영역만 결정적으로 crop한다.
5. 임의 square crop은 하지 않는다.
6. 작은 이미지는 특별한 이유가 없으면 확대하지 않는다. 16 배수 정렬에 필요한 최소 crop 정책을 사용한다.
7. resize 크기, interpolation, crop box와 최종 해상도를 manifest에 기록한다.

`max_long_side` 또는 `max_pixels`, crop anchor, interpolation 방식은 품질·처리량·feature cache 크기를 측정한 뒤 고정한다. C-RADIO 특징을 사전 계산할 것이므로 cache용 기본 view는 결정적이어야 한다. 학습 augmentation이 필요하다면 cache 설계와 함께 별도로 정한다.

## 7. WD EVA02 태깅과 content caption

최종 150,000장 전체를 지정한 WD EVA02 Tagger로 batch inference한다. 현재 링크는 Hugging Face Space이므로, 본 처리 전에 다음을 반드시 고정한다.

- Space 또는 실제 model artifact의 revision
- 전처리 해상도와 normalization
- tag vocabulary revision
- general/character 임계값과 정렬 규칙
- batch runtime 및 backend

최종 문자열만 저장하지 않고 가능한 경우 각 태그의 원시 확률 또는 logit도 함께 저장한다. 그러면 이미지 재추론 없이 threshold와 caption 규칙을 바꿀 수 있다.

Content caption은 이미지의 내용을 설명하되 작가 정체성과 직접적인 스타일 누출을 줄이는 방향으로 만든다. 기본적으로 general, character, copyright 계열의 내용 태그를 후보로 사용하고, artist 태그와 스타일·품질·메타 성격의 태그는 제거한다. 정확한 allow/deny 목록은 태거 출력 분포를 확인한 뒤 버전 관리한다.

## 8. C-RADIO와 SigLIP2-g 특징 구성

C-RADIOv4-SO400M은 backbone의 summary/spatial 특징과 `siglip2-g` teacher adaptor 출력을 함께 제공한다. 또한 intermediate layer 출력을 얻을 수 있으므로, 다음 두 계열을 Resampler 후보 입력으로 둔다.

1. C-RADIO backbone의 공간 토큰: 선, 질감, 형태, 배치 등 국소·공간 구조 보존
2. SigLIP2-g의 image 특징에서 content-caption text 특징을 뺀 residual: 내용 성분을 약화한 전역 후보 특징

초기 residual 정의는 다음과 같다.

```text
s_visual = normalize(siglip2_g_image)
s_text   = normalize(siglip2_g_text(content_caption))
s_resid  = normalize(s_visual - lambda_content * s_text)
```

`lambda_content=0`인 subtraction 없는 기준선부터 여러 값을 비교한다. 단순 벡터 subtraction이 곧 스타일 분리를 보장하지는 않으므로, residual의 작가 검색 성능과 content leakage를 함께 측정한다.

SigLIP2-g summary의 차원, backbone spatial token의 차원과 선택 레이어가 다를 수 있으므로 각각 projection한 뒤 Resampler에 제공한다. 이때 content residual과 공간 특징을 별도 네트워크로 분기해 서로 경쟁시키지 않고, 동일한 Resampler memory를 만드는 하나의 경로 안에서 결합한다.

## 9. 사용할 C-RADIO 레이어 결정

전체 150,000장에 여러 레이어 특징을 모두 저장하기 전에 pilot subset으로 probe한다. 후보 비교는 다음 정도로 제한한다.

- 최종 레이어만 사용
- 얕은·중간·깊은 레이어를 소수 선택한 sparse multi-layer 입력
- 여러 레이어의 learned weighted mixture
- backbone 공간 특징 단독
- SigLIP2-g residual 단독
- 공간 특징 + residual 결합

중간 레이어 번호는 사전에 확정하지 않는다. C-RADIO 실제 block 수와 출력 텐서를 확인한 뒤 균등한 깊이의 몇 지점을 후보로 삼는다. 선택 기준은 다음과 같다.

- held-out 이미지의 작가 retrieval / prototype 분류
- decoder의 구조·특징 재구성 품질
- content/character/copyright를 예측하는 leakage probe
- 이미지당 cache 크기와 처리량
- 동일 작가 내 안정성과 서로 다른 작가 간 분리도

레이어 선택 후에만 전체 데이터 특징 cache를 생성한다. 그렇지 않으면 다중 레이어 공간 토큰 때문에 저장공간이 불필요하게 커진다.

## 10. Resampler 표현 학습

### 10.1 단일 공통 경로

```mermaid
flowchart TD
    A["C-RADIO spatial tokens + SigLIP2-g residual"] --> B["Shared Style Resampler"]
    B --> C["Style memory tokens"]
    C --> D["Artist embedding head"]
    C --> E["Training-only reconstruction decoder"]
    D --> F["Supervised contrastive + prototype loss"]
    E --> G["Feature / structure reconstruction loss"]
    C -. later .-> H["Anima 28-block Style K/V"]
```

한쪽 경로가 비활성화되는 것을 피하기 위해 style용 encoder와 reconstruction용 encoder를 따로 두지 않는다. 하나의 Style Resampler가 만든 memory token을 두 loss가 함께 제약한다.

초기 목적함수는 다음 형태로 둔다.

```text
L_repr = lambda_rec   * L_reconstruction
       + lambda_con   * L_supervised_contrastive
       + lambda_proto * L_prototype
```

- `L_reconstruction`: backbone 공간 특징 또는 선택한 시각 target의 복원
- `L_supervised_contrastive`: 같은 작가의 서로 다른 이미지를 positive로 사용
- `L_prototype`: 작가별 prototype 주변으로 표현을 모음

Loss 가중치, decoder target, memory token 수, hidden dimension, Resampler depth는 feature probe 뒤 정한다. 원본 publication 시점과 실제 스타일 시기가 없고 스타일 유사도 필터도 하지 않으므로, 1차 실험에서는 작가당 하나의 prototype을 둔다는 가정을 유지하되 결과 해석 시 label noise를 고려한다.

### 10.2 여러 reference 지원

Resampler는 1~8개의 reference 이미지가 들어올 수 있도록 set 입력을 지원해야 한다. 이미지 순서에 과도하게 의존하지 않게 이미지별 표시와 permutation augmentation 또는 permutation-invariant aggregation을 검토한다.

표현 학습 초기에는 single-image reconstruction도 유지하여 각 이미지의 유용한 정보가 사라지지 않게 한다. 동시에 같은 작가의 서로 다른 이미지들을 묶어 만든 memory가 안정적인 작가 embedding을 갖도록 학습한다. Reference 수가 1, 2, 4, 8일 때의 retrieval·재구성·최종 생성 품질 곡선을 별도로 측정한다.

## 11. Anima Style Adapter로의 연결

Resampler의 표현과 규모가 정해진 뒤 Anima의 28개 block 각각에 Style Adapter용 K/V projection을 추가한다. 이 단계의 기본 학습 샘플은 같은 작가의 reference 1~8장, target의 content prompt, target 이미지로 구성한다.

초기에는 reference 집합에 target 이미지를 포함하는 비율을 높여 어떤 시각 특징을 이용해야 하는지 빠르게 학습한다. 이후 target과 겹치지 않는 reference만 사용하는 비율을 점진적으로 증가시킨다. 정확한 ramp schedule과 self-reference 비율은 Resampler 검증 뒤 결정한다.

Decoder는 이 표현 학습의 보조 장치이므로 최종 Style Transfer 추론 경로에는 포함하지 않는다.

## 12. 캐시와 산출물

각 단계는 다시 실행할 수 있고 중간 결과만 교체할 수 있도록 sharded Parquet 또는 유사한 columnar manifest로 저장한다.

| 산출물 | 핵심 내용 |
|---|---|
| `artist_stats.parquet` | 작가별 전체 수, 필터 후 수, 최초·최근 시점, 가능한 window |
| `candidate_manifest.parquet` | 표집 점수, seed, 시간 창, 다운로드 후보 |
| `dedup_manifest.parquet` | kept/removed ID, exact 또는 perceptual 근거, 거리 |
| `image_manifest.parquet` | 최종 150k ID, artist, URL, md5, artist split, image split/pool |
| `preprocess_manifest.parquet` | 원본·최종 크기, resize, crop box, hash |
| `tagger_outputs` | tagger revision, vocabulary, raw score, 선택 태그, content caption |
| `feature_probe` | 후보 레이어별 지표와 cache 비용 |
| `cradio_features` | 확정 레이어와 adaptor의 sharded BF16/FP16 특징 |
| `experiment_config` | 모든 threshold, seed, revision 및 loss 설정 |

특히 데이터셋, tagger, C-RADIO와 코드의 revision hash를 함께 저장한다. Feature cache key에는 이미지 hash, 전처리 버전, 해상도, crop, 모델 revision, adaptor와 레이어 목록이 모두 포함되어야 한다.

## 13. 1차 실험의 완료 조건

다음 조건을 만족하면 150,000장 데이터 준비 단계가 완료된 것으로 본다.

- 5,000명의 작가마다 검증된 최종 이미지가 정확히 30장 존재
- 각 작가의 실제 업로드 날짜 범위와 표집 가중치가 기록됨
- exact/near-duplicate 제거 이력과 수동 감사 결과가 존재
- 모든 이미지의 결정적 전처리 정보와 태거 결과가 존재
- 데이터 누락 없이 artist-level 및 image-level train/validation/test manifest를 재생성할 수 있음

Resampler 구조 확정 전에는 다음 결과가 필요하다.

- 선택 레이어 조합별 작가 retrieval 및 prototype 성능
- 재구성 품질과 content leakage 비교
- 1/2/4/8 reference에 따른 성능 변화
- feature cache 크기, throughput 및 VRAM 측정
- 선택한 모델 크기가 과적합 없이 150,000장 실험에 적합하다는 근거

## 14. 아직 결정하지 않을 항목

다음 값은 데이터 분포와 pilot 실험 없이 임의로 고정하지 않는다.

- 작가별 최대 날짜 창과 단계적 완화 규칙
- 작가·이미지 최신성 감쇠 상수
- 다운로드 후보 배수
- rating, 다중 작가 및 각 status flag의 포함 정책
- perceptual hash 종류와 거리·SSIM 임계값
- 최대 입력 크기, pixel/token 예산, resize interpolation과 crop anchor
- Tagger threshold와 content caption allow/deny 목록
- C-RADIO 중간 레이어와 feature 정밀도
- content subtraction 계수
- Resampler 차원, token 수, layer 수와 decoder 구조
- loss 가중치와 multi-reference 구성 비율
- Anima K/V 차원, block별 공유 여부와 학습 schedule

## 15. 참고 자료

- [Shio-Koube/Danbooru-2026-parquet-metadata](https://huggingface.co/datasets/Shio-Koube/Danbooru-2026-parquet-metadata)
- [ashen-sensored/wd-eva02-tagger-2026-canary Space](https://huggingface.co/spaces/ashen-sensored/wd-eva02-tagger-2026-canary)
- [nvidia/C-RADIOv4-SO400M model card](https://huggingface.co/nvidia/C-RADIOv4-SO400M)
- [NVlabs/RADIO official repository](https://github.com/NVlabs/RADIO)
