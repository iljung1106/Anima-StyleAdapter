# B′ Dual-query Style Resampler 학습 계획

## 목표

현재 C-RADIO-only Resampler의 semantic retrieval 능력은 유지하면서, Anima 생성에 필요한 선·색·질감·렌더링 정보를 Qwen Image VAE latent에서 보충한다. 여러 reference에서 공통된 작가 스타일을 추출하되, 개별 이미지의 내용·구도를 그대로 복사하는 shortcut은 억제한다.

## 입력과 구조

- Semantic bank: C-RADIOv4-SO400M `L18`, `L24` spatial token. 각 레이어를 별도 LayerNorm/Linear로 projection하고 layer-type embedding을 더한다.
- Perceptual bank: Qwen Image VAE `/8`, 16-channel latent. `2×2 stride-2 convolution + 2 residual blocks`로 token 수를 줄인 뒤 독립 projection한다.
- Query bank:
  - `64×1024` spatial/perceptual query (`8×8` 논리 grid, normalized 2D RoPE)
  - `16×1024` global/style query (위치 embedding 없음)
- Fusion: 4개 block에서 query self-attention, C-RADIO cross-attention, VAE cross-attention, MLP를 수행한다. 두 modality의 K/V projection·normalization·residual gate는 공유하지 않는다.
- 2D 좌표는 영상별 `[-1, 1]`로 정규화하고 aspect-ratio/resolution global embedding을 별도로 제공한다.

## Artist descriptor head

Spatial/global token에 각각 작가 loss를 걸지 않고, 전체 `80×1024` token을 읽는 하나의 작은 attention-pooling head를 둔다.

- 4개 learned pooling query와 2층 cross-attention
- `4×512 → 512` projection 후 L2-normalized artist descriptor
- descriptor에서 만든 `2~4×1024` artist-summary token은 버리지 않고 실제 Set Transformer 입력에 포함
- Head가 분류만 대신 해결하지 못하도록 작게 유지하고, VAE branch `10%`, C-RADIO branch `5%` modality dropout을 사용

## Multi-reference Set Transformer

각 reference의 80개 query token과 artist-summary token을 하나의 set으로 취급한다. Reference 순서 embedding은 사용하지 않는다. 32개 learned output query가 전체 set에 cross-attention한 뒤 2층 cross-slot Transformer를 거쳐 `32×1024` style context token을 만든다. 최종 token은 Anima LLM Adapter의 입력이 아닌 **post-LLM Adapter `512×1024` context의 미사용 zero position**을 교체한다.

## 사전학습 loss

```text
L = L_semantic_reconstruction
  + 0.10 * L_vae_reconstruction
  + 0.05 * L_artist
  + 0.01 * L_token_diversity
```

- Semantic reconstruction: 무작위 128~256 grid 위치의 L18/L24를 cosine + SmoothL1로 복원
- VAE reconstruction: spatial query에서 낮은 가중치로 latent patch/low-frequency 성분을 복원. 정확한 이미지 복사가 주 목적이 되지 않게 한다.
- Artist loss: episodic angular prototypical loss를 주 loss로, supervised contrastive를 약하게 보조로 사용한다.
  - `L_artist = L_angular_prototype + 0.25 * L_supcon`
  - descriptor 512, scale `16`, angular margin `0.10`
  - support/query를 같은 작가의 서로 다른 이미지로 구성
  - hard negative는 가능하면 유사 캐릭터·일반 태그의 다른 작가에서 선택
- 고정 5,000-class ArcFace head는 필수가 아니며, 추가하더라도 약한 보조 loss로만 사용한다.

초기 episodic descriptor가 모든 작가에 같은 값을 내는 대칭점에 머물 경우에만, 선택된 train 작가 3,000명의 training-only CosFace proxy를 `0.5 × L_proxy`로 `L_artist` 안에 추가한다. Proxy는 validation의 unseen artist 평가에는 사용하지 않으며 사전학습 뒤 제거한다.

위 가중치는 초기값이며 raw loss가 아닌 각 경로의 gradient norm과 validation Pareto를 기준으로 조정한다.

## 학습 단계

1. **Per-reference pretraining:** C-RADIO/VAE를 동결하고 dual-query Resampler와 decoder/head를 reconstruction + artist loss로 학습한다.
2. **Context bootstrap:** Resampler를 동결하고 Set Transformer/context projection을 exact-self flow로 학습한다.
3. **Joint alignment:** Resampler 상위 1~2 block과 artist head/Set Transformer를 열고 flow loss와 함께 공동학습한다. Reconstruction/artist loss는 사전학습의 `1/5~1/10`로 낮춘다.
4. **Multi-reference curriculum:** target 포함률을 점차 낮추고 1~8개 target-excluded reference로 확장한다.

## Artist-summary token 전달 비교

Artist descriptor head와 artist loss는 두 조건 모두 유지한다. 차이는 descriptor에서 만든 `2~4×1024` artist-summary token을 Multi-reference Set Transformer에 실제로 전달하는지 여부뿐이다.

- **summary 전달:** reference별 80개 query token과 artist-summary token을 함께 사용
- **summary 미전달:** reference별 80개 query token만 사용하며 descriptor는 artist loss 계산에만 사용

동일한 초기화·데이터 순서·학습량으로 두 조건을 비교한다. artist-summary token은 held-out artist의 1/2/4/8-reference 성능, correct-vs-wrong reference 차이, reference-view consistency, common-output ratio와 고정 prompt/seed 생성 샘플을 함께 보고 유지 여부를 결정한다.

이 비교는 per-reference 사전학습 단계가 아니라 summary token이 실제 입력으로 사용되는 **Context bootstrap 이후**에 수행한다. 구현에서는 같은 encoder/head 가중치를 두고 `include_artist_summary`만 바꾸므로, descriptor 학습 유무나 파라미터 수 차이가 결과에 섞이지 않는다.

## 구현 및 실행 계약

- 모델: `src/anima_style_data/dual_query_resampler.py`
- 실제 캐시 학습: `anima-data --config configs/anima500k-human.yaml dual-query-resampler-train`
- 독립 2-step 검증: `anima-data --config configs/anima500k-human.yaml dual-query-resampler-smoke`
- 입력 캐시: `style_features_l18_l24_siglip_l24`와 `anima_latent_cache_qwen_2d`의 image ID 교집합
- episode: 작가 4명 × 서로 다른 이미지 2장. 따라서 angular prototype과 supervised contrastive loss 모두 매 step 실제 positive support를 갖는다.
- 첫 사전학습 inventory: train 작가 3,000명 × 15장 = 45,000장. 별도의 validation 작가 150명 × 15장 = 2,250장은 optimizer에 노출하지 않는다. 선택은 seed로 고정해 resume와 재실행에서 동일하게 유지한다.
- 현재 사전학습 구성은 reconstruction decoder, artist head와 training-only proxy를 포함해 약 130.0M parameter다. C-RADIO와 Qwen VAE는 캐시만 사용하므로 학습 그래프에 포함되지 않는다.
