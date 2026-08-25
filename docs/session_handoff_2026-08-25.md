# Anima Style Adapter 세션 인계 문서

작성 시각: 2026-08-25 (Asia/Seoul)

이 문서는 다른 Codex 세션이 현재 작업을 처음부터 다시 조사하지 않고 바로 이어가기 위한 인계 자료다. 현재 실행 상태, 프로젝트 목표, 이미 검증한 사실, 실패 양상, 다음 실험에서 반드시 바꿔야 할 점을 함께 기록한다.

## 1. 최종 목표

`circlestone-labs/Anima`의 원본 가중치는 동결하고, 한 장 또는 여러 장의 레퍼런스 이미지로부터 작가/화풍 정보를 추출해 Anima의 28개 블록에 별도의 Style Cross-Attention으로 주입하는 범용 Style Adapter를 학습한다.

중요한 제품 요구사항은 다음과 같다.

- 학습 때 보지 못한 작가와 이미지에도 일반화해야 한다.
- 단일 레퍼런스 성능이 가장 중요하며 2/4/8장에서는 점진적으로 좋아져야 한다.
- 이미지의 색, 선화, 채색, 질감, 이목구비와 형태 표현을 재현해야 한다.
- 어느 정도 content leakage가 생기는 것은 허용한다. 현재는 약하고 무의미한 스타일 출력보다 재현력을 우선한다.
- 텍스트 프롬프트는 Full, Tag Dropout, Short, Empty를 섞는다.
- Empty prompt는 반드시 단일 `reference=target` exact-self episode로 제한한다.
- 추론 시 사용자 strength는 Artist residual에만 적용하고 Common에는 적용하지 않는 것이 장기 목표다.
- `val/functional/panel`과 `val/functional/fixed_reference`를 500스텝마다 생성해야 한다.
- `fixed_reference`는 같은 prompt/seed/noise에서 레퍼런스만 바꾸므로 작가별 functional separation의 핵심 정성 평가다.

## 2. 인프라와 경로

### 로컬

- 저장소: `D:\AI Research\Anima_StyleAdapter`
- 브랜치: `main`
- 현재 커밋: `270b96a Add v31 v34 training with LoRA auxiliary`
- 로컬은 다음 untracked 항목이 있으며 사용자 소유이므로 삭제하거나 커밋하지 말 것:
  - `.codex-review/`
  - `ModelBackups/`
  - `SampleImages/`
  - `artifacts/`
  - `scripts/export_panel_references.py`

### RunPod

- SSH: `ssh root@87.120.211.205 -p 15263 -i C:\Users\1wndr\.ssh\id_ed25519`
- 원격 저장소: `/workspace/Anima-StyleAdapter`
- venv: `/workspace/Anima-StyleAdapter/.venv`
- 데이터 루트: `/workspace/Anima-StyleAdapter/data/anima500k-human`
- 로그 루트: `/workspace/logs/anima-data`
- Hugging Face cache: `/workspace/.cache/huggingface`
- 원격 커밋도 `270b96a`
- 원격 untracked `wandb/`와 `53132;`는 관련 없는 항목이므로 삭제하지 말 것.

사용자는 이 프로젝트 범위에서 Git commit/push/pull, SSH 전송과 원격 실행을 허용했다. 다음 세션도 실제 실행 전 현재 프로세스를 확인하고 관련 없는 프로세스는 건드리지 않아야 한다.

## 3. 현재 실행 중인 실험

실험명: `detail_preserving_style_cross_attention_v34_lora_joint_v1`

- CLI: `detail-style-v34-lora-joint-train`
- PID 파일: `/workspace/logs/anima-data/detail-style-v34-lora-joint-v1.pid`
- PID: `163169`
- 로그: `/workspace/logs/anima-data/detail-style-v34-lora-joint-v1.log`
- W&B: <https://wandb.ai/1wndrla17-kyung-hee-university/anima-style-adapter/runs/detail-style-v34-lora-joint-v1>
- 마지막 확인 시점: `1260/3000`
- 체크포인트: 250, 500, 750, 1000, 1250 스텝이 존재한다.
- 프로세스는 살아 있었고 H100에서 실행 중이었다.

이 실험은 중단하지 않았다. 그러나 다음 본 실험의 출발점으로 권장하지 않는다. v31 체크포인트에서 시작했고 LoRA common 회귀가 Artist 경로의 공통 출력 붕괴를 강화하는 것으로 진단됐다.

안전한 상태 확인 예:

```bash
pid=$(cat /workspace/logs/anima-data/detail-style-v34-lora-joint-v1.pid)
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 30 /workspace/logs/anima-data/detail-style-v34-lora-joint-v1.log
```

사용자가 중지를 지시하면 먼저 `SIGINT`로 체크포인트 저장 기회를 주고 기다린다. 임의로 다른 Python 프로세스를 종료하면 안 된다.

## 4. 현재 모델 구조

### Frozen 구성

- Frozen Anima DiT 28 blocks
- Frozen Dual-query Resampler
- C-RADIO 특징과 Qwen VAE latent를 읽도록 사전학습된 Dual-query Resampler 체크포인트:
  - `dual_query_resampler_bprime_v8/checkpoints/step-010000.pt`
- 캐시된 Resampler token:
  - `dual_query_reference_tokens_v8_step10000`

### Dual-query 토큰

레퍼런스마다 84개 typed token을 만든다.

- spatial: 64개
- global: 16개
- summary: 4개
- dimension: 1024

초기 C-RADIO 특징 선정은 다음이었다.

- L18 spatial
- L24 spatial
- L24 SigLIP CLS/global 정보

### Detail Reader

84개 typed token과 여러 reference의 memory를 읽어 28개 canonical style slot을 만든다.

- dim 1024
- reader cross-attention 2층
- cross-slot mixer 2층
- FF dim 3072
- output slots 28
- slot type counts `[16, 8, 4]`
- same-slot은 attention bias만 받고 다른 slot도 읽을 수 있다.
- reference identity, token type, slot identity가 구분된다.

### Style Cross-Attention

- Anima의 기존 Q와 full-rank O, `gate_cross(t)`를 재사용한다.
- Text attention과 Style attention의 softmax는 분리한다.
- Style branch는 Text cross-attention 뒤, MLP 이전 residual 경로다.
- K/V는 텍스트 K/V 복사본이 아니라 Style용으로 학습한다.
- 28개 블록 K/V는 4개의 full-rank shared base와 블록별 rank-64 delta로 구성한다.
- medoid blocks: `[3, 12, 18, 26]`
- block mapping은 config `adapter.block_to_base` 참조.
- delta는 작은 nonzero 초기화다.
- 별도 trainable scalar gate는 두지 않는다.
- timestep/block 강도 profile과 `global_gain=1.0`을 쓴다.

### Common / Artist 분리

현재 adapter는 `separated_common_artist_shared_base`다.

- Common: reference-free, Q-conditioned common K/V dictionary, 16 tokens
- Artist: Reader가 만든 reference-conditioned 28 tokens
- Artist null residual: real reference 출력에서 동일 K/V 경로의 null-token 출력을 빼 reference-independent 성분을 줄인다.
- Common과 Artist의 효과는 Frozen Anima 내부에서 합쳐진다.

문제는 구조상 분리되어 있어도 loss가 LoRA Teacher의 raw common 성분을 Artist combined 출력에 다시 넣으면 Artist가 공통 방향을 학습할 수 있다는 것이다.

## 5. 현재 v34 + LoRA joint 설정

Config key:

```yaml
detail_preserving_style_cross_attention_v34_lora_joint
```

중요 설정:

- `extends: detail_preserving_style_cross_attention`
- `initial_checkpoint: detail_preserving_style_cross_attention_v31_artist_null_residual/checkpoints/step-0003500.pt`
- 총 3000스텝
- main LR `1e-4`
- common LR `3e-4`
- decay start 2500
- reconstruction off
- prompt modes: Full 30%, Dropout 40%, Short 20%, Empty 10%

Reference curriculum:

- 0~250: exact single
- 250~1000: target included, refs 1/2/4
- 1000~2000: target 포함률 감소, refs 1/2/4/8

LoRA functional teacher:

- teacher directory: `artist_lora_teachers_rank16_256_b2_v5`
- bank: `lora_functional_teacher_bank_rank16_256_v3_broad12x6`
- human ref cache: `dual_query_reference_tokens_v8_step10000`
- synthetic ref cache: `lora_teacher_references_rank16_256x8_v2/dual_query_reference_tokens_v8_step10000`
- 2스텝마다 적용
- 0~500: single only
- 이후 `[single, single, pair, triple]`
- human/synthetic domain 교대
- backward scale 0.25
- batch 6
- refs 1/2/4 = 60/30/10%
- 현재 objective: `teacher_decomposed`

현재 LoRA loss weights:

- centered Huber 1.0
- centered direction 0.75
- centered magnitude 0.20
- functional InfoNCE 0.25
- common Huber 0.05
- common direction 0.025
- common-ratio excess 0.25

마지막 세 항, 특히 raw common Huber/direction이 현재 문제의 핵심 후보다.

## 6. 현재 common-output 억제 loss와 실제 약점

현재 common-output 억제는 세 곳에 존재한다.

1. Heldout functional artist loss
   - `common_output_weight=0.10`
   - artist functional pass는 2스텝마다 실행
   - threshold는 약 0.80에서 0.55로 감소
2. Main controlled common loss
   - `main_common_output_weight=0.05`
   - 4스텝마다 batch 4 probe
3. LoRA functional teacher
   - teacher common ratio보다 margin 0.05 이상 큰 Student common ratio를 벌점
   - `common_ratio_excess=0.25`

하지만 1000스텝에서 validation common ratio가 약 0.92이고 threshold가 약 0.65라면 heldout common weighted loss는 대략:

```text
0.10 * (0.92 - 0.65)^2 = 0.0073
```

2스텝마다만 계산하므로 optimizer-step당 평균 영향은 약 0.0037이다. 매 스텝 약 0.1인 flow loss와 비교하면 작다.

더 중요한 충돌:

- `common_ratio_excess`는 공통 출력을 줄이려 한다.
- LoRA의 `common_huber`와 `common_direction`은 LoRA Teacher의 공통 출력을 재현시킨다.
- Common 경로가 v31에서 사실상 고정된 상태라 LoRA Teacher의 common 성분을 Artist 경로가 흡수할 수 있다.

따라서 현재 loss weight만 키우면 Artist 전체 출력을 줄이는 shortcut으로 갈 위험이 있다. 먼저 LoRA common 회귀를 Artist 목표에서 제거해야 한다.

## 7. 최신 정량 결과

### 500 -> 1000 qualitative-control 변화

Fixed-reference 1x:

- baseline 대비 RMS: `0.1444 -> 0.1341`
- reference 간 pairwise pixel RMS: `0.1146 -> 0.0944`
- pairwise/baseline ratio: `0.794 -> 0.704`

즉 학습이 진행되며 레퍼런스별 차이가 오히려 약 18% 감소했다.

Fixed-reference 2x:

- baseline 대비 RMS: `0.1656 -> 0.1851`
- pairwise RMS: `0.0829 -> 0.1247`

강도를 키우면 차이는 커지지만 한 샘플에서 과도한 왜곡이 생겼다. 방향 학습보다 단순 strength 문제만은 아니다.

### Validation 500 -> 1000

- functional common-output ratio: `0.912 -> 0.920`
- single-reference common ratio: `0.907 -> 0.917`
- artist retrieval top1: `0.969 -> 0.750`
- single-reference retrieval: `0.813 -> 0.563`
- repeatable artist ratio: `0.651 -> 0.575`
- heldout paired improvement: `-0.0498 -> -0.0754`
- heldout direction cosine: `0.0283 -> 0.0077`
- heldout style output ratio: `0.0722 -> 0.0778`

### Validation 1250

- heldout paired improvement: `-0.04934`
- heldout direction cosine: `0.02402`
- heldout desired ratio: `0.24246`
- heldout style output ratio: `0.06719`
- functional common-output ratio: `0.93216`
- single-reference common-output ratio: `0.93093`
- functional retrieval top1: `0.875`
- single-reference retrieval top1: `0.8125`
- functional centered Student/Teacher RMS: 약 `1.079`
- positive cosine: 약 `0.646`
- negative cosine: 약 `-0.201`

1250에서 retrieval과 centered 지표 일부는 회복됐지만 common ratio는 오히려 0.93으로 높다. 즉 토큰이나 centered subspace에서는 작가를 구별할 수 있어도 최종 functional effect 대부분이 공통 방향인 상태다.

### LoRA teacher 경로의 1000 부근

- centered cosine 약 0.03
- InfoNCE accuracy 약 random 수준
- cosine gap 음수인 시점 존재
- Student common ratio 약 0.96
- Teacher common ratio 약 0.64

LoRA auxiliary가 작가별 centered 방향을 성공적으로 가르치지 못하고 공통 출력을 늘린다는 진단과 일치한다.

## 8. Panel과 Fixed Reference가 다르게 보이는 이유

`panel`은 각 episode마다 target caption, content, seed가 다르다. Frozen Anima baseline 자체가 서로 다르기 때문에 결과가 그럴듯하고 다양해 보일 수 있다. 이는 생성 품질은 보여주지만 레퍼런스에 따른 작가 구분력을 통제해서 보여주지 않는다.

`fixed_reference`는 다음을 고정한다.

- 동일 prompt
- 동일 negative prompt
- 동일 seed/noise
- 동일 sampler 설정
- 레퍼런스만 변경

현재 fixed prompt는 Tateyama Ayano, upper body/close-up 계열이며 CFG 4, 30 steps다. 1x에서는 Common은 항상 1배이고 strength는 Artist residual에만 적용된다.

Fixed 결과가 거의 같은 갈색 머리/푸른 하늘 방향으로 모이는 것은 sampler 오류보다 실제 reference-independent/common functional collapse를 드러낸다. Panel만 보고 모델을 선택하면 안 된다.

## 9. 이미 확인한 중요한 연구 방향과 실패 교훈

- 가장 성공적이었던 과거 계열은 소형 Style Tokenizer와 v34 step 500의 비교적 높은 fixed-reference 다양성이다.
- v34 step 500의 fixed-reference pairwise/baseline ratio는 약 0.897이었고 Artist-only에서 차이가 나왔다.
- Native context token을 직접 흉내내는 방식은 본 적 없는 스타일 일반화와 재현 천장이 낮았다.
- Flow MSE만으로는 Frozen Anima가 이미 잘 예측하므로 style residual을 0에 가깝게 만드는 shortcut이 강하다.
- 그렇다고 전체 residual RMS를 강제로 크게 만들면 cracked-wall/VAE-noise 형태의 직교 고주파 출력으로 망가졌다.
- 방향을 모르는 magnitude loss만 과도하게 키우면 안 된다.
- 인간 이미지에 Native artist-tag 반응을 직접 증류하는 것은 의미가 불명확하므로 제외했다.
- Native artist teacher는 실제 Anima의 `@artist`로 생성된 synthetic 도메인에 사용한다.
- MegaStyle은 사용자의 명시로 현재 학습에서 제외한다.
- Resampler reconstruction/prototype 사전학습은 완료되어 있으며 본 Style Adapter 학습 중에는 Frozen이어야 한다.
- 지나치게 많은 보조 loss를 매 스텝 계산하면 느리고 gradient 충돌 분석도 어려워진다. 핵심 objective를 적게 유지한다.

## 10. 다음 권장 실험: 완전 fresh Common/Artist curriculum

현재 v31 체크포인트를 재사용하지 않는다.

유지:

- Frozen Anima
- Frozen pretrained Dual-query Resampler와 캐시

새로 초기화:

- Detail Reader
- Common K/V
- Artist shared K/V bases
- block delta
- Artist null context
- slot mixer

새 output/config key를 만들고 기존 산출물을 덮어쓰지 않는다.

### Phase A: Common bootstrap, 0~500

- Common-only 경로만 학습
- Reference/Reader/Artist는 비활성
- 동일 `x_t`, timestep, prompt에서 synthetic Native artist teacher의 작가 평균 효과를 Common 목표로 사용
- Common direction + RMS band만 사용
- Common RMS 목표: native common의 0.8 -> 1.0배
- Main flow MSE는 사용하지 않음

이후 Common을 완전히 방치하거나 Combined flow로 자유롭게 학습시키지 않는다. 별도의 Common-only teacher update로만 느리게 계속 교정한다.

### Phase B: Artist exact-self bootstrap, 500~1500

- 단일 `reference=target`
- Common은 Common-only update로만 학습
- Reader와 Artist K/V를 연다.
- Native/LoRA Teacher 모두 batch artist mean을 제거한 centered Artist effect만 지도
- LoRA는 single teacher만 사용
- LoRA raw `common_huber`, `common_direction`, raw common ratio 회귀 제거
- Flow MSE는 0.25 -> 1.0으로 ramp
- Empty prompt는 단일 exact-self만 허용
- Artist anti-common은 첫 Artist 스텝부터 활성화

권장 핵심 loss:

1. Rectified-flow MSE
2. Centered teacher direction loss
3. Centered magnitude band
4. Artist-only common-output penalty
5. 작가 구분용 functional InfoNCE는 bootstrap 후반부터 약하게 추가 가능

Artist loss를 계산할 효과는 Combined 전체가 아니라 다음이어야 한다.

```text
artist_effect = output(common + artist) - output(common_only)
```

Controlled batch는 같은 Q, `x_t`, timestep, prompt를 사용하고 artist/reference만 다르게 한다.

Common penalty:

```text
common_ratio = RMS(mean_artist(artist_effect))
             / RMS(centered_native_artist_effect)
loss = relu(common_ratio - threshold)^2
```

초기 권장값:

- threshold `0.75 -> 0.60`
- controlled common weight `0.3 -> 0.5`
- centered magnitude floor `0.25 -> 0.60`
- LoRA centered backward scale `0.15~0.25`

Common penalty만 세게 하면 모든 Artist 출력을 줄이므로 centered magnitude floor와 반드시 함께 사용한다.

### Phase C: Self 포함, 1500~3000

- refs 1/2/4 = 60/30/10%
- target 포함률 1.0 -> 0.5
- single-LoRA centered teacher 유지
- Native centered teacher 2~4스텝마다
- functional InfoNCE/correct-vs-wrong 시작
- main controlled common probe 4스텝마다

다음 조건을 두 번 연속 만족하기 전에는 target 포함률을 더 낮추거나 mixture teacher를 열지 않는다.

- validation Artist common ratio < 0.70
- centered cosine gap > 0
- retrieval이 random보다 명확히 높음
- fixed-reference pairwise variation이 직전 평가보다 감소하지 않음

### Phase D: Target 제외 전환, 3000~6000

- target 포함률 0.5 -> 0
- refs 1/2/4/8 = 45/30/17/8%
- LoRA schedule을 `[single, single, pair, triple]`로 확장
- mixture도 raw common이 아니라 centered functional effect만 지도
- main flow는 매 스텝
- Native Teacher는 4~8스텝마다

### Phase E: 일반화, 6000~10000

- 대부분 target-excluded
- single-reference 비중을 가장 높게 유지
- teacher는 희소 보정
- LR decay는 7500 이후
- 500스텝마다 panel/fixed-reference 생성
- 모델 선택은 fixed-reference separation, 생성 안정성, heldout 성능을 함께 사용

## 11. 필요한 코드 수정

권장 구현 위치:

- 설정: `configs/anima500k-human.yaml`
- 학습 runner: `src/anima_style_data/detail_style_training.py`
- LoRA objective: `src/anima_style_data/lora_functional_distillation.py`
- Artist losses: `src/anima_style_data/artist_effect_losses.py`

필수 변경:

1. 새 fresh config/CLI/output directory를 만든다.
2. `initial_checkpoint: null`, `resume_checkpoint: null`로 둔다.
3. 기존 `teacher_decomposed` 대신 Artist centered-only objective를 추가한다.
4. LoRA Student effect를 `combined - common_only`로 계산한다.
5. LoRA Teacher effect는 batch mean을 제거한다.
6. LoRA `common_huber`, `common_direction`은 Artist backward에서 제거한다.
7. Common-only bootstrap/update와 Artist update의 gradient 경로를 분리한다.
8. Anti-common denominator는 Student total RMS가 아니라 frozen Native centered scale을 사용한다.
9. common loss와 magnitude floor를 함께 적용한다.
10. Phase 전환을 metric gate로 제어하거나, 최소한 single-only 기간을 1500까지 늘린다.
11. W&B에는 raw loss와 weighted loss를 구분해 기록한다.
12. Fixed-reference pairwise variation/common ratio를 checkpoint 선택 지표로 기록한다.

위 수정에서 기존 코드를 대규모로 파괴하지 말고 새 config와 작은 objective 분기로 구현한다. 기존 체크포인트 호환을 억지로 유지할 필요는 없지만 기존 실험 파일은 보존한다.

## 12. 테스트 원칙

AGENTS.md 지침에 따라 과도한 mock이나 수십 개의 경계 테스트를 만들지 않는다.

필요한 테스트만 수행한다.

- centered-only LoRA objective에서 batch mean에 gradient가 가지 않거나 제거되는지
- `combined - common_only`가 Artist effect로 계산되는지
- Common parameter가 Artist-only update에서 gradient를 받지 않는지
- 2-step real Anima smoke가 끝까지 실행되는지
- panel/fixed-reference가 지정 간격에 한 번씩 생성되는지

기존 관련 테스트:

```powershell
python -m pytest -q tests/test_detail_style_cross_attention.py tests/test_lora_functional_distillation.py
```

최근 `270b96a`에서는 관련 64개 테스트가 통과했고, 원격 full-Anima 2-step smoke도 성공했다.

## 13. 작업 절차

1. 현재 원격 PID와 로그 확인
2. 사용자가 현재 실험 중지를 지시했다면 해당 PID만 정상 종료
3. 로컬에서 새 config/objective 작성
4. 필요한 핵심 테스트 실행
5. Git commit/push
6. 원격에서 `git pull`로만 반영
7. 원격 2-step smoke
8. 기존 output을 덮어쓰지 않는 새 output directory로 본 학습 시작
9. W&B URL, PID, 로그, checkpoint 경로 보고
10. 250/500스텝에서 수치와 fixed-reference를 함께 진단

원격에 파일을 직접 덮어써서 로컬과 다른 코드를 만들지 말고 반드시 Git pull 방식으로 동기화한다.

## 14. 다음 세션에서 가장 먼저 할 판단

현재 joint run을 더 오래 두어도 centered retrieval은 좋아질 수 있지만 functional common ratio가 0.93이므로 목표 구조를 검증하기 어렵다. 사용자의 최근 방향은 v31을 재사용하지 않고 fresh curriculum을 구성하는 것이다.

따라서 다음 세션은 다음 순서가 적절하다.

1. 이 인계 문서와 현재 config/code를 읽는다.
2. 최신 원격 상태를 다시 확인한다.
3. 사용자에게 현재 run을 중지하고 fresh 실험 구현/실행할지 짧게 확인하거나, 사용자가 실행을 명시했다면 바로 진행한다.
4. 단순 weight 증폭이 아니라 LoRA common target 제거와 Artist-only centered objective부터 고친다.

핵심 진단 한 줄:

> 현재 모델은 레퍼런스에서 작가를 분류할 정보는 어느 정도 보존하지만, 최종 Anima functional effect에서는 LoRA raw common 목표와 약한 anti-common 규제 때문에 대부분의 출력을 공통 방향으로 만들고 있다.
