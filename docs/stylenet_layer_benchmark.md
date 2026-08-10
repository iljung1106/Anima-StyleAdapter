# StyleNet C-RADIO layer benchmark

`Shio-Koube/stylenet`을 외부 controlled benchmark로 사용해, production에서 선택한 C-RADIO L20/L24와 다른 중간 레이어의 style 분리 능력을 비교한다. 데이터셋 revision은 `4bb6753c7c3c236495b8899552572b97e7ebb081`로 고정한다.

StyleNet은 작가별 tar로 구성된다. 각 group에는 tar의 기준 작가가 그린 `Original` 한 장과 같은 캐릭터를 다른 작가가 그린 `DiffArtist` 세 장이 있다. 다른 group의 Original 1/2/4/8장으로 기준 작가 prototype을 만든 뒤, 각 query group의 네 후보 중 Original을 맞히게 한다. Top-1, MRR과 positive-hardest-negative cosine margin을 기록한다. 이 protocol은 일반적인 작가 분류보다 캐릭터·내용 shortcut을 강하게 통제한다.

## 1차 pooled screen

- 후보 tap: L4, L8, L12, L16, L20, L24, L26
- 각 tap 표현: native summary, spatial mean, spatial mean+std
- 조합: L20+L24 mean, L20+L24 mean+std, L8+L20+L24 mean+std
- C-RADIO 입력: production과 같은 aspect-ratio 보존, 최대 512² pixel, 16 배수 정렬
- 검은 placeholder·decode 실패·64 pixel 미만 이미지 제외
- group 안에 유효하고 서로 다른 이미지 네 장과 Original 하나가 모두 있을 때만 평가
- 전역 exact duplicate인 Original은 reference와 query에서 제외

Pooled screen은 모든 레이어를 약 2 GB 이하의 특징으로 빠르게 비교하기 위한 1차 평가다. 이는 full spatial-token Resampler 성능을 대신하지 않는다.

1차 결과 이후 L10, L14, L18, L22를 증분 추출하는 dense follow-up을 수행한다. L8~L24의 모든 2-layer spatial-mean/full-summary 조합, layer 간 exact linear CKA, 단일-layer oracle 및 오류 겹침을 계산해 단순 점수뿐 아니라 서로 보완적인 tap인지도 판별한다.

실행 결과와 해석은 [stylenet_layer_benchmark_results.md](stylenet_layer_benchmark_results.md)에 기록한다.

## 2차 spatial-token 검증

Pooled 결과와 기존 1,000-artist pilot을 함께 보고 후보를 좁힌다. dense follow-up 결과에 따라 L18, L18+L24를 추가 후보로 삼고 L20+L24는 production 기준선으로 유지한다. 상위 후보만 10,000-image pilot subset에 full spatial token으로 캐시한다. 동일한 width·token 수·step 예산의 per-reference Resampler를 각 후보에 학습해 controlled ranking과 reconstruction을 비교한다. 최종 tap 변경은 이 2차 결과와 Anima 생성 평가가 함께 개선될 때만 허용한다.

## 실행

```bash
.venv/bin/anima-data --config configs/anima500k-human.yaml stylenet-prepare
.venv/bin/anima-data --config configs/anima500k-human.yaml stylenet-extract
.venv/bin/anima-data --config configs/anima500k-human.yaml stylenet-evaluate
```

세 단계를 연속 실행하려면 `stylenet-benchmark`를 사용한다. Production cache 추출과 동시에 실행하지 않고 C-RADIO GPU 작업이 끝난 뒤 시작한다.
