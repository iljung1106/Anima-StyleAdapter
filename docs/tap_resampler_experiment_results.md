# C-RADIO tap / per-reference Resampler pilot 결과

- 실행일: 2026-08-10
- 데이터: train 작가 1,000명 × 10장
- 작가 분할: meta-train 800 / validation 100 / meta-test 100
- 공통 병목: reference당 16 tokens, width 512
- 공통 복원 목표: C-RADIO block 8/16/24 spatial features
- 결론: **block 20 + block 24 spatial tokens, SigLIP global token 없음**

## Validation 결과

Top-1은 학습에 사용하지 않은 validation 작가 100명의 고정 query 200장에 대한 episodic prototype 분류 결과다. `Rec.`은 block 8/16/24 reconstruction cosine의 평균이다.

| 입력 특징 | 1-ref | 2-ref | 4-ref | 8-ref | 평균 Top-1 | Rec. |
|---|---:|---:|---:|---:|---:|---:|
| **L20 + L24 spatial** | **28.0%** | **30.0%** | **37.5%** | **41.5%** | **34.25%** | 0.6750 |
| L8 + L16 + L24 spatial | 25.0% | 26.5% | 32.5% | 37.5% | 30.37% | 0.6530 |
| L20 + L24 + final SigLIP visual | 23.0% | 23.5% | 30.5% | 34.5% | 27.88% | 0.6721 |
| L12 + L24 + final SigLIP visual | 22.5% | 26.0% | 27.0% | 32.5% | 27.00% | 0.6749 |
| L24 spatial | 19.0% | 26.0% | 30.0% | 31.5% | 26.62% | 0.6648 |
| L12 + L24 spatial | 22.0% | 24.0% | 29.0% | 29.5% | 26.12% | 0.6524 |
| L20 + L24 + native SigLIP CLS | 22.0% | 21.0% | 29.5% | 31.0% | 25.87% | **0.6781** |
| L12 + L20 + L24 spatial | 18.0% | 21.5% | 27.0% | 33.5% | 25.00% | 0.6525 |
| L12 + L24 + native SigLIP CLS | 19.5% | 20.5% | 28.0% | 31.0% | 24.75% | 0.6516 |
| L8 + L16 + L24 + native SigLIP CLS | 16.0% | 22.0% | 24.0% | 26.5% | 22.12% | 0.6582 |

L20+L24 spatial은 차점 spatial 구성보다 평균 Top-1이 3.88%p 높고 복원도 높다. Native SigLIP CLS는 복원을 0.0031 높이는 대신 prototype 성능을 8.38%p 낮췄다. 이 작은 복원 차이는 global token과 추가 구조를 유지할 근거로 충분하지 않다. Final SigLIP visual 역시 동일한 L20+L24 통제에서 분류와 복원을 모두 개선하지 못했다.

선택 모델의 학습된 tap mixture는 L20 52.24%, L24 47.76%였다. 두 tap이 모두 실질적으로 사용됐으며 하나가 사실상 비활성화된 결과가 아니다.

## 최종 meta-test

Validation으로 L20+L24 spatial을 선택한 뒤, 보존한 meta-test 작가 100명·1,000장을 한 번 평가했다.

| 지표 | 1-ref | 2-ref | 4-ref | 8-ref | 평균 |
|---|---:|---:|---:|---:|---:|
| Prototype Top-1 | 25.5% | 38.0% | 46.0% | 52.0% | 40.38% |
| MRR | 0.4050 | 0.5109 | 0.5916 | 0.6429 | 0.5376 |

Meta-test reconstruction cosine은 L8 0.6498, L16 0.6551, L24 0.7242였다. Reference 수가 늘어날 때 prototype 성능이 일관되게 상승하므로 multi-reference 입력으로 확장할 근거도 확인됐다.

## 처리 효율과 저장량

- 5개 spatial tap, 각 native CLS와 final SigLIP visual을 포함한 10,000장 pilot cache: 83.35 GB
- 선택된 spatial tap 2개만 같은 정밀도로 캐싱할 경우: pilot 약 33 GB, 150,000장 환산 약 500 GB
- 초기 동기식 학습 loader: 약 6~8초/step, GPU 대부분 유휴
- bounded prefetch 적용 후: 약 0.15~0.30초/step, data wait 약 0.001~0.01초
- validation evaluator에도 동일한 prefetch를 적용해 125 batch를 약 8~12초에 처리

Prefetch episode는 `(seed, step)`으로 직접 결정되므로 queue가 먼저 진행되어도 checkpoint 재개와 후보 간 batch 공정성이 유지된다.

## 다음 단계

전체 데이터 특징 캐시는 C-RADIO block 20과 24 spatial token만 FP16/BF16으로 생성한다. 첫 공통 style encoder에는 SigLIP CLS와 final visual embedding을 넣지 않는다. 이 결정은 현재 pilot의 작가 prototype·feature reconstruction 목적에 대한 것이며, 이후 실제 Anima 생성 평가에서 global 색채나 구도 정보 부족이 관찰될 때만 SigLIP 계열을 별도 ablation으로 다시 검토한다.

이번 결과는 단일 seed와 작가당 10장인 pilot이다. 따라서 정확한 절대 성능보다 후보 간 큰 차이와 비용 방향을 선택 근거로 사용한다.
