# StyleNet C-RADIO layer benchmark 결과

- 데이터: 48,984 images, 12,246 controlled four-way groups
- Query: 같은 캐릭터를 그린 네 작가 중 reference 작가의 Original 선택
- Chance Top-1: 25%
- Reference: query와 다른 group의 Original 1/2/4/8장
- 특징 정밀도: FP16 cache, cosine prototype 평가는 FP32

## 주요 결과

| 표현 | 차원 | 1-ref | 2-ref | 4-ref | 8-ref | 평균 Top-1 | 평균 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| L20+L24 full summary | 4,608 | 58.44% | 62.34% | 65.93% | 67.01% | **63.43%** | 0.7815 |
| **L24 SigLIP CLS** | **1,152** | **58.41%** | **62.29%** | **65.73%** | **66.60%** | **63.26%** | **0.7814** |
| L20+L24 SigLIP CLS | 2,304 | 57.62% | 61.42% | 64.70% | 66.04% | 62.44% | 0.7758 |
| L24 full summary | 2,304 | 57.23% | 61.47% | 64.91% | 65.67% | 62.32% | 0.7748 |
| L20 full summary | 2,304 | 57.08% | 60.57% | 63.75% | 65.43% | 61.71% | 0.7710 |
| L20 DINO CLS | 1,152 | 57.52% | 60.67% | 63.93% | 64.36% | 61.62% | 0.7693 |
| L20+L24 spatial mean | 2,304 | 51.23% | 54.66% | 58.07% | 59.15% | 55.78% | 0.7328 |
| L20 spatial mean | 1,152 | 51.59% | 54.48% | 57.84% | 58.96% | 55.72% | 0.7322 |
| L24 spatial mean | 1,152 | 50.35% | 54.11% | 57.21% | 58.63% | 55.08% | 0.7283 |
| L8+L20+L24 spatial mean/std | 6,912 | 50.17% | 54.38% | 56.88% | 58.24% | 54.92% | 0.7257 |

C-RADIO summary는 `[SigLIP2-g CLS 1,152 | DINOv3-7B CLS 1,152]` 두 teacher slot의 결합이다. L24에서는 SigLIP slot이 평균 63.26%인 반면 DINO slot은 53.96%였다. L20에서는 반대로 DINO slot이 61.62%이고 SigLIP slot은 58.82%였다. 여러 slot을 무조건 합치는 것보다 깊이와 teacher의 조합이 중요하다.

L24 SigLIP CLS 단독은 L20+L24 full summary와 Top-1 차이가 0.17%p에 불과하면서 차원은 1/4이다. L20 SigLIP을 추가하면 오히려 0.82%p 하락했다. 따라서 global 후보는 L24 SigLIP CLS 하나로 제한하는 것이 가장 효율적이다.

모든 주요 표현에서 reference 수가 늘수록 성능이 일관되게 상승했다. L24 SigLIP CLS는 1-reference 58.41%에서 8-reference 66.60%로 8.19%p 향상되어 multi-reference aggregator의 필요성을 다시 확인했다.

## 기존 pilot과의 관계

1,000-artist learned Resampler pilot에서는 L20+L24 full spatial token이 최상이었고 L24 native SigLIP CLS를 추가하면 prototype 성능이 낮아졌다. 이번 결과는 학습 없는 pooled cosine 평가이므로 그 결론을 즉시 뒤집지 않는다.

- StyleNet은 같은 캐릭터의 다른 작가를 negative로 두어 content shortcut을 통제한다.
- Pooled CLS는 작가 구분에는 강하지만 reconstruction에 필요한 위치 정보를 제공하지 않는다.
- 기존 Resampler의 global-token fusion 또는 256차원 projection loss가 CLS를 효과적으로 사용하지 못했을 수 있다.
- Full spatial token은 평균 pooling 결과보다 learned cross-attention에서 더 유용할 수 있다.

따라서 production의 L20/L24 spatial cache 선택은 유지한다. 이미 저장한 L8 mean/std도 비용이 작으므로 삭제하지 않지만 주 입력으로 강제하지 않는다. 전체 데이터의 CLS를 다시 추출하기 전, 기존 10,000-image pilot cache에서 직접 style-token prototype loss와 개선된 global fusion을 사용해 다음 두 변형만 재비교한다.

1. L20+L24 spatial
2. L20+L24 spatial + L24 SigLIP CLS

두 번째 변형이 held-out artist prototype과 reconstruction Pareto를 개선할 때만 전체 150k에 L24 SigLIP CLS를 추가 캐싱한다.
