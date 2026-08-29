# 대형 Reference-QKVO 모델 학습 메모

## 목표

v24의 단순한 reference-conditioned 생성 방식을 확장하되, DiT 블록의 고유성과 전체 네트워크에서 누적되는 효과를 보존한다. Student는 새로 초기화하며 기존 Student 체크포인트를 사용하지 않는다.

## 모델 구조

- 기존 Reader의 시각 표현 능력은 유지하고 end-to-end로 미세조정한다.
- 공통 Generator trunk: width 1024, 16 heads, 4 layers, FFN 4096.
- 28개 DiT 블록을 서로 다른 위치로 취급한다. 블록 그룹화나 블록 간 head 공유를 사용하지 않는다.
- 각 블록에 다음 경로를 독립적으로 둔다.
  - cross-attention Q/K/V/O
  - self-attention Q/K/V/O
- 각 블록의 cross/self attention에 `16 experts × rank 64, top-4`를 둔다. 블록 사이에는 factor나 router를 공유하지 않는다.
- 선택된 expert 내부의 64개 rank 채널은 reference에 의해 signed dense control된다. 경로당 전체 저장 rank는 1024, 샘플당 활성 rank는 최대 256이다.
- cross K/V에도 Q/O와 동일한 expert 내부 channel gate 및 reference gain을 적용한다.
- reference와 현재 activation으로부터 low-rank residual activation을 직접 생성한다. Teacher의 LoRA operator나 projection weight 자체를 추출·적용하지 않는다.
- MLP residual은 목표 범위에서 제외한다. 모델은 원 Teacher의 완전 복제가 아니라 reference 기반 일반화를 목표로 한다.

## 감독과 Loss

- 핵심 감독은 여러 블록을 모두 통과한 **최종 velocity delta**이다.
- block별 K/V/Q/O 값, operator, projection weight를 직접 감독하지 않는다.
- 초기 가중치 기준:
  - normalized Huber: 2.0
  - direction loss: 2.0
  - final retrieval: 0.15~0.25
  - RMS band: 1.0
  - relative common-direction occupancy 제한: 0.25~0.5
- batch mean 제거는 하지 않는다. 공통방향 제한은 서로 달라야 하는 작가/mixture 사이의 상대적 점유율에만 적용한다.
- Flow는 일반 flow-matching MSE와 prior preservation을 사용하며, distillation 샘플보다 강하게 반영한다.

## 학습 데이터와 배치

- 기존 320개 LoRA single 및 pair/triple/amplified/signed mixture bank.
- External CivitAI LoRA single 및 mixture bank.
- Synthetic reference는 동일 스타일에 대해 다양한 content를 포함한다.
- Human reference는 실제 이미지 Flow 학습과 reference 일반화에 사용한다.
- grouped support/query batch를 유지하고, 한 optimizer update에 여러 작가/스타일 그룹을 포함한다.
- timestep stratification, reference subset dropout, tag dropout, group-shared block dropout을 유지한다.

## 1차 학습 커리큘럼

- 총 5,000 optimizer steps의 fresh run으로 구조를 먼저 검증한다.
- distillation : Flow update 비율은 약 5:1.
- Flow loss multiplier는 2.5~3.0부터 시작한다.
- Generator LR: 2e-6, warmup 300 steps, 3.5k까지 유지 후 5k까지 2e-7로 decay.
- Reader LR: 3e-7~5e-7. 초반 적응 후 Generator보다 느리게 움직이게 한다.
- Generator와 Reader EMA: 0.999.
- 샘플: 2k까지 500 step마다, 이후 1k step마다 fixed reference와 기존 8장 panel을 모두 생성하고 W&B에 기록한다.

## 판단 기준

- 최종 velocity의 Huber/direction 개선이 실제 fixed reference 및 panel의 스타일 구분과 함께 나타나야 한다.
- RMS 크기만 증가하거나 공통방향 점유율만 낮아지는 것은 성공으로 보지 않는다.
- routing은 persistent population-aggregated loss-free bias와 약한 overload-only guard만 사용한다. 강한 균등 사용이나 specialization loss는 사용하지 않는다.
