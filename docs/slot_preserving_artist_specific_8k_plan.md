# Slot-preserving artist-specific 8k experiment

## 목적

기존 global-query 모델에서 관찰된 final-slot 붕괴와 약한 공통 출력
shortcut을 함께 제거한다. Frozen Dual-query Resampler와 Frozen Anima는
유지하고 Style Tokenizer만 처음부터 8,000 step 학습한다.

## 모델 변경

- 84개 typed reference memory와 16개 최종 style token은 유지한다.
- 최종 query에는 고정 직교 slot basis와 작은 학습 가능 delta를 사용한다.
- 최종 두 cross-attention 층에서는 query 간 self-attention을 제거한다.
  여러 reference는 memory에서 합쳐지지만 최종 slot identity는 서로 섞이지 않는다.
- 공통 출력 projection 뒤에 slot별 rank-32 delta를 둔다.
- 최종 token RMS를 강제로 정규화하거나 loss로 맞추지 않는다. 초기 RMS만
  약 0.15로 두고 이후 강도는 flow와 teacher residual 목적함수가 결정한다.

## 손실과 커리큘럼

- 기본 rectified-flow loss는 그대로 사용한다.
- attention-map diversity와 reference-centered token diversity는 끈다.
- 8개 대표 Anima block의 실제 frozen K/V projection 뒤에서 slot energy,
  reference energy 및 약한 decorrelation을 측정한다.
- human/synthetic teacher는 서로 독립된 reference domain으로 유지한다.
  동일 content와 timestep의 artist-centered residual을 직접 맞춘다.
- centered student effect가 teacher effect 절대 RMS의 0.30에서 0.80까지
  도달하도록 soft lower bound를 올린다. 과도한 출력은 1.40에서 제한한다.
- 같은 artist teacher residual을 positive로, 같은 probe의 다른 artist를
  hard negative로 삼는 symmetric contrastive와 ranking을 추가한다.
- common-output penalty의 분모를 teacher-relative centered absolute energy가
  보장하는 student RMS로 두어 작은 출력으로 비율만 회피할 수 없게 한다.
- teacher cadence는 step 1--1,000 매 step, 1,001--3,000 매 2 step,
  이후 8,000까지 매 4 step으로 낮춘다.

## 실행 및 선택 기준

- 새 출력 디렉터리와 W&B run을 사용하고 과거 체크포인트를 재사용하지 않는다.
- 250 step마다 validation/checkpoint, 500 step마다 panel sample,
  1,000 step마다 fixed-reference sample과 확장 지표를 기록한다.
- 핵심 선행 조건은 functional slot/reference energy가 하한 위에 있고,
  teacher-relative centered RMS가 상승하며, teacher retrieval과 ranking이
  개선되는 것이다. 그와 동시에 heldout paired-flow improvement와 정성
  샘플이 함께 개선되지 않으면 8,000 step 완료만으로 성공으로 판단하지 않는다.
