---
name: rag-evaluation-workflow
description: RAG 검색 품질 자동 평가 워크플로우. Recall/MRR/NDCG 목표 대비 자동 비교, 성능 저하 진단, 개선 제안. RAG 파이프라인 변경 후 성능 검증, A/B 테스트, 회귀 테스트 시 사용.
---

# RAG Evaluation Workflow Skill

RAG 파이프라인 변경 시 검색 품질을 자동으로 평가하고 목표 대비 결과를 비교하는 워크플로우.

> **참조**: `legal-rag-experiment-tracking/SKILL.md` (실험 추적 템플릿), `Antigravity.md` (성능 목표)

## 1. 성능 목표 (Baseline)

| 지표 | 목표값 | 설명 |
|------|--------|------|
| Recall@5 | >= 0.7 | 상위 5건에 정답 포함 비율 |
| Recall@10 | >= 0.8 | 상위 10건에 정답 포함 비율 |
| MRR | >= 0.7 | 평균 역순위 (첫 정답 위치) |
| Hit Rate | >= 0.9 | 정답이 1건 이상 포함된 쿼리 비율 |
| NDCG@10 | >= 0.75 | 정규화 할인 누적 이득 |

## 2. 평가 워크플로우

### Step 1: 변경 전 베이스라인 측정

```bash
cd backend

# 현재 성능 측정 (변경 전)
uv run python -m evaluation.runners.evaluation_runner \
    --dataset evaluation/datasets/eval_dataset_v1.json \
    --output evaluation/results/baseline_$(date +%Y%m%d).json
```

### Step 2: RAG 파이프라인 변경 적용

변경 대상 파일:
- `backend/app/services/rag/query_rewrite.py` - 쿼리 재작성
- `backend/app/services/rag/retrieval.py` - 벡터 검색
- `backend/app/services/rag/rerank.py` - 재순위화
- `backend/app/services/rag/pipeline.py` - 파이프라인 통합
- `backend/app/services/rag/__init__.py` - 설정/상수

### Step 3: 변경 후 성능 측정

```bash
# 변경 후 성능 측정
uv run python -m evaluation.runners.evaluation_runner \
    --dataset evaluation/datasets/eval_dataset_v1.json \
    --output evaluation/results/experiment_$(date +%Y%m%d).json
```

### Step 4: 비교 분석

```bash
# Gradio UI로 비교
uv run python -m evaluation
# → http://localhost:7860 에서 결과 비교
```

## 3. 성능 비교 판정 기준

### 통과 (PASS)

```
모든 지표가 목표값 이상이면 통과
  Recall@5  >= 0.7  ✅
  Recall@10 >= 0.8  ✅
  MRR       >= 0.7  ✅
  Hit Rate  >= 0.9  ✅
  NDCG@10   >= 0.75 ✅
```

### 회귀 (REGRESSION)

```
하나라도 베이스라인 대비 5% 이상 하락하면 회귀
  예: Recall@10 0.82 → 0.76 (7.3% 하락) → REGRESSION ⚠️
```

### 개선 (IMPROVEMENT)

```
목표 달성 + 베이스라인 대비 5% 이상 향상
  예: MRR 0.68 → 0.75 (10.3% 향상) → IMPROVEMENT ✅
```

## 4. 쿼리 유형별 분석

법률 RAG 시스템은 쿼리 유형에 따라 성능이 다릅니다.

| 쿼리 유형 | 예시 | 주요 지표 |
|----------|------|----------|
| 판례 검색 | "손해배상 판례 알려줘" | Recall, MRR |
| 법령 검색 | "민법 제750조 내용" | Hit Rate, NDCG |
| 혼합 검색 | "교통사고 관련 법령과 판례" | 전체 지표 |
| 구체적 질의 | "2024년 대법원 부동산 판결" | MRR, Hit Rate |
| 추상적 질의 | "이혼할 때 재산분할 어떻게" | Recall@10 |

### 유형별 목표

```yaml
precedent_search:
  recall_at_5: 0.75
  mrr: 0.7

statute_search:
  hit_rate: 0.95
  ndcg_at_10: 0.8

mixed_search:
  recall_at_10: 0.8
  mrr: 0.65
```

## 5. 성능 저하 진단 가이드

### Recall 저하 시

1. **query_rewrite 확인**: 쿼리 변환이 핵심 키워드를 누락하는지
2. **retrieval top_k 확인**: 검색 범위가 충분한지 (기본 20 → 50 시도)
3. **임베딩 모델 확인**: KURE-v1 모델이 올바르게 로드되었는지
4. **LanceDB 데이터 확인**: 인덱스 손상, 데이터 누락 여부

### MRR 저하 시

1. **rerank 모델 확인**: 재순위화 모델이 올바르게 동작하는지
2. **rerank threshold 확인**: 너무 엄격한 threshold로 정답이 필터링되는지
3. **쿼리 확장 확인**: 확장된 쿼리가 노이즈를 추가하는지

### Hit Rate 저하 시

1. **데이터 커버리지 확인**: 해당 법령/판례가 LanceDB에 존재하는지
2. **청킹 전략 확인**: 관련 조문이 하나의 청크에 포함되는지
3. **유사도 거리 확인**: cosine 유사도 분포 확인

## 6. 평가 데이터셋 관리

```bash
# 데이터셋 검증
uv run python -m evaluation.tools.validate_dataset eval_dataset_v1.json

# Solar 기반 자동 질문 생성 (데이터셋 확장)
uv run python -m evaluation.tools.solar_generator --count 30

# 데이터셋 통계
uv run python -m evaluation.tools.validate_dataset eval_dataset_v1.json --stats
```

### 데이터셋 품질 기준

- 최소 50개 이상 쿼리
- 쿼리 유형별 균등 분포 (판례/법령/혼합 각 30% 이상)
- 각 쿼리에 정답 문서 ID 1개 이상 라벨링
- 난이도 분포: 쉬움(30%), 보통(40%), 어려움(30%)

## 7. 결과 보고 형식

```
## RAG 평가 결과 보고

### 실험 정보
- 실험 ID: EXP-YYYYMMDD-NNN
- 변경 내용: [변경 사항 요약]
- 데이터셋: eval_dataset_v1.json (N개 쿼리)

### 성능 비교

| 지표 | 목표 | Baseline | 실험 | 변화 | 판정 |
|------|------|----------|------|------|------|
| Recall@5 | 0.70 | 0.XX | 0.XX | +X.X% | ✅/⚠️ |
| Recall@10 | 0.80 | 0.XX | 0.XX | +X.X% | ✅/⚠️ |
| MRR | 0.70 | 0.XX | 0.XX | +X.X% | ✅/⚠️ |
| Hit Rate | 0.90 | 0.XX | 0.XX | +X.X% | ✅/⚠️ |
| NDCG@10 | 0.75 | 0.XX | 0.XX | +X.X% | ✅/⚠️ |

### 최종 판정: [PASS / REGRESSION / IMPROVEMENT]

### 쿼리별 실패 분석 (상위 5건)
1. ...
2. ...
```

