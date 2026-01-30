---
name: legal-rag-experiment-tracking
description: 한국 법령/판례 RAG 시스템의 검색 성능 실험을 체계적으로 추적하고 기록하기 위한 스킬입니다.
---

# Legal RAG Experiment Tracking Skill

## Overview
한국 법령/판례 RAG 시스템의 검색 성능 실험을 체계적으로 추적하고 기록하기 위한 스킬입니다.

## 실험 기록 템플릿

### 1. 실험 메타데이터
```yaml
experiment_id: EXP-YYYYMMDD-NNN
date: YYYY-MM-DD
experimenter: [이름]
objective: [실험 목적 - 한 문장]
hypothesis: [가설 - 무엇을 검증하려 하는가]
status: [planned | running | completed | failed]
```

### 2. 시스템 구성
```yaml
vector_db:
  type: [LanceDB | Qdrant | Other]
  version: x.x.x
  index_type: [HNSW | IVF | etc.]
  distance_metric: [cosine | euclidean | dot_product]

embedding:
  model: [text-embedding-3-small | multilingual-e5-large | etc.]
  dimension: [1536 | 768 | etc.]
  provider: [OpenAI | HuggingFace | etc.]

chunking:
  strategy: [fixed | semantic | recursive | by_article]
  chunk_size: [tokens or characters]
  overlap: [tokens or characters]
  
graph_db: # Optional
  type: [Neo4j | FalkorDB | PostgreSQL+AGE | None]
  version: x.x.x
  
infrastructure:
  cloud: [OCI | AWS | Local]
  instance_type: [VM.Standard.A1.Flex | etc.]
  memory_gb: NN
  storage_gb: NN
```

### 3. 데이터셋 정보
```yaml
dataset:
  total_documents: NNN
  total_chunks: NNN
  source: [법제처 OpenAPI | 대법원 | etc.]
  data_types:
    - 법령: NNN건
    - 판례: NNN건
    - 행정해석: NNN건
  date_range: YYYY-MM-DD ~ YYYY-MM-DD
```

### 4. 평가 쿼리셋

#### 쿼리 유형별 분류
| 유형 | 설명 | 예시 | 난이도 |
|------|------|------|--------|
| 단순조회 | 특정 조문 직접 검색 | "민법 제750조 내용" | Easy |
| 개념검색 | 법적 개념으로 검색 | "불법행위 손해배상 요건" | Medium |
| 비교검색 | 여러 법령 비교 | "상법과 민법의 대리 규정 차이" | Hard |
| 참조추적 | 법령 간 참조 관계 | "민법 제750조를 인용한 판례" | Hard |
| 시간검색 | 개정 이력 관련 | "2020년 이후 개정된 근로기준법" | Medium |
| 복합검색 | 여러 조건 결합 | "대법원 판례 중 계약해제 관련 최근 5년" | Hard |

#### 평가 쿼리 템플릿
```yaml
query_id: Q-NNN
query_text: "검색 쿼리"
query_type: [단순조회 | 개념검색 | 비교검색 | 참조추적 | 시간검색 | 복합검색]
expected_results:
  - doc_id: "법령ID_조문번호"
    relevance: [highly_relevant | relevant | partially_relevant]
  - doc_id: "판례일련번호"
    relevance: [highly_relevant | relevant | partially_relevant]
notes: "평가 시 주의사항"
```

### 5. 성능 지표

#### 검색 품질 지표
| 지표 | 설명 | 목표값 |
|------|------|--------|
| Recall@K | 상위 K개 중 관련 문서 비율 | ≥ 0.8 |
| Precision@K | 상위 K개의 정확도 | ≥ 0.7 |
| MRR (Mean Reciprocal Rank) | 첫 번째 관련 문서 순위 | ≥ 0.7 |
| NDCG@K | 순위 가중 관련성 점수 | ≥ 0.75 |
| Hit Rate | 관련 문서 1개 이상 검색 비율 | ≥ 0.9 |

#### 시스템 성능 지표
| 지표 | 설명 | 목표값 |
|------|------|--------|
| Latency P50 | 50번째 백분위수 응답시간 | < 200ms |
| Latency P95 | 95번째 백분위수 응답시간 | < 500ms |
| Latency P99 | 99번째 백분위수 응답시간 | < 1000ms |
| Throughput | 초당 처리 쿼리 수 | ≥ 10 QPS |

#### 비용 지표
| 지표 | 설명 | 월 예산 |
|------|------|---------|
| Embedding Cost | 임베딩 생성 비용 | $XX |
| Inference Cost | LLM 추론 비용 | $XX |
| Infrastructure Cost | 인프라 비용 | $XX |
| Total Cost | 총 운영 비용 | ≤ $100 |

##### 비용 계산 가이드
```python
# OpenAI text-embedding-3-small 기준
embedding_cost_per_1m_tokens = 0.02  # USD

# 예시: 76만 청크, 평균 500 토큰
total_tokens = 760000 * 500
initial_embedding_cost = (total_tokens / 1_000_000) * 0.02

# 쿼리당 비용 (평균 50 토큰)
query_embedding_cost = (50 / 1_000_000) * 0.02

# LLM 비용 (Claude 3.5 Sonnet 기준)
input_cost_per_1m = 3.00   # USD
output_cost_per_1m = 15.00  # USD
```

### 6. 실험 결과 기록

```yaml
results:
  search_quality:
    recall_at_5: 0.XX
    recall_at_10: 0.XX
    precision_at_5: 0.XX
    precision_at_10: 0.XX
    mrr: 0.XX
    ndcg_at_10: 0.XX
    hit_rate: 0.XX
    
  system_performance:
    latency_p50_ms: NN
    latency_p95_ms: NN
    latency_p99_ms: NN
    throughput_qps: NN
    
  cost:
    embedding_initial_usd: NN.NN
    embedding_monthly_usd: NN.NN
    llm_monthly_usd: NN.NN
    infra_monthly_usd: NN.NN
    total_monthly_usd: NN.NN

  failure_analysis:
    - query_id: Q-NNN
      issue: "검색 실패 원인"
      category: [chunking | embedding | retrieval | relevance]
```

### 7. 비교 실험 요약

| 실험 ID | Vector DB | Embedding | Chunk Size | Recall@10 | P95 Latency | Monthly Cost |
|---------|-----------|-----------|------------|-----------|-------------|--------------|
| EXP-001 | LanceDB   | ada-002   | 512        | 0.75      | 180ms       | $45          |
| EXP-002 | Qdrant    | e5-large  | 256        | 0.82      | 220ms       | $55          |
| EXP-003 | LanceDB   | 3-small   | 조문단위   | 0.88      | 150ms       | $40          |

## 팀 협업 가이드

### Notion 기록
- 각 실험마다 Notion 페이지 생성
- 실험 요약과 주요 인사이트 기록
- 스크린샷, 그래프 등 시각 자료 첨부

### Git 기록
```
experiments/
├── EXP-YYYYMMDD-NNN/
│   ├── config.yaml        # 실험 설정
│   ├── results.yaml       # 결과 데이터
│   ├── analysis.md        # 분석 및 인사이트
│   └── queries/           # 테스트 쿼리셋
│       └── eval_queries.yaml
```

### 커밋 메시지 컨벤션
```
[EXP] 실험 ID: 간단한 설명

예시:
[EXP] EXP-20250129-001: LanceDB HNSW 파라미터 튜닝
[EXP] EXP-20250129-002: 조문 단위 청킹 vs 고정 크기 비교
```

## Claude 활용 팁

### 실험 설계 요청 시
```
"[법령 RAG 실험] LanceDB와 Qdrant의 검색 성능을 비교하고 싶어. 
현재 76만 청크, 1536차원 임베딩 사용 중. 실험 설계 도와줘."
```

### 결과 분석 요청 시
```
"[법령 RAG 분석] EXP-001 결과야: Recall@10=0.75, P95=180ms. 
목표는 Recall@10 ≥ 0.85인데, 개선 방안 제안해줘."
```

### 비용 최적화 요청 시
```
"[법령 RAG 비용] 현재 월 $80 쓰고 있는데 $60으로 줄이면서 
성능은 유지하고 싶어. 어떤 옵션이 있을까?"
```
