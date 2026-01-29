# RAG 평가 시스템

## 개요

PostgreSQL(판례/법령 원본)과 LanceDB(벡터 청크)를 기반으로 RAG 챗봇 평가 데이터셋 생성 및 Gradio 분석 UI 제공

## 빠른 시작

### 1. Gradio UI 실행

```bash
cd backend
uv run python -m evaluation
# → http://localhost:7860 접속
```

### 2. Solar 자동 질문 생성

```bash
# .env에 UPSTAGE_API_KEY 설정 필요
uv run python -m evaluation.tools.solar_generator \
    --count 30 \
    --output evaluation/datasets/solar_generated.json
```

### 3. 평가 실행

```bash
uv run python -m evaluation.runners.evaluation_runner \
    --dataset evaluation/datasets/eval_dataset_v1.json \
    --experiment-id EXP-20260129-001
```

### 4. 데이터셋 검증

```bash
uv run python -m evaluation.tools.validate_dataset eval_dataset_v1.json
```

## 디렉토리 구조

```
evaluation/
├── __init__.py
├── __main__.py           # CLI 진입점
├── schemas.py            # Pydantic 스키마
├── config.py             # 설정 관리
├── CLAUDE.md             # 이 문서
│
├── datasets/
│   ├── eval_queries.yaml # 평가 쿼리 템플릿
│   └── *.json            # 생성된 데이터셋
│
├── experiments/
│   └── EXP-YYYYMMDD-NNN/
│       ├── config.yaml
│       ├── results.yaml
│       └── analysis.md
│
├── metrics/
│   ├── retrieval.py      # Recall@K, MRR, NDCG
│   ├── generation.py     # Citation Accuracy
│   └── rag.py            # Faithfulness
│
├── tools/
│   ├── dataset_builder.py    # 데이터셋 빌더
│   ├── solar_generator.py    # Solar 자동 생성
│   └── validate_dataset.py   # 검증 도구
│
├── runners/
│   ├── evaluation_runner.py  # 평가 실행기
│   └── experiment_tracker.py # 실험 추적
│
├── ui/
│   ├── gradio_app.py         # 메인 앱
│   ├── dataset_editor.py     # 데이터셋 빌더 탭
│   ├── search_analyzer.py    # 검색 분석 탭
│   └── experiment_viewer.py  # 실험 결과 탭
│
└── reports/
    └── report_generator.py   # 리포트 생성
```

## 주요 스키마

### EvalQuery (평가 쿼리)

```python
{
    "id": "Q-001",
    "question": "임대차 보증금 반환 청구 요건은?",
    "metadata": {
        "category": "민사",
        "query_type": "개념검색",
        "difficulty": "medium"
    },
    "ground_truth": {
        "source_documents": [
            {"doc_id": "76396", "doc_type": "precedent"}
        ],
        "key_points": ["계약 종료 요건", "동시이행"],
        "required_citations": ["민법 제621조"]
    },
    "source": "manual"  # or "solar"
}
```

### ExperimentResult (실험 결과)

```python
{
    "config": {...},
    "metrics": {
        "recall_at_5": 0.72,
        "recall_at_10": 0.84,
        "mrr": 0.68,
        "hit_rate": 0.92,
        "ndcg_at_10": 0.78
    },
    "metrics_by_type": {...},
    "metrics_by_category": {...},
    "query_results": [...]
}
```

## 성능 목표

| 지표 | 목표값 |
|------|--------|
| Recall@5 | ≥ 0.7 |
| Recall@10 | ≥ 0.8 |
| MRR | ≥ 0.7 |
| Hit Rate | ≥ 0.9 |
| NDCG@10 | ≥ 0.75 |
| Latency P50 | < 200ms |
| Latency P95 | < 500ms |

## 쿼리 유형

| 유형 | 설명 | 난이도 | 비율 |
|------|------|--------|------|
| 단순조회 | 특정 조문 직접 검색 | Easy | 20% |
| 개념검색 | 법적 개념으로 검색 | Medium | 30% |
| 비교검색 | 여러 법령 비교 | Hard | 15% |
| 참조추적 | 법령 간 참조 관계 | Hard | 15% |
| 시간검색 | 개정 이력 관련 | Medium | 10% |
| 복합검색 | 여러 조건 결합 | Hard | 10% |

## Gradio UI 기능

### 탭1: 데이터셋 빌더 (역추적 방식)

1. 문서 검색 (키워드로 PostgreSQL 조회)
2. Ground Truth 문서 선택
3. 질문 작성 + 메타데이터 입력
4. 데이터셋 저장

### 탭2: 검색 분석

1. 실시간 벡터 검색 테스트
2. 검색 결과 시각화 (유사도 점수)
3. 문서 상세 보기
4. 결과를 Ground Truth로 추가

### 탭3: 실험 결과

1. 실험 목록 조회
2. 메트릭 대시보드
3. 유형/카테고리별 분석
4. 실험 비교

## 환경 변수

```bash
# .env
UPSTAGE_API_KEY=your_api_key       # Solar 자동 생성용
UPSTAGE_MODEL=solar-pro3-260126    # reasoning 모델 (기본값)
EVAL_GRADIO_HOST=0.0.0.0
EVAL_GRADIO_PORT=7860
```

### Upstage Solar Pro 3 참고사항

- **모델명**: `solar-pro3-260126` (별칭: `solar-pro3`)
- **타입**: Reasoning 모델 (사고 과정 + 응답 생성)
- **응답 구조**: `{"content": "실제 응답", "reasoning": "사고 과정..."}`
- **max_tokens**: 1500 (reasoning 토큰 포함 필요)
- **무료 기간**: 2026년 3월 2일까지

## 개발 가이드

### 새 메트릭 추가

1. `metrics/` 폴더에 함수 추가
2. `metrics/__init__.py`에 export
3. `runners/evaluation_runner.py`에서 호출

### 새 UI 탭 추가

1. `ui/` 폴더에 새 파일 생성
2. `create_xxx_tab()` 함수 정의
3. `ui/gradio_app.py`에서 import 및 탭 추가

## 트러블슈팅

### 임베딩 모델 로드 실패

```bash
# sentence-transformers 재설치
uv pip install --force-reinstall sentence-transformers
```

### Gradio 포트 충돌

```bash
# 다른 포트로 실행
uv run python -m evaluation --port 7861
```

### Solar API 오류

1. `.env`에 `UPSTAGE_API_KEY` 확인
2. API 키 유효성 확인
3. 모델명 확인 (`solar-pro3-260126` 또는 `solar-pro3`)

### Solar API 테스트

```bash
# API 연결 및 질문 생성 테스트
uv run python evaluation/tools/test_solar_api.py
```
