# 외부 컨설팅 설계 리뷰 보고서: Legal News Pipeline Design

> **검증 도구**: Codex CLI (External Consultant, gpt-5.3-codex)
> **검증 대상**: `docs/02-design/features/legal-news-pipeline.design.md` (v0.1.0)
> **검증 일시**: 2026-02-26

---

## 1. 설계 강점

- 파이프라인 단계(수집→정제→PII→요약→저장→청킹) 분리가 명확해 장애 격리와 재처리에 유리
- `BaseNewsSource` 기반 확장 구조는 소스 추가 비용을 낮추는 좋은 설계
- 기존 인프라(`get_chat_model`, `async_session_factory`, 임베딩 함수) 재사용 전략이 현실적
- Fail-safe(부분 실패 허용, fallback, 에러 리포팅) 철학이 운영 관점에서 적절
- 내부 데이터 모델이 타입 중심으로 정리되어 테스트/검증 자동화 기반이 갖춰짐

## 2. 개선 권장사항

### 업계 대비 핵심 격차 (4가지)

1. **데이터 거버넌스**: 스키마 버전 필드만 있고 "계약 기반 검증" 약함 → 입력/출력 데이터 계약(JSON Schema/Pydantic strict) 강제
2. **파이프라인 관측성**: 단계별 통계는 있으나 표준 계측 부족 → OpenTelemetry 기반 run/article 단위 Trace ID 연결
3. **계보/영향도 분석**: 기사→요약→청크→벡터 인덱스의 lineage 미명시 → OpenLineage 이벤트 발행
4. **검색 품질 고도화**: LanceDB 저장은 있으나 하이브리드 검색/리랭킹 미정 → Vector+FTS hybrid + reranker 적용

### 데이터 파이프라인 고도화 우선순위

- **P0**: idempotency 강화, exactly-once upsert 정책, dead-letter queue, 품질 게이트
- **P1**: CDC/증분 처리, 재임베딩 조건 최소화, 비용/성능 최적화(요약 캐시, 배치 임베딩)
- **P2**: 법률 도메인 신뢰도 레이어(참조 정확도 score, 휴먼 리뷰 큐)

## 3. 테스트 전략 제안

| 레이어 | 비율 | 내용 |
|--------|------|------|
| **Unit** | 60% | 정제기/PII/중복제거/청커 deterministic 테스트, 요약 파서 schema validation |
| **Contract** | 20% | 소스별 입력 계약, DB 저장 계약 검증 |
| **Integration** | 15% | httpx mock, DB/LanceDB test container, upsert·재처리·중복 시나리오 |
| **E2E** | 5% | "정상", "소스 장애", "LLM 타임아웃", "부분 실패 후 재실행" 4개 골든 시나리오 |

### 운영 검증

- 매일 샘플 n건 수동 평가(요약 정확성/법령 참조 정확성)
- 주간 drift 리포트
- KPI: 수집 성공률, 요약 성공률, 중복률, 평균 처리시간, 참조 검증 정확도

## 4. 향후 확장 로드맵

| 시기 | 내용 |
|------|------|
| 0~2개월 | 데이터 계약/품질 게이트/재시도 정책 확정, OTel 기본 계측, KPI 대시보드 |
| 2~4개월 | OpenLineage 연동, 하이브리드 검색+리랭킹, 증분 재수집/재임베딩 |
| 4~6개월 | 이벤트 기반 오케스트레이션, 비용 최적화(캐시/배치/우선순위 큐) |
| 6개월+ | 법률 지식그래프(법령-판례-기사 연결), 휴먼 인더루프 검수 워크플로우 |

### 참고 문헌

- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
- OpenLineage: https://openlineage.io/
- dbt Model Contracts: https://docs.getdbt.com/docs/mesh/govern/model-contracts
- LanceDB Hybrid Search: https://lancedb.com/documentation/guides/search/hybrid-search/
