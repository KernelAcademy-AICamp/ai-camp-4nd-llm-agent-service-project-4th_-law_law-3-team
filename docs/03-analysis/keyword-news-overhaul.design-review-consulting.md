# 외부 컨설팅 설계 리뷰 보고서 — 키워드/뉴스 수집 시스템 전면 재설계

> **작성자**: Codex CLI (External Consultant, gpt-5.3-codex)
> **대상**: `docs/02-design/features/keyword-news-overhaul.design.md` v1.1
> **작성일**: 2026-03-01

---

## 1. 격차 분석 (현재 설계 vs 업계 최선)

### 강점
- 수집/중복제거/RRF/스코어링을 단계화해 파이프라인 관심사 분리가 잘 되어 있음
- CircuitBreaker, Provider Abstraction, 2단계 Dedup 도입 방향은 업계 모범 사례와 일치

### 격차
| # | 항목 | 설명 |
|---|------|------|
| 1 | SRP 위반 | `RawTrendItem`이 수집 원시 데이터와 랭킹/ML 파생값을 함께 담아 단일 책임 원칙 약화 |
| 2 | OCP/SoC | `collector`가 동적 쿼리, 소스 오케스트레이션, 병합, 점수 반영까지 담당 → 변경 영향 범위 큼 |
| 3 | 가중치 고정 | 5차원 스코어링 가중치가 고정값이며 실험/튜닝 체계(A/B, feature flag) 미명시 |
| 4 | 하이퍼파라미터 관리 | Dedup 임계치, RRF 가중치, 백오프 구간 등 설정/버전관리/롤백 전략 부재 |
| 5 | API 계약 안정성 | 버전 전략, 호환성, 필드 nullable 정책 미명확 |

---

## 2. 아키텍처 개선 제안

1. `RawTrendItem`을 `CollectedItem`(원시) + `EnrichedItem`(파생) + `RankedItem`(점수)으로 분리 → SRP 강화
2. `collector`를 Orchestrator로 축소, `QueryBuilder`/`SourceFanout`/`MergeStrategy`/`ScoringEngine` 인터페이스 분리 → OCP
3. `ScoringProfile`(카테고리별 가중치 세트) 기반 런타임 교체 가능한 전략 패턴 도입
4. 하이퍼파라미터를 코드 상수 대신 설정 저장소(환경변수+원격 config)로 이동, 버전 태그 포함
5. Provider Abstraction에 표준 계약 추가: timeout, retry budget, partial failure code, freshness timestamp, source confidence
6. 이벤트 기반 비동기 파이프라인(큐/워크플로 엔진)으로 수집-랭킹 느슨한 결합
7. API `/v1` 명시 버저닝 + `response_meta`(scoring_version, dedup_version, source_coverage) 표준화

---

## 3. 누락 항목 및 엣지 케이스

1. 카테고리 미지정/오타/다국어 입력 시 정규화 규칙과 fallback 카테고리 정책 부재
2. UTM/단축링크/모바일 도메인 차이 URL canonicalization 정의 필요
3. YouTube Shorts/일반 영상 혼재 시 engagement 비교 기준(길이 보정, 조회수 시간 보정) 부재
4. 저품질 소스 급증 시 RRF 왜곡 → source-level rate cap + diversity penalty 필요
5. SSE 클라이언트 취소, 재연결, 중복 이벤트 전송(idempotency key) 처리 미정의
6. 임베딩 서비스 장애 시 Dedup 비활성 fallback(lexical dedup only) + 품질 저하 알림
7. `MIN_ITEMS_FOR_DEDUP` 미만 구간의 점수 신뢰도 표시 부재
8. **법률 도메인 설명 가능성**: `score_breakdown`, `why_ranked` 필드 필요

---

## 4. 데이터 파이프라인 고도화 제안

1. Feature Store 도입 → `engagement_velocity`, `source_authority`, `legal_signal` 시계열 관리
2. 오프라인 평가셋 구축 → NDCG@k, MRR, dedup precision/recall을 릴리스 게이트로 운용
3. LLM 스코어링 캐시 키(키워드+카테고리+문서 해시) 기반 재사용 + 배치 추론 비용 절감
4. `convergence_score` → 시간창(1h/6h/24h) 다중 스케일 지표 확장
5. Early Signal: z-score + change-point detection(CUSUM/BOCPD) 병행으로 안정성 향상
6. Human-in-the-loop 검수 큐 → 법률/정책 카테고리 결과 샘플링 검증 + 피드백 루프
7. 데이터 계보(lineage) + 실험 추적(MLflow/W&B 수준) → 재현 가능성 확보

---

## 5. 보안 및 운영 안정성

1. 외부 API 키 KMS/Secret Manager 중앙 관리 + 키 로테이션 + 사용량 한도
2. 수집 데이터 입력 검증(스크립트/HTML/URL 스킴 화이트리스트) + 저장 전 sanitize
3. LLM 입력 전 content safety filter + 출처 신뢰도 기반 가중 감쇄
4. bulkhead(소스별 격리) + timeout budget + global rate limiter (CircuitBreaker 보완)
5. 개인정보/민감정보(댓글, UGC) 마스킹/보존기간/삭제정책 명문화
6. SLI/SLO 정의: 수집 성공률, p95 지연, dedup 오차율, 스코어 드리프트, 소스 커버리지
7. canary + feature flag + 자동 롤백 조건(정확도/지연/오류 임계치) 포함

---

## Agent Team 반영 결정

| # | Codex 제안 | 반영 여부 | 사유 |
|---|-----------|----------|------|
| 1 | RawTrendItem SRP 분리 | **부분 채택** | 현재 3단계 분리는 과도. Dataclass→Pydantic 경계 변환으로 충분 (Gemini 채택 완료) |
| 2 | collector Orchestrator 분리 | **Phase 2 채택** | RRF/Scorer 분리는 Phase 2에서 rrf.py, article_scorer.py로 이미 분리됨 |
| 3 | ScoringProfile 동적 가중치 | **Phase 3 연기** | 초기 고정값 운영, 데이터 축적 후 도입 |
| 4 | 하이퍼파라미터 설정 저장소 | **채택** | 환경변수 기반 설정으로 이동 (config.py Settings 활용) |
| 5 | URL canonicalization | **채택** | URL 정규화 유틸 추가 (UTM 제거, 단축링크 해제) |
| 6 | score_breakdown 설명 가능성 | **채택** | NewsArticle 응답에 score_breakdown dict 추가 |
| 7 | LLM 스코어링 캐시 | **채택** | 키워드+카테고리+문서 해시 기반 캐시 적용 |
| 8 | SSE idempotency | **Phase 2 연기** | 현재 SSE 재연결 시 전체 재시작으로 충분 |
| 9 | Feature Store / CUSUM / Human-in-loop | **Phase 3+ 연기** | 고도화 단계에서 검토 |
| 10 | SLI/SLO / canary / bulkhead | **Phase 3+ 연기** | 운영 안정화 단계에서 도입 |
