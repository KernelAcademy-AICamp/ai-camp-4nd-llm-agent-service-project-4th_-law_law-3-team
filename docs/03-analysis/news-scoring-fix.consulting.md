# 외부 컨설팅 보고서 - News Article Scoring Fix

## 검증 대상
- `docs/01-plan/features/news-scoring-fix.plan.md`
- `docs/03-analysis/news-scoring-fix.redteam.md`

## 검증 도구
- Codex CLI (External Consultant, gpt-5.3-codex)

---

## 1. 격차 분석 (현재 vs 업계 최고)

| 영역 | 현재 상태 | 업계 최고 수준 | 핵심 격차 |
|---|---|---|---|
| 점수 신뢰성 | recency만 유효 | 다중 신호 안정 결합 | 랭킹 신호 3축 중 2축 미작동 |
| 법률 신호 | 응답 루트에만 RAG 결과 | 기사 단위 법률 엔티티/쟁점 태깅 | 스키마/파이프라인 단위 불일치 |
| 관련도 | 한국어 복합어 substring 의존 | 형태소·토큰·구문 기반 하이브리드 매칭 | 한국어 특성 미반영 |
| 아키텍처 | RAG 결과와 스코어러 결합도 높음 | 수집-풍부화-스코어링 분리 | 확장/테스트/교체 비용 큼 |
| 운영/평가 | 수동 대응 중심 | 오프라인 평가셋 + 온라인 A/B + 드리프트 감시 | 품질 가시성 부족 |

**진단 요약**: "버그 수정"을 넘어 "신호 주입 경로와 점수 계약(데이터 계약) 부재"가 본질

---

## 2. 기술 고도화 제안

1. 기사 단위 Enrichment 계약 확정
2. 점수 계산 파이프라인 분리 (ingest → enrich → score → rank)
3. Legal score 변별력 보강 (기사별 top-k 법률 매핑 + confidence 가중치)
4. Relevance 개선: exact + token + phrase proximity 혼합 점수
5. 회귀 방지: "relevance/legal 전부 0" 가드레일 알람

---

## 3. UX 개선 제안

1. 랭킹 근거 설명 (Explainability UI) - 기여도 배지/툴팁
2. 정렬 옵션 다변화 (종합, 최신순, 법률 관련도순, 쟁점 집중순)
3. 피드백 루프 ("관련도 낮음 신고" 액션)

---

## 4. AI/ML 고도화 제안

1. Semantic Relevance 단계적 도입 (lexical + 임베딩 유사도)
2. 한국어 법률 도메인 임베딩 최적화
3. 라벨링/평가 체계 (골든셋, NDCG@10, MRR)
4. 모델 거버넌스 (버전 분리 관리)

---

## 5. 비즈니스 전략 제안

1. "근거 제시형 법률 인텔리전스" 포지셔닝
2. 세그먼트별 상품화 (개인/전문가)
3. KPI 재정의 (기술 + 비즈니스)
4. 90일 로드맵: 버그 수정 → 하이브리드 relevance → semantic relevance A/B
