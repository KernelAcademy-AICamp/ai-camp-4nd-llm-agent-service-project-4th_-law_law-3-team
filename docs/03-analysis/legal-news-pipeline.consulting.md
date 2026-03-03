# 외부 컨설팅 보고서: Legal News Pipeline

> **검증 도구**: Codex CLI (External Consultant, gpt-5.3-codex)
> **검증 대상**: `docs/01-plan/features/legal-news-pipeline.plan.md` (v0.2)
> **검증 일시**: 2026-02-26

---

## 외부 컨설팅 보고서

### 1. 격차 분석 (현재 vs 목표)

- **현재 문서 수준**: 운영 가능한 v1 배치 파이프라인 설계로는 충분히 탄탄함
- **업계 최고 수준 대비 핵심 격차**:
  1. **데이터 범위**: 법률신문+네이버 중심 → 글로벌/전문 DB형 서비스 대비 커버리지 협소
  2. **근거 신뢰성**: 요약은 좋지만 Shepardize/KeyCite류 인용 검증/권위 출처 연결 깊이 부족
  3. **검색 품질**: 벡터 중심 설계, 하이브리드 검색+리랭킹 고도화 미흡
  4. **평가 체계**: 정량 KPI는 있으나 RAG 구성요소별 자동평가(faithfulness, context relevance) 미흡
  5. **운영 성숙도**: cron/systemd는 시작점, 장애복구/SLA/데이터 계보 관점에서 엔터프라이즈급 오케스트레이션 필요
  6. **UX 연결**: "백엔드 전용" 스코프라 사용자 가치 체감 설계 미비

### 2. 기술 고도화 제안

1. **검색 스택 업그레이드**: LanceDB 하이브리드 검색(FTS+Vector)+RRF, Cross-Encoder 리랭킹
2. **오케스트레이션**: cron → Dagster/Prefect 전환 (재시도, 백필, 데이터 계보, 경보 표준화)
3. **권리/컴플라이언스**: 소스별 정책 테이블 분리 (저장 허용범위, 보관기간, 표시 문구 자동화)
4. **관측성**: 수집/정제/요약/임베딩 단계별 SLI 대시보드화

### 3. UX 개선 제안

1. **근거 계층 분리**: 1차 근거(법령/판례) vs 2차 근거(뉴스 요약) 시각적 분리
2. **신뢰 시그널**: 기사 발행일, 최신성 배지, 요약 신뢰도, 원문 링크 노출
3. **탐색 경험**: 이슈 타임라인/주제 클러스터로 "검색→맥락→심화" 흐름
4. **사용자 제어**: "뉴스 근거 포함/제외" 토글, 기간/관할/기관 필터

### 4. AI/ML 고도화 제안

1. **Adaptive RAG**: 질의 유형별 검색 강도 동적 조절 (Self-RAG/CRAG)
2. **GraphRAG 보조계층**: 법률 이슈 지도 (법원-법령-사건-기관 엔터티 그래프)
3. **평가 자동화**: RAGAS/ARES 프레임으로 faithfulness/context relevance 배치 평가
4. **환각 방어**: 법령/판례 정규화 파서 + 내부 DB 대조, 불일치 시 자동 강등/제거
5. **학습 루프**: QA 샘플링 + 전문가 피드백으로 프롬프트/리랭커/청킹 주기적 튜닝

### 5. 비즈니스 전략 제안

1. **상품 계층화**: Free (데일리 브리핑) → Pro (실무 시사점/알림) → Enterprise (API, DMS 연동)
2. **수익화 포인트**: 규제/판결 변화 알림, 산업별 컴플라이언스 리포트
3. **확장 전략**: 뉴스 → 행정예고/국회 의안/규제기관 공지 확장
4. **리스크 관리**: 저작권/라이선스 준수 체계를 제품 기능으로 내장

### 참고 문헌

- Thomson Reuters Westlaw AI-Assisted Research
- LexisNexis Lexis+ AI (Protege)
- Self-RAG (arXiv 2310.11511)
- CRAG (arXiv 2401.15884)
- GraphRAG (arXiv 2404.16130)
- RAGAS (arXiv 2309.15217)
- Anthropic Contextual Retrieval
- LanceDB Hybrid Search docs
