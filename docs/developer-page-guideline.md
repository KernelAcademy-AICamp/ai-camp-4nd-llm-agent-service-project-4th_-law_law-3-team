# Developer 페이지 가이드라인

> pitch.html의 Investor/Developer 듀얼 레이어 구조 분석 및 Developer 페이지 개선 제안

---

## 1. 현재 구조 요약

pitch.html은 `data-audience` 속성으로 Investor/Developer 모드를 전환합니다.
각 섹션(`<section class="slide">`)마다 `layer--investor`와 `layer--developer` 두 레이어가 존재하며, 선택된 모드에 따라 하나만 표시됩니다.

**전체 섹션 목록 (21개 슬라이드, Investor 뷰 20개 / Developer 뷰 21개):**

| # | ID | Investor 라벨 | Developer 라벨 |
|---|-----|---------------|----------------|
| 1 | s01 | Legal President (Hero) | Tech Stack |
| 2 | s02 | Problem | Design Philosophy |
| 3 | s02b | Vision | Platform Strategy |
| 4 | s03 | Computational Thinking | CT → Implementation Mapping |
| 5 | s06b | Evolution & Refinement | **Iteration Log** |
| 6 | s04 | Solution | Chat Router Architecture |
| 7 | s05 | Demo Preview | Demo Checkpoints |
| 8 | s06 | Live Demo | DevTools 관점 |
| 9 | s07 | Technical Moat | Comparison |
| 10 | s08 | Architecture | Engineering Decisions |
| 11 | s09 | AI System | LangGraph Routing |
| 12 | s10 | ~~Reliability~~ (삭제됨) | Conclusion (Summary & Future Vision) |
| 13 | s11 | Scale & Proof | Ingest Pipeline |
| 14 | s12 | Cost Discipline | Cost Engineering |
| 15 | s12b | Team | Builder Profile |
| **16** | s12c | **Differentiation** | **Competitive Analysis** |
| 17 | s13 | Strategy | Go-to-Market |
| 18 | s13b | Roadmap | Expansion Phases |
| 19 | s13c | Investment & Partnership | Technical Moat Summary |
| **20** | **s13d** | **Retrospective** | **Tech Retrospective** |
| 21 | s14 | Q&A (공통) | Q&A (공통) |

---

## 2. Investor 페이지 — 섹션별 내용 요약

### S01 Hero — "법률 정보의 민주화"
- 타이틀: 법률 정보의 민주화
- 서브텍스트: 검색을 넘어, 사건을 '행동'으로 바꾸는 법률 AI 플랫폼
- 칩: $100 Budget MVP / Used $20 / Forecast $55 / Buffer $25
- CTA: Watch Demo, See Architecture

### S02 Problem — "법률 문제는 '검색'이 아니라 '결정'과 '절차'다"
- 3개 카드 (이미지+텍스트): 이해의 벽 / 근거의 부재 / 실행의 단절
- 각 카드 클릭 시 모달 (법률 용어 DB 현황, RAG 데이터 현황, 주요 기능)

### S02b Vision — "AI는 법조계의 위협이 아니라 '시장 확대'의 열쇠"
- 일반인 → 사건 구조화 → 변호사 → 효율화 흐름
- 핵심 메시지: "파이를 나누는 것이 아니라, 파이 자체를 키우는 혁명"

### S03 Computational Thinking
- CT 4단계: 분해 → 패턴 인식 → 추상화 → 알고리즘 설계
- 각 카드 클릭 시 상세 모달

### S06b Evolution & Refinement — "현장의 목소리로 벼려낸 '법률 대통령'"
- 도입부: 중간 발표 이후 법률 전문가/예비 사용자 피드백을 받아 서비스를 진화시킨 스토리
- Before → Feedback → After 3단 타임라인 구조
  - Before: "법률 정보 검색 AI" / 단일 챗봇 / 변호사 접점 부재
  - Feedback: "다음에 뭘 해야 하는지 모르겠어요" / "사건 파악 시간이 문제" / "출처 없는 AI 불신"
  - After: 정보→행동 / 모듈 오케스트레이션 / 양면 플랫폼
- 클로징: "법률 시장의 구조를 바꾸는 더 큰 그림으로 진화"

### S04 Solution — "Role-based Modular Platform"
- 역할 분기 다이어그램 이미지
- 3개 역할 카드: 일반 사용자 / 공공기관 / 변호사 (각각 모달 연결)

### S05 Demo Preview — "3단계로 보는 AI 법률 상담"
- 3개 카드: 질문 입력 → AI 스트리밍 응답 → 문서 생성
- CTA: 라이브 데모 이동

### S06 Live Demo
- Live Demo 열기 버튼 (새 탭)
- Replay 영상 토글

### S07 Technical Moat — "3가지 기술 해자"
- 3개 카드: Evidence-first RAG / Stateful Orchestration / Streaming UX + Reliability

### S08 Architecture — "시스템 아키텍처"
- 시스템 아키텍처 다이어그램 이미지 1장

### S09 AI System — "RAG 5단계 파이프라인"
- RAG 파이프라인 다이어그램 이미지 1장

### ~~S10 Reliability~~ (삭제됨)
- ~~4개 카드: 품질 자동 폴백 / 스트리밍 안정 / 대규모 데이터 / 세션 보안~~
- **Investor 뷰에서 삭제됨** — CSS `display: none`으로 s10 슬라이드 전체가 investor 모드에서 숨겨짐
- Developer 뷰의 s10 (Conclusion — Summary & Future Vision)은 유지

### S11 Scale & Proof — "실제 데이터로 증명합니다"
- 4개 숫자 카드 (카운트업 애니메이션): 597K+ 문서 / 3.16M+ 벡터 청크 / 425K+ BM25 FTS / 142K+ 코드 라인

### S12 Cost Discipline — "$100 예산으로 만든 MVP"
- 2개 다이어그램 이미지: 예산 워터폴 + Before vs After 비용 절감

### S12b Team — "이 거대한 시스템을 현실로 만든 사람"
- 빌더 프로필 카드: Founder & Solo Builder
- 3개 역량 그룹: Domain & Product / AI Engineering / Full-Stack Development

### S12c Differentiation — "왜 '법률 대통령'이어야 하는가?"
- 서브텍스트: 기존 법률 AI 서비스와 본질적으로 다른 3가지 구조적 차별점
- 3개 카드 (번호 배지 + 경쟁 상대 라벨 + 강점 박스):
  - Card 1 (Cyan): VS 일반 AI 챗봇 → "출처 없는 답변은 없다" → Evidence-first RAG (656K 법률 청크 구조적 인용)
  - Card 2 (Green): VS 법률 검색 포털 → "검색이 아닌 '실행'까지" → Stateful Workflow (9-에이전트 소장 작성·모의법정 실행)
  - Card 3 (Blue): VS 변호사 매칭 플랫폼 → "양면 플랫폼의 구조적 해자" → Two-sided Platform (AI 사건 구조화 → 변호사 요약 전달)
- 클로징: "기술의 깊이 × 도메인 이해 × 플랫폼 구조 — 세 가지가 동시에 갖춰진 법률 AI는 아직 없습니다."

### S13 Strategy — "독점이 아닌 '상생'을 위한 인프라"
- 중앙 허브(LP) + 4개 타겟 카드: 법제처(B2G) / 대한변호사협회 / 대형 로펌 / 법률 학원

### S13b Roadmap — "3단계 확장 전략"
- 계단식 타임라인: 검증(PoC) → 확산(Platform Standard) → 도약(Ecosystem)

### S13c Investment & Partnership — "왜 지금 우리와 함께해야 하는가"
- 좌측: 확보된 기술 자산 (4개 체크리스트)
- 우측: 파트너십 기대 효과 (4개)
- 클로징 메시지

### S13d Retrospective — "프로젝트 회고"
- 서브텍스트: 이번 프로젝트에서 잘한 점과 개선할 점을 솔직하게 돌아봅니다
- 2열 그리드 레이아웃:
  - 좌측 (Green): 잘한 점 — 카드 3개 (제목 + 설명)
  - 우측 (Red): 개선할 점 — 카드 3개 (제목 + 설명)
- 클로징: "솔직한 회고가 다음 단계의 가장 강력한 출발점입니다."
- **현재 플레이스홀더 상태** — 내용은 별도 제공 예정

### S14 Q&A — "Thank You"
- 투자 문의 / PoC 요청 / 파트너십 버튼 (공통)

---

## 3. 현재 Developer 페이지 — 섹션별 현황 및 평가

### S01 Tech Stack ✅ 유지
- **현재**: 기술 스택 배지(Next.js, FastAPI, LangGraph, PostgreSQL, LanceDB) + 핵심 차별점 3줄
- **평가**: 좋음. Hero 슬라이드로서 기술 아이덴티티를 명확히 전달

### S02 Design Philosophy ⚠️ 개선 필요
- **현재**: UX 철학 3원칙 (Evidence-first UX / Workflow-first State / Role-based Entry)
- **문제**: Investor의 "Problem"과 대응이 약함. 왜 이런 철학이 필요한지 기술적 배경이 빠져 있음
- **제안**: "기술적 도전 과제" — 법률 도메인에서 AI를 적용할 때의 기술적 난제 3가지로 변경

### S02b Platform Strategy ⚠️ 개선 필요
- **현재**: 양면 플랫폼 의사코드 (User → CaseStructurizer → Lawyer) 3줄
- **문제**: 내용이 너무 적음. Investor의 풍부한 Vision 섹션과 밀도 차이가 큼
- **제안**: 시스템 수준의 양면 플랫폼 아키텍처 다이어그램 또는 데이터 플로우 추가

### S03 CT → Implementation Mapping ✅ 유지 (보완)
- **현재**: CT 4단계 매핑 테이블 + CaseState 코드 스니펫(타이핑 애니메이션)
- **평가**: 좋음. CT 개념을 실제 코드로 매핑하는 구조가 명확
- **보완**: 테이블에 구체적 파일 경로나 모듈명 추가하면 더 좋음

### S04 Chat Router Architecture ⚠️ 개선 필요
- **현재**: "NAVIGATE 액션 라우팅" 제목 + 코드 블록(타이핑 애니메이션)만 있음
- **문제**: Investor의 3개 역할 카드 + 다이어그램에 비해 콘텐츠가 빈약
- **제안**: Chat Router의 전체 분기 로직 다이어그램 + 9개 에이전트 매핑 테이블 추가

### S05 Demo Checkpoints ⚠️ 개선 필요
- **현재**: "기술 체크포인트" 제목 + 코드 블록(타이핑 애니메이션)만 있음
- **문제**: Investor의 3개 이미지 카드에 비해 내용 부족
- **제안**: 데모에서 확인할 기술 포인트 3개 카드 (SSE 스트리밍 / RAG 출처 / 상태 전환)

### S06 DevTools 관점 ✅ 유지
- **현재**: 4개 불릿 (Network 탭 SSE, 이벤트 타입, LangGraph 상태, 세션 헤더)
- **평가**: 실용적. 데모 시 개발자가 확인할 포인트를 명확히 제시

### S06b Iteration Log ✅ 신규 추가
- **현재**: 피드백 기반 기술 진화 비교 테이블 (피드백 / Before / After) 5행
- **평가**: Investor의 Before→Feedback→After 스토리라인과 대칭. 각 피드백이 어떤 기술적 전환을 유발했는지 명확히 보여줌

### S07 Comparison ✅ 유지
- **현재**: 비교 테이블 (일반 AI 챗봇 vs 법률 대통령) — 출처/상태관리/스트리밍/폴백/검색
- **평가**: 좋음. 기술적 차별점을 한눈에 보여줌

### S08 Engineering Decisions ✅ 유지 (보완)
- **현재**: 3개 카드 (SSE 프록시 분리 / HttpOnly 세션 / asyncio.gather 병렬)
- **평가**: 좋음. 핵심 엔지니어링 결정을 간결하게 전달
- **보완**: 각 결정의 "왜?"를 한 줄 추가하면 더 설득력 있음

### S09 LangGraph Routing ✅ 유지
- **현재**: LangGraph 에이전트 라우팅 맵 다이어그램 이미지
- **평가**: 좋음. Investor의 RAG 파이프라인과 대칭적으로 에이전트 구조를 보여줌

### S10 Conclusion (Summary & Future Vision) ✅ 유지
- **현재**: The Foundation (4개 수치 카드: 2.4M Data / 14x Speed / 9 Agents / $0 API Cost) + The Growth (로드맵 4단계: 2026 Q2~2027) + 클로징 타이핑 애니메이션
- **평가**: Investor의 Reliability 섹션이 삭제되어, Developer 전용 마무리 슬라이드로 기능. 기술 성과 요약 + 미래 비전을 효과적으로 전달
- **참고**: Investor 모드에서는 s10 슬라이드 전체가 `display: none`으로 숨겨짐

### S11 Ingest Pipeline ✅ 유지
- **현재**: 파이프라인 다이어그램 + 사이드 카드 2개 (LanceDB 벡터 / PostgreSQL FTS)
- **평가**: 좋음. 데이터 파이프라인의 기술적 깊이를 잘 보여줌

### S12 Cost Engineering ✅ 유지
- **현재**: 2개 이미지 카드 (Elasticsearch 제거 + 데이터 50GB→15GB 축소)
- **평가**: 좋음. 비용 절감의 기술적 방법을 구체적으로 보여줌

### S12b Builder Profile ⚠️ 개선 필요
- **현재**: JSON 스타일 의사코드 (builder 객체)
- **문제**: Investor의 풍부한 프로필 카드(역량 태그 3그룹)에 비해 내용이 단순
- **제안**: GitHub 스타일 기여도 요약 또는 기술 스택 깊이 레벨 차트 추가

### S12c Competitive Analysis ✅ 신규 추가
- **현재**: 4열 비교 테이블 (비교 항목 / 일반 AI 챗봇 / 법률 검색 포털 / 법률 대통령) × 5행 (출처 제공, 상태 관리, 검색 엔진, 워크플로우, 플랫폼)
- **평가**: Investor의 3개 차별점 카드와 대칭. 경쟁사 대비 기술 스택 우위를 표 형태로 명확히 보여줌

### S13d Tech Retrospective ✅ 신규 추가
- **현재**: 2열 비교 테이블 (잘한 점 Keep / 개선할 점 Improve) — 각 3행, 영역+내용 컬럼
- **평가**: Investor의 기획 회고와 대칭. 기술적 관점의 KPT(Keep/Problem/Try) 형식
- **현재 플레이스홀더 상태** — 내용은 별도 제공 예정

### S13 Go-to-Market ✅ 유지 (보완)
- **현재**: JSON 스타일 의사코드 (strategy 객체)
- **평가**: 괜찮으나 내용이 적음
- **보완**: 각 타겟별 기술 통합 포인트를 카드로 확장

### S13b Expansion Phases ✅ 유지
- **현재**: 3줄 Phase 요약 (PoC → SaaS API → 판결 예측 ML)
- **평가**: 간결하지만 명확

### S13c Technical Moat Summary ✅ 유지
- **현재**: 비교 테이블 (경쟁 우위 vs 구현 상세) 4행
- **평가**: 좋음. 마지막 섹션에서 기술적 해자를 정리

---

## 4. Developer 페이지 개선 제안 — Investor 흐름과 동기화

### 원칙
1. **Investor와 동일한 스토리 흐름** 유지 (Hero → 문제 → 비전 → 해결 → 데모 → 기술 → 증명 → 클로징)
2. **Developer는 "How"에 집중** — Investor가 "What/Why"라면, Developer는 "How/Implementation"
3. **시각적 밀도 균형** — Investor와 유사한 카드/다이어그램 밀도 유지
4. **코드 블록은 핵심만** — 타이핑 애니메이션은 인상적이나, 내용 없는 코드 블록은 오히려 약점

### 섹션별 개선 제안

| # | 섹션 | 현재 상태 | 제안 | 우선순위 |
|---|------|----------|------|---------|
| S01 | Tech Stack | ✅ 유지 | 그대로 유지 | - |
| S02 | Design Philosophy | ⚠️ 개선 | **"Technical Challenges"로 변경** — 법률 도메인 AI 적용 시 3가지 기술적 난제 (검색 품질/상태 관리/출처 신뢰성) | 높음 |
| S02b | Platform Strategy | ⚠️ 개선 | **데이터 플로우 다이어그램 추가** — User Input → NLP → Agent Selection → RAG → Response + Sources 전체 흐름 | 높음 |
| S03 | CT → Implementation | ✅ 보완 | 테이블에 파일 경로 컬럼 추가 | 낮음 |
| S04 | Chat Router | ⚠️ 개선 | **9개 에이전트 라우팅 테이블 + 분기 조건** 추가 (코드 블록만으로는 부족) | 높음 |
| S05 | Demo Checkpoints | ⚠️ 개선 | **3개 기술 체크포인트 카드** (SSE 이벤트 구조 / RAG 출처 형식 / LangGraph 상태 전환) | 중간 |
| S06 | DevTools 관점 | ✅ 유지 | 그대로 유지 | - |
| S06b | Iteration Log | ✅ 신규 | **구현 완료** — 피드백 기반 Before/After 비교 테이블 5행 | - |
| S07 | Comparison | ✅ 유지 | 그대로 유지 | - |
| S08 | Engineering Decisions | ✅ 보완 | 각 카드에 "왜?" 한 줄 추가 | 낮음 |
| S09 | LangGraph Routing | ✅ 유지 | 그대로 유지 | - |
| S10 | Conclusion | ✅ 유지 | Investor Reliability 삭제됨 → Developer 전용 마무리 슬라이드 (The Foundation + The Growth + 클로징) | - |
| S11 | Ingest Pipeline | ✅ 유지 | 그대로 유지 | - |
| S12 | Cost Engineering | ✅ 유지 | 그대로 유지 | - |
| S12b | Builder Profile | ⚠️ 개선 | **기술 스택 깊이 표시** 추가 (각 기술별 구현 규모/복잡도 시각화) | 중간 |
| S12c | Competitive Analysis | ✅ 신규 | **구현 완료** — 4열 비교 테이블 (일반 AI 챗봇 vs 법률 검색 포털 vs 법률 대통령) | - |
| S13d | Tech Retrospective | ✅ 신규 | **구현 완료 (플레이스홀더)** — Keep/Improve 2열 테이블, 내용 별도 제공 예정 | - |
| S13 | Go-to-Market | ✅ 보완 | 각 타겟별 API/통합 포인트 카드로 확장 | 낮음 |
| S13b | Expansion Phases | ✅ 유지 | 그대로 유지 | - |
| S13c | Technical Moat Summary | ✅ 유지 | 그대로 유지 | - |

---

## 5. 상세 개선안

### S02: "Design Philosophy" → "Technical Challenges"

**현재 (삭제 대상):**
```
UX 철학 3원칙
- Evidence-first UX
- Workflow-first State
- Role-based Entry
```

**제안 (새 내용):**
```
기술적 도전 과제 — 법률 AI가 풀어야 할 3가지 난제

카드 1: 검색 품질 (Search Quality)
  "60만 법률 문서에서 정확한 조항을 찾아야 한다"
  → 해결: BM25 + 벡터 하이브리드 검색, MeCab 법률 사전

카드 2: 대화 상태 관리 (State Management)
  "멀티턴 법률 상담에서 맥락을 잃으면 안 된다"
  → 해결: LangGraph Stateful 에이전트, CaseState 모델

카드 3: 출처 신뢰성 (Source Reliability)
  "판례/법령 출처 없는 AI 답변은 위험하다"
  → 해결: Evidence-first RAG, 구조화된 sources 이벤트
```

### S02b: "Platform Strategy" 보강

**유지:** 현재 의사코드

**추가:**
```
전체 데이터 플로우 다이어그램 (ASCII 또는 이미지)

User Input
  ↓
Intent Classification (LangGraph Router)
  ↓
Agent Selection (9개 중 1개)
  ├── case_consultation    → RAG(판례) + 구조화
  ├── small_claims         → 인터뷰 위저드
  ├── lawyer_finder        → 지도 + 필터
  ├── storyboard           → 타임라인 생성
  ├── mock_trial           → Phaser 게임
  └── ... (4개 더)
  ↓
RAG Pipeline (BM25 + Vector → RRF → Rerank)
  ↓
SSE Streaming Response + Sources
```

### S04: "Chat Router Architecture" 보강

**유지:** 코드 블록 타이핑 애니메이션

**추가:**
```
에이전트 라우팅 테이블

| 에이전트 | 트리거 키워드 | 출력 타입 |
|---------|-------------|----------|
| case_consultation | 상담, 법률, 사건 | 텍스트 + 출처 |
| small_claims | 소액소송, 소장 | 인터뷰 위저드 |
| lawyer_finder | 변호사, 찾기 | 지도 네비게이션 |
| storyboard | 타임라인, 증거 | 간트차트 |
| mock_trial | 모의법정, 재판 | 게임 |
| law_search | 법령, 법률검색 | 체계도 |
| law_study | 학습, 공부 | 퀴즈 |
| case_precedent | 판례, 선례 | 검색 결과 |
| legal_news | 뉴스, 소식 | 뉴스 피드 |
```

### S05: "Demo Checkpoints" 보강

**유지:** 코드 블록 타이핑 애니메이션

**추가 (3개 글래스 카드):**
```
카드 1: SSE 이벤트 구조
  data: {"type":"token","content":"..."} — 실시간 토큰
  data: {"type":"sources","items":[...]} — 출처 목록

카드 2: RAG 출처 형식
  판례: 대법원 2023다12345 (유사도 0.87)
  법령: 민법 제750조 (BM25 스코어 4.2)

카드 3: 상태 전환 추적
  LangGraph interrupt → 사용자 입력 대기
  LangGraph resume → 다음 단계 진행
```

### ~~S10: "Decision Cards" 구조화~~ (불필요)

> S10 Investor (Reliability) 섹션이 삭제되었으므로, Developer 측 개선 제안도 더 이상 적용되지 않습니다.
> Developer의 S10은 이미 Conclusion (Summary & Future Vision)으로 구현 완료 상태입니다.

### S12b: "Builder Profile" 보강

**유지:** builder 객체 의사코드

**추가:**
```
시스템 규모 요약 (수치 기반)

| 영역 | 규모 |
|------|------|
| Backend API | 9 모듈, 40+ 엔드포인트 |
| AI 에이전트 | 9 LangGraph 에이전트 |
| 데이터 파이프라인 | 21개 타입 인제스트 |
| 벡터 DB | 3.16M 청크 (LanceDB) |
| BM25 인덱스 | 425K 문서 (PostgreSQL) |
| 법률 사전 | 37K+ MeCab 엔트리 |
| 프론트엔드 | 13 모듈 페이지 |
| 코드 | 142K+ 라인 |
```

---

## 6. 삭제/대체 요약

| 섹션 | 삭제 대상 | 대체 내용 |
|------|----------|----------|
| S02 | "UX 철학 3원칙" 카드 3개 | "기술적 도전 과제" 카드 3개 |
| S10 (Investor) | Reliability 4개 카드 전체 | **삭제됨** — investor 모드에서 s10 슬라이드 숨김 |

S10 Developer (Conclusion)은 그대로 유지됩니다.
나머지는 **유지 + 보강** 방식이므로 기존 콘텐츠를 삭제하지 않습니다.

---

## 7. 구현 우선순위

### Phase 1 (높음) — Investor와 밀도 격차가 큰 섹션
1. S02: Technical Challenges 카드 3개
2. S02b: 데이터 플로우 다이어그램 추가
3. S04: 에이전트 라우팅 테이블 추가

### Phase 2 (중간) — 시각적 개선
4. S05: 기술 체크포인트 카드 3개
5. S12b: 시스템 규모 테이블 추가

### Phase 3 (낮음) — 마무리 보완
7. S03: 파일 경로 컬럼 추가
8. S08: "왜?" 한 줄 추가
9. S13: 타겟별 카드 확장

---

## 8. Developer 페이지 전체 섹션 — 작성 가이드라인 (기술자용)

> **목적**: pitch.html의 Developer 페이지 모든 섹션에 들어갈 내용을 수집하기 위한 가이드라인입니다.
> 기술자 2명이 각자 아래 양식에 맞춰 작성한 뒤, 에이전트 팀이 최종 발표자료를 업데이트합니다.

### 공통 작성 규칙

1. **Developer 페이지 = 기술 관점** — "How"와 "Implementation"에 집중
2. 발표 슬라이드이므로 **간결하게** — 한 항목당 최대 1~2줄
3. **구체적인 수치/기술명/파일 경로**를 포함하면 설득력이 높아집니다
4. 모든 내용은 **1 viewport(화면 1페이지)** 안에 들어가야 합니다 — 과도한 내용은 잘림
5. Investor 페이지가 "What/Why"를 보여주므로, Developer는 같은 주제의 "How"를 보여줍니다
6. 현재 내용이 `✅ 유지`인 섹션도 **보완이 필요한 경우** 수정 내용을 작성해 주세요

### 표시 형식 범례

각 섹션의 현재 HTML 구조에 따라 작성 양식이 다릅니다:

| 형식 | 설명 | 해당 섹션 |
|------|------|----------|
| **배지 + 불릿** | 기술 배지 나열 + 핵심 포인트 목록 | S01 |
| **글래스 카드 N개** | 제목 + 설명 카드 | S02, S08 |
| **의사코드 블록** | `const obj = { ... }` 형태 JSON 스타일 코드 | S02b, S12b, S13, S13b |
| **매핑 테이블** | CT 단계 ↔ 구현 매핑 | S03 |
| **코드 블록 (타이핑)** | 타이핑 애니메이션으로 표시되는 코드 | S04, S05, S10 |
| **불릿 목록** | 확인 항목 리스트 | S06 |
| **비교 테이블** | 기능별 비교 (Before/After 또는 경쟁사 비교) | S06b, S07, S12c, S13c, S13d |
| **다이어그램 이미지** | 이미지 파일 1장 | S09 |
| **이미지 + 사이드 카드** | 다이어그램 + 보조 정보 카드 | S11 |
| **이미지 카드** | 이미지 포함 글래스 카드 | S12 |

---

### 8-1. S01 Tech Stack (Hero)

**Investor 대응**: "Legal President (Hero)" — 법률 정보의 민주화
**Developer 목적**: 기술 스택과 핵심 차별점을 한눈에 전달
**현재 상태**: ✅ 유지 (보완 가능)
**형식**: 배지 + 불릿 3개

#### 작성 양식

**기술 배지 (현재 5개 — 추가/변경 가능):**

| # | 배지 텍스트 |
|---|-----------|
| 1 | *(예: Next.js 14)* |
| 2 | *(예: FastAPI)* |
| 3 | *(예: LangGraph)* |
| 4 | *(예: PostgreSQL 17)* |
| 5 | *(예: LanceDB)* |

**핵심 차별점 (3줄, 각 줄은 한 문장):**

| # | 내용 |
|---|------|
| 1 | *(예: 단일 챗봇이 아닌 **모듈 오케스트레이션** 플랫폼)* |
| 2 | *(예: Elasticsearch 없이 **BM25 + 하이브리드**로 복잡도/비용 최소화)* |
| 3 | *(예: 9개 LangGraph 에이전트가 **Chat Router**로 자동 분기)* |

---

### 8-2. S02 Design Philosophy → Technical Challenges

**Investor 대응**: "Problem" — 법률 문제는 '검색'이 아니라 '결정'과 '절차'다
**Developer 목적**: 법률 도메인에서 AI를 적용할 때의 기술적 난제 3가지
**현재 상태**: ⚠️ 개선 필요 — UX 철학 3원칙 → **기술적 도전 과제**로 변경 제안
**형식**: 글래스 카드 3개 (제목 + 설명)

#### 작성 양식

| # | 카드 제목 | 설명 (1~2줄) |
|---|----------|-------------|
| 1 | *(기술적 난제 1)* | *(왜 어려운지 + 어떻게 해결했는지)* |
| 2 | *(기술적 난제 2)* | *(왜 어려운지 + 어떻게 해결했는지)* |
| 3 | *(기술적 난제 3)* | *(왜 어려운지 + 어떻게 해결했는지)* |

#### 예시

| # | 카드 제목 | 설명 |
|---|----------|------|
| 1 | 검색 품질 (Search Quality) | 60만 법률 문서에서 정확한 조항을 찾아야 한다 → BM25 + 벡터 하이브리드 검색, MeCab 법률 사전 |
| 2 | 대화 상태 관리 (State Management) | 멀티턴 법률 상담에서 맥락을 잃으면 안 된다 → LangGraph Stateful 에이전트, CaseState 모델 |
| 3 | 출처 신뢰성 (Source Reliability) | 판례/법령 출처 없는 AI 답변은 위험하다 → Evidence-first RAG, 구조화된 sources 이벤트 |

---

### 8-3. S02b Platform Strategy

**Investor 대응**: "Vision" — AI는 시장 확대의 열쇠
**Developer 목적**: 양면 플랫폼의 기술 아키텍처 / 데이터 플로우
**현재 상태**: ⚠️ 개선 필요 — 의사코드 3줄만 있어 내용 부족
**형식**: 의사코드 블록 (+ 데이터 플로우 다이어그램 추가 제안)

#### 작성 양식

**의사코드 (현재 유지 + 보강):**

```
// 주석 (한 줄 설명)
흐름1: 입력 → 처리 → 출력
흐름2: 입력 → 처리 → 출력
// 결론 주석
```

**추가 제안 — 전체 데이터 플로우 (텍스트로 작성, 에이전트 팀이 다이어그램으로 변환):**

| # | 단계 | 설명 |
|---|------|------|
| 1 | *(입력)* | *(예: User Input — 자연어 법률 질문)* |
| 2 | *(분류)* | *(예: Intent Classification — LangGraph Router)* |
| 3 | *(에이전트 선택)* | *(예: Agent Selection — 9개 중 1개 자동 분기)* |
| 4 | *(검색)* | *(예: RAG Pipeline — BM25 + Vector → RRF → Rerank)* |
| 5 | *(응답)* | *(예: SSE Streaming Response + Sources)* |

#### 예시

```
// Two-sided platform architecture
User(대화) → CaseStructurizer → StructuredCase
StructuredCase → LawyerMatcher → Lawyer(효율화)
// Result: Market expansion, not displacement
```

---

### 8-4. S03 CT → Implementation Mapping

**Investor 대응**: "Computational Thinking" — 4단계 CT
**Developer 목적**: CT 개념을 실제 코드/모듈로 매핑
**현재 상태**: ✅ 유지 (파일 경로 보완 가능)
**형식**: 매핑 테이블 (4행) + 코드 스니펫 (타이핑 애니메이션)

#### 작성 양식

**매핑 테이블:**

| CT 단계 | 구현 매핑 | 파일 경로 (선택) |
|---------|----------|----------------|
| Decomposition | *(구현 설명)* | *(예: modules/storyboard/)* |
| Pattern Recognition | *(구현 설명)* | *(예: agents/router_node.py)* |
| Abstraction | *(구현 설명)* | *(예: models/case_state.py)* |
| Algorithm Design | *(구현 설명)* | *(예: services/rag_pipeline.py)* |

**코드 스니펫 (CaseState 등 핵심 데이터 모델):**

```
작성 내용: 타이핑 애니메이션으로 표시될 코드 (10줄 이내)
예: CaseState 모델 정의, 에이전트 라우팅 로직 등
```

#### 예시

| CT 단계 | 구현 매핑 | 파일 경로 |
|---------|----------|----------|
| Decomposition | Evidence Upload / Timeline / Wizard | modules/storyboard/, modules/small_claims/ |
| Pattern Recognition | LangGraph 서브그래프 (소액소송/스토리보드/모의법정) | agents/sub_graphs/ |
| Abstraction | CaseState 코드 스니펫 | models/case_state.py |
| Algorithm Design | RAG 5단계 파이프라인 (하이브리드→RRF→리랭킹) | services/rag/ |

---

### 8-5. S04 Chat Router Architecture

**Investor 대응**: "Solution" — 3개 역할 카드 + 다이어그램
**Developer 목적**: 채팅 라우터가 어떻게 9개 에이전트로 분기하는지
**현재 상태**: ⚠️ 개선 필요 — 코드 블록만 있어 내용 부족
**형식**: 코드 블록 (타이핑) + **에이전트 라우팅 테이블 추가 제안**

#### 작성 양식

**에이전트 라우팅 테이블 (추가):**

| 에이전트 | 트리거 키워드/조건 | 출력 타입 |
|---------|-------------------|----------|
| *(에이전트명)* | *(어떤 키워드/의도로 분기?)* | *(텍스트/지도/게임 등)* |
| ... | ... | ... |

**코드 블록 (현재 유지 — 변경 원하면 작성):**

```
작성 내용: NAVIGATE 액션 라우팅 로직 등 (10줄 이내)
```

#### 예시

| 에이전트 | 트리거 키워드 | 출력 타입 |
|---------|-------------|----------|
| case_consultation | 상담, 법률, 사건 | 텍스트 + 출처 |
| small_claims | 소액소송, 소장 | 인터뷰 위저드 |
| lawyer_finder | 변호사, 찾기 | 지도 네비게이션 |
| storyboard | 타임라인, 증거 | 간트차트 |
| mock_trial | 모의법정, 재판 | 게임 |
| law_search | 법령, 법률검색 | 체계도 |
| law_study | 학습, 공부 | 퀴즈 |
| case_precedent | 판례, 선례 | 검색 결과 |
| legal_news | 뉴스, 소식 | 뉴스 피드 |

---

### 8-6. S05 Demo Checkpoints

**Investor 대응**: "Demo Preview" — 3단계로 보는 AI 법률 상담
**Developer 목적**: 데모에서 확인할 기술 체크포인트
**현재 상태**: ⚠️ 개선 필요 — 코드 블록만 있어 내용 부족
**형식**: 코드 블록 (타이핑) + **3개 기술 체크포인트 카드 추가 제안**

#### 작성 양식

**기술 체크포인트 카드 (3개, 각각 제목 + 2줄 이내 설명):**

| # | 카드 제목 | 내용 (기술 포인트 설명) |
|---|----------|----------------------|
| 1 | *(체크포인트 1)* | *(데모에서 확인할 기술적 요소)* |
| 2 | *(체크포인트 2)* | *(데모에서 확인할 기술적 요소)* |
| 3 | *(체크포인트 3)* | *(데모에서 확인할 기술적 요소)* |

#### 예시

| # | 카드 제목 | 내용 |
|---|----------|------|
| 1 | SSE 이벤트 구조 | `data: {"type":"token","content":"..."}` 실시간 토큰 / `data: {"type":"sources","items":[...]}` 출처 목록 |
| 2 | RAG 출처 형식 | 판례: 대법원 2023다12345 (유사도 0.87) / 법령: 민법 제750조 (BM25 스코어 4.2) |
| 3 | 상태 전환 추적 | LangGraph interrupt → 사용자 입력 대기 / resume → 다음 단계 진행 |

---

### 8-7. S06 DevTools 관점

**Investor 대응**: "Live Demo" — 직접 보세요
**Developer 목적**: 데모 중 개발자가 DevTools에서 확인할 포인트
**현재 상태**: ✅ 유지
**형식**: 불릿 목록 (4개)

#### 작성 양식

| # | 확인 항목 (불릿 1줄) |
|---|---------------------|
| 1 | *(예: Network 탭: SSE 스트림 `text/event-stream` 확인)* |
| 2 | *(예: `data: {"type":"token"}` vs `data: {"type":"sources"}` 이벤트 구분)* |
| 3 | *(예: LangGraph `interrupt`/`resume` 상태 전환 관찰)* |
| 4 | *(예: Response Headers: `X-Session-ID` 세션 추적)* |

> 현재 내용 유지 시 빈칸으로 두세요. 수정 원하면 새 내용을 작성하세요.

---

### 8-8. S06b Iteration Log

**Investor 대응**: "Evolution & Refinement" — Before→Feedback→After
**Developer 목적**: 피드백이 어떤 기술적 전환을 유발했는지 보여줌
**현재 상태**: ✅ 구현 완료
**형식**: 비교 테이블 (피드백 / Before / After) — 5행

#### 작성 양식

| # | 피드백 | Before | After |
|---|--------|--------|-------|
| 1 | *("사용자 피드백 원문")* | *(이전 기술 접근)* | *(변경된 기술 접근)* |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |
| 4 | ... | ... | ... |
| 5 | ... | ... | ... |

#### 예시 (현재 구현된 내용)

| # | 피드백 | Before | After |
|---|--------|--------|-------|
| 1 | "출처가 없으면 못 믿어요" | 텍스트만 응답 | Evidence-first RAG — 판례·법령 구조화 인용 |
| 2 | "다음에 뭘 해야 하는지 모르겠어요" | 정보 전달형 Q&A | Stateful Workflow — 소장 작성·사건 구조화까지 |
| 3 | "사건 파악에 시간이 너무 걸려요" | 일반인 전용 챗봇 | 양면 플랫폼 — 구조화 요약본 → 변호사 전달 |
| 4 | "하나의 챗봇이 다 하려니 부정확해요" | 단일 LLM 체인 | LangGraph 9-에이전트 모듈 오케스트레이션 |
| 5 | "법률 용어를 검색해도 엉뚱한 결과" | 벡터 검색 only | BM25 + 벡터 하이브리드 + MeCab 법률 사전 |

> 현재 내용 유지 시 빈칸으로 두세요. 수정 원하면 새 내용을 작성하세요.

---

### 8-9. S07 Comparison

**Investor 대응**: "Technical Moat" — 3가지 기술 해자
**Developer 목적**: 일반 챗봇 vs 법률 대통령 기술 비교
**현재 상태**: ✅ 유지
**형식**: 비교 테이블 (기능 / 일반 AI 챗봇 / 법률 대통령) — 5행

#### 작성 양식

| 기능 | 일반 AI 챗봇 | 법률 대통령 |
|------|-------------|-----------|
| *(비교 항목 1)* | *(약점)* | *(강점 — 볼드 처리됨)* |
| *(비교 항목 2)* | *(약점)* | *(강점)* |
| ... | ... | ... |

#### 예시 (현재 구현된 내용)

| 기능 | 일반 AI 챗봇 | 법률 대통령 |
|------|-------------|-----------|
| 출처 제공 | 없음 / 링크만 | 판례·법령 구조화 인용 |
| 상태 관리 | Stateless | LangGraph Stateful |
| 스트리밍 | 단순 텍스트 | SSE + rAF 최적화 |
| 폴백 | 단일 모델 | ONNX 자동 폴백 |
| 검색 | 벡터 only | BM25 + 벡터 하이브리드 |

> 현재 내용 유지 시 빈칸으로 두세요. 행 추가/변경 원하면 새 내용을 작성하세요.

---

### 8-10. S08 Engineering Decisions

**Investor 대응**: "Architecture" — 시스템 아키텍처 다이어그램
**Developer 목적**: 핵심 엔지니어링 결정 3가지
**현재 상태**: ✅ 유지 (각 카드에 "왜?" 한 줄 보완 제안)
**형식**: 글래스 카드 3개 (제목 + 설명)

#### 작성 양식

| # | 카드 제목 | 설명 (현재) | 왜? (보완 — 선택) |
|---|----------|-----------|-----------------|
| 1 | *(결정 1)* | *(1줄 설명)* | *(이 결정을 내린 이유)* |
| 2 | *(결정 2)* | *(1줄 설명)* | *(이 결정을 내린 이유)* |
| 3 | *(결정 3)* | *(1줄 설명)* | *(이 결정을 내린 이유)* |

#### 예시 (현재 구현된 내용 + 보완 제안)

| # | 카드 제목 | 설명 | 왜? |
|---|----------|------|-----|
| 1 | SSE 프록시 분리 | Next.js API Route로 SSE 버퍼링 우회, 직접 스트리밍 | Next.js 기본 프록시가 SSE를 버퍼링하여 UX 지연 발생 |
| 2 | HttpOnly 세션 | 쿠키 기반 세션으로 XSS 방어, JWT 토큰 노출 방지 | 법률 데이터의 민감성상 클라이언트 토큰 노출 불허 |
| 3 | asyncio.gather 병렬 | BM25 + 벡터 검색을 병렬 실행, 응답 지연 최소화 | 순차 실행 시 2배 지연, 병렬로 50% 응답시간 단축 |

---

### 8-11. S09 LangGraph Routing

**Investor 대응**: "AI System" — RAG 5단계 파이프라인
**Developer 목적**: LangGraph 에이전트 라우팅 맵 시각화
**현재 상태**: ✅ 유지 (다이어그램 이미지)
**형식**: 다이어그램 이미지 1장

#### 작성 양식

> 이 섹션은 **이미지 파일**(A09_LANGGRAPH_ROUTING_MAP.png)로 표시됩니다.
> 이미지 교체가 필요하면 새 다이어그램 이미지를 제공하거나, 아래에 텍스트로 다이어그램 내용을 설명해 주세요.

**이미지 변경이 필요한 경우:**

```
새 다이어그램 설명:
- 노드 목록: (예: Router → case_consultation, small_claims, ...)
- 연결 관계: (예: Router → 9개 에이전트 → Response)
- 추가 요소: (예: interrupt/resume 상태 표시)
```

> 현재 이미지 유지 시 빈칸으로 두세요.

---

### 8-12. S10 Conclusion (Summary & Future Vision)

**Investor 대응**: ~~Reliability~~ (삭제됨 — investor 모드에서 s10 슬라이드 숨김)
**Developer 목적**: 기술 성과 요약 + 미래 비전 로드맵
**현재 상태**: ✅ 유지 — Developer 전용 마무리 슬라이드
**형식**: 2열 레이아웃 (The Foundation 수치 카드 + The Growth 로드맵) + 클로징 타이핑 애니메이션

#### 현재 구현 내용

**The Foundation (좌측 — 4개 수치 카드):**

| 수치 | 라벨 | 설명 |
|------|------|------|
| 2.4M | Data | 21개 기관 법률 빅데이터 |
| 14x | Speed | ONNX + IVF_FLAT 최적화 |
| 9 | Agents | LangGraph 멀티에이전트 |
| $0 | API Cost | 외부 의존성 제거 |

**The Growth (우측 — 로드맵 4단계):**

| 시기 | 마일스톤 | 설명 |
|------|---------|------|
| 2026 Q2 | 멀티모달 확장 | PDF/이미지 분석 모델 |
| 2026 Q3 | 수익 모델 고도화 | 매칭 알고리즘 + 결제 |
| 2026 Q4 | Legal-sLLM | 도메인 특화 파인튜닝 |
| 2027~ | 전국민 법률 비서 | 소송 대행 보조 시스템 |

**클로징 (타이핑 애니메이션):**
> "변호사에게는 **효율적인 업무 파트너**를, 대중에게는 **낮아진 법률 문턱**을 제공합니다."

> 수치 업데이트나 로드맵 변경 원하면 위 테이블을 수정하세요.

---

### 8-13. S11 Ingest Pipeline

**Investor 대응**: "Scale & Proof" — 실제 데이터로 증명
**Developer 목적**: 21개 타입 인제스트 파이프라인 기술 상세
**현재 상태**: ✅ 유지
**형식**: 다이어그램 이미지 + 사이드 카드 2개

#### 작성 양식

**사이드 카드 1 — LanceDB 벡터:**

| # | 내용 (불릿 1줄) |
|---|---------------|
| 1 | *(예: 법령 + 판례 656K 청크)* |
| 2 | *(예: 자치법규 ~240만 청크)* |
| 3 | *(예: KURE-v1 임베딩 (2.3GB))* |

**사이드 카드 2 — PostgreSQL FTS:**

| # | 내용 (불릿 1줄) |
|---|---------------|
| 1 | *(예: 425K search_text BM25)* |
| 2 | *(예: MeCab 법률 사전 (37K+ 엔트리))* |
| 3 | *(예: pg_textsearch 확장)* |

> 현재 내용 유지 시 빈칸으로 두세요. 수치 업데이트나 항목 변경 원하면 새 내용을 작성하세요.

---

### 8-14. S12 Cost Engineering

**Investor 대응**: "Cost Discipline" — $100 예산으로 만든 MVP
**Developer 목적**: 재현 가능한 비용 절감 기술
**현재 상태**: ✅ 유지
**형식**: 이미지 카드 2개 (이미지 + 제목 + 설명)

#### 작성 양식

| # | 카드 제목 | 설명 (1~2줄) |
|---|----------|-------------|
| 1 | *(비용 절감 기술 1)* | *(어떻게 비용을 줄였는지)* |
| 2 | *(비용 절감 기술 2)* | *(어떻게 비용을 줄였는지)* |

#### 예시 (현재 구현된 내용)

| # | 카드 제목 | 설명 |
|---|----------|------|
| 1 | Elasticsearch 제거 | PostgreSQL pg_textsearch + MeCab 법률 사전으로 동등한 BM25 품질 달성 |
| 2 | 데이터 축소 (50GB→15GB) | 구조화 요약 적재로 검색 품질 유지하며 저장 비용 70% 절감 |

> 현재 내용 유지 시 빈칸으로 두세요.

---

### 8-15. S12b Builder Profile

**Investor 대응**: "Team" — 역량 태그 3그룹 + 프로필 카드
**Developer 목적**: 빌더의 기술 역량을 코드 스타일로 표현
**현재 상태**: ⚠️ 개선 필요 — 의사코드만 있어 단순. **시스템 규모 테이블 추가 제안**
**형식**: 의사코드 블록 + (추가) 시스템 규모 테이블

#### 작성 양식

**의사코드 블록 (현재 유지 or 수정):**

```
// 주석
const builder = {
  domain: "...",
  ai: "...",
  backend: "...",
  frontend: "...",
  data: "...",
  budget: "...",
};
```

**추가 제안 — 시스템 규모 요약 테이블:**

| 영역 | 규모 |
|------|------|
| *(영역 1)* | *(수치)* |
| *(영역 2)* | *(수치)* |
| ... | ... |

#### 예시

| 영역 | 규모 |
|------|------|
| Backend API | 9 모듈, 40+ 엔드포인트 |
| AI 에이전트 | 9 LangGraph 에이전트 |
| 데이터 파이프라인 | 21개 타입 인제스트 |
| 벡터 DB | 3.16M 청크 (LanceDB) |
| BM25 인덱스 | 425K 문서 (PostgreSQL) |
| 법률 사전 | 37K+ MeCab 엔트리 |
| 프론트엔드 | 13 모듈 페이지 |
| 코드 | 142K+ 라인 |

---

### 8-16. S12c Competitive Analysis

**Investor 대응**: "Differentiation" — 3가지 구조적 차별점
**Developer 목적**: 경쟁사 대비 기술 스택 우위를 표 형태로 비교
**현재 상태**: ✅ 구현 완료
**형식**: 4열 비교 테이블 (비교 항목 / 일반 AI 챗봇 / 법률 검색 포털 / 법률 대통령) — 5행

#### 작성 양식

| 비교 항목 | 일반 AI 챗봇 | 법률 검색 포털 | 법률 대통령 |
|----------|-------------|-------------|-----------|
| *(항목 1)* | *(약점)* | *(약점)* | *(강점 — 볼드 처리됨)* |
| *(항목 2)* | ... | ... | ... |
| ... | ... | ... | ... |

#### 예시 (현재 구현된 내용)

| 비교 항목 | 일반 AI 챗봇 | 법률 검색 포털 | 법률 대통령 |
|----------|-------------|-------------|-----------|
| 출처 제공 | 없음 / 환각 | 링크 나열 | 판례·법령 구조화 인용 |
| 상태 관리 | Stateless | 검색 세션만 | LangGraph CaseState |
| 검색 엔진 | 벡터 only | 키워드 only | BM25 + 벡터 하이브리드 |
| 워크플로우 | 단발성 Q&A | 자체 검색 | 소장 작성·모의법정 등 실행 |
| 플랫폼 | 일반인 전용 | 전문가 전용 | 양면 (일반인 + 변호사) |

> 현재 내용 유지 시 빈칸으로 두세요. 행 추가/변경 원하면 새 내용을 작성하세요.

---

### 8-17. S13 Go-to-Market

**Investor 대응**: "Strategy" — 상생 인프라 + 4개 타겟
**Developer 목적**: 각 타겟별 기술 통합 포인트
**현재 상태**: ✅ 유지 (카드 확장 보완 제안)
**형식**: 의사코드 블록

#### 작성 양식

```
// 주석
const strategy = {
  타겟1: "기술 통합 설명",
  타겟2: "기술 통합 설명",
  타겟3: "기술 통합 설명",
  타겟4: "기술 통합 설명",
};
```

#### 예시 (현재 구현된 내용)

```
// Integration points per target segment
const strategy = {
  b2g: "법제처 API 연동 → 공공 법률 검색 고도화",
  bar_assoc: "변협 회원 SaaS → 사건 구조화 자동화",
  law_firm: "로펌 API → RAG 파이프라인 화이트라벨",
  academy: "교육 플랫폼 → 모의법정 + 학습 콘텐츠",
};
```

> 현재 내용 유지 시 빈칸으로 두세요. 타겟 변경이나 통합 포인트 수정 원하면 새 내용을 작성하세요.

---

### 8-18. S13b Expansion Phases

**Investor 대응**: "Roadmap" — 3단계 확장 전략 (계단식 타임라인)
**Developer 목적**: 기술 로드맵 3단계
**현재 상태**: ✅ 유지
**형식**: 의사코드 블록 (3줄)

#### 작성 양식

```
// 주석
Phase 1: "기술 마일스톤 1"
Phase 2: "기술 마일스톤 2"
Phase 3: "기술 마일스톤 3"
```

#### 예시 (현재 구현된 내용)

```
// L-Infra Vision — technical milestones
Phase 1: "PoC → 베타 피드백 → RAG 품질 Recall@10 ≥ 0.8"
Phase 2: "SaaS API → 변협/로펌 연동 → 법제처 파트너십"
Phase 3: "판결 예측 ML → 오픈 API → 다국어 K-Law"
```

> 현재 내용 유지 시 빈칸으로 두세요.

---

### 8-19. S13c Technical Moat Summary

**Investor 대응**: "Investment & Partnership" — 기술 자산 + 파트너십 효과
**Developer 목적**: 기술 해자를 경쟁 우위 × 구현 상세로 정리
**현재 상태**: ✅ 유지
**형식**: 비교 테이블 (경쟁 우위 / 구현 상세) — 4행

#### 작성 양식

| 경쟁 우위 | 구현 상세 |
|----------|----------|
| *(기술 해자 1)* | *(구체적 수치/기술 포함)* |
| *(기술 해자 2)* | ... |
| *(기술 해자 3)* | ... |
| *(기술 해자 4)* | ... |

#### 예시 (현재 구현된 내용)

| 경쟁 우위 | 구현 상세 |
|----------|----------|
| Evidence-first RAG | 656K 청크 하이브리드 검색 · Recall@10 ≥ 0.8 |
| Stateful Orchestration | LangGraph 9-에이전트 · CaseState 기반 |
| 도메인 최적화 | MeCab 37K+ 법률 사전 · ONNX 리랭커 |
| 비용 효율 | $20 MVP · ES 제거 · 데이터 70% 축소 |

> 현재 내용 유지 시 빈칸으로 두세요. 행 추가/변경 원하면 새 내용을 작성하세요.

---

### 8-20. S13d Tech Retrospective (기술 회고)

**Investor 대응**: "Retrospective" — 기획 관점 회고 (잘한 점/개선할 점)
**Developer 목적**: 기술 관점의 회고 (Keep/Improve)
**현재 상태**: ✅ 플레이스홀더 — **내용 작성 필수**
**형식**: 2열 비교 테이블 (잘한 점 Keep / 개선할 점 Improve) — 각 3행

#### 작성 양식

**잘한 점 — Keep (3개):**

| # | 영역 | 내용 |
|---|------|------|
| 1 | *(기술 영역)* | *(구체적 성과 또는 좋은 결정)* |
| 2 | *(기술 영역)* | *(구체적 성과 또는 좋은 결정)* |
| 3 | *(기술 영역)* | *(구체적 성과 또는 좋은 결정)* |

**개선할 점 — Improve (3개):**

| # | 영역 | 내용 |
|---|------|------|
| 1 | *(기술 영역)* | *(구체적 문제점 또는 아쉬운 점)* |
| 2 | *(기술 영역)* | *(구체적 문제점 또는 아쉬운 점)* |
| 3 | *(기술 영역)* | *(구체적 문제점 또는 아쉬운 점)* |

#### 예시

**잘한 점 (Keep):**

| # | 영역 | 내용 |
|---|------|------|
| 1 | RAG 파이프라인 | BM25 + 벡터 하이브리드 검색으로 Recall@10 ≥ 0.8 달성, ES 없이 PostgreSQL만으로 동등 품질 확보 |
| 2 | 모듈 아키텍처 | 9개 에이전트를 독립 모듈로 설계, 새 에이전트 추가 시 라우터 수정 없이 플러그인 가능 |
| 3 | 비용 최적화 | ONNX INT8 리랭커로 GPU 없이 추론, 데이터 50GB→15GB 축소로 인프라 비용 70% 절감 |

**개선할 점 (Improve):**

| # | 영역 | 내용 |
|---|------|------|
| 1 | 테스트 커버리지 | 단위/통합 테스트 부족, LangGraph 에이전트 간 상태 전파 검증이 수동 의존 |
| 2 | CI/CD 파이프라인 | 자동 배포 미구축, 수동 배포로 인한 휴먼 에러 리스크 존재 |
| 3 | 모니터링/로깅 | 프로덕션 에러 추적/성능 모니터링 체계 부재, Sentry/Prometheus 도입 필요 |

---

### 8-21. S13d Investor 회고 (기획 관점) — 작성 양식

> **참고**: 이 섹션은 Investor 페이지에 표시됩니다. 기획/비즈니스/프로덕트 관점에서 작성해 주세요.

**잘한 점 (3개):**

| # | 제목 | 설명 |
|---|------|------|
| 1 | *(제목을 입력하세요)* | *(1~2줄 설명)* |
| 2 | *(제목을 입력하세요)* | *(1~2줄 설명)* |
| 3 | *(제목을 입력하세요)* | *(1~2줄 설명)* |

**개선할 점 (3개):**

| # | 제목 | 설명 |
|---|------|------|
| 1 | *(제목을 입력하세요)* | *(1~2줄 설명)* |
| 2 | *(제목을 입력하세요)* | *(1~2줄 설명)* |
| 3 | *(제목을 입력하세요)* | *(1~2줄 설명)* |

#### 예시

**잘한 점:**

| # | 제목 | 설명 |
|---|------|------|
| 1 | 사용자 중심 피드백 루프 | 중간 발표 후 피드백을 즉시 반영, "검색 AI → 실행 플랫폼"으로 피봇 성공 |
| 2 | 양면 플랫폼 기획 | 일반인+변호사 동시 연결 구조를 초기부터 설계, 시장 확대 모델 확보 |
| 3 | $100 예산 규율 | $20만 사용하고 핵심 기능 모두 구현 |

**개선할 점:**

| # | 제목 | 설명 |
|---|------|------|
| 1 | 사용자 테스트 부족 | 법률 비전문가 대상 유저빌리티 테스트 미수행 |
| 2 | 수익 모델 구체화 미흡 | SaaS 요금제, API 과금 체계 미확정 |
| 3 | 경쟁사 심층 분석 부족 | 기존 서비스와의 정량적 비교 부족 |

---

### 8-22. 작성 완료 후 워크플로우

```
1. 기술자 2명이 위 양식(8-1 ~ 8-21)을 각자 작성
   - 현재 내용 유지하려는 섹션은 빈칸으로 두세요
   - 수정/보완하려는 섹션만 새 내용을 작성하세요
   - ⚠️ 표시된 섹션(S02, S02b, S04, S05, S10, S12b)은 개선 제안이 있으니 적극 검토해 주세요
2. 작성된 내용을 이 문서(developer-page-guideline.md)의 양식에 채워서 제출
3. 에이전트 팀이 제출된 내용을 pitch.html 각 섹션에 반영
4. 최종 발표자료 업데이트 완료
```

### 8-23. pitch.html 반영 시 HTML 구조 참고

**글래스 카드 (S02, S08 등):**
```html
<div class="glass-card reveal">
  <h3 style="font-size: 18px; color: var(--accent-cyan);">카드 제목</h3>
  <p class="subtext" style="font-size: 21px;">카드 설명</p>
</div>
```

**비교 테이블 (S06b, S07, S12c, S13c 등):**
```html
<table class="compare-table reveal">
  <thead><tr><th>열1</th><th class="highlight-cell">열2(강조)</th></tr></thead>
  <tbody><tr><td>데이터</td><td class="highlight-cell">강조 데이터</td></tr></tbody>
</table>
```

**의사코드 블록 (S02b, S12b, S13, S13b 등):**
```html
<div class="glass-card reveal" style="max-width: 880px; margin: 0 auto; padding: 2rem;">
  <div style="font-family: var(--font-mono); font-size: 15px; color: var(--accent-cyan); line-height: 2;">
    <span style="color: var(--subtext);">// 주석</span><br>
    코드 내용<br>
  </div>
</div>
```

**회고 테이블 (S13d Developer):**
```html
<table class="compare-table" style="width: 100%;">
  <thead><tr><th style="width: 30%;">영역</th><th>내용</th></tr></thead>
  <tbody><tr><td>영역명</td><td class="highlight-cell">내용</td></tr></tbody>
</table>
```

**회고 카드 (S13d Investor):**
```html
잘한 점 (녹색 테두리 카드):
  <p style="font-weight: 600;">제목</p>
  <p style="font-size: 12px; color: var(--subtext);">설명</p>

개선할 점 (빨간색 테두리 카드):
  동일 구조
```
