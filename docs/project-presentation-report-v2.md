---
title: "‘법률 대통령’ 최종 IR Web Deck — 웹페이지 개발 기획서 v5"
purpose: "30분 최종 발표(투자 발표회 성격) + 개발자 청중 설득 + 라이브 데모"
format: "Apple-like Scrollytelling + Scroll-Snap Web Deck (단일 HTML/CSS/JS 파일)"
brand: "법률 대통령"
date: "2026-03-04"
---

> 이 문서는 기존 Web Deck 기획서(v3~v4)의 **발표 구조/콘텐츠**를 유지하면서,  
> 사용자가 추가로 제공한 개발 가이드(다크 모드 베이스, 단일 파일 구현, 듀얼 레이어 X-ray 전환)를 **실제 구현 지침 수준으로 반영**한 v5입니다. fileciteturn0file0  
> 또한 프로젝트 “최종 발표 보고서”의 핵심 설계/기술 근거(스택, RAG, BM25, 스트리밍, LangGraph)를 deck의 Dev 레이어 설명과 다이어그램 요구사항에 일치시키도록 정리했습니다. fileciteturn0file3

---

# 0) 최우선 요구사항(한 줄 체크리스트)
- [ ] **한 화면, 두 레이어**: Investor 기본 / Developer 토글 시 X-ray처럼 딥다이브 전환
- [ ] **Apple-like**: 큰 타이포 + 여백 + 이미지 주도 + 과장 없는 모션
- [ ] **Scroll-Snap**: 13개 섹션이 100vh 단위로 스냅
- [ ] **IntersectionObserver reveal + Count-up**
- [ ] **단일 파일**: `pitch.html` 하나로 실행(내부 `<style>`, `<script>` 포함)
- [ ] **S12 Cost Discipline** 반드시 포함($100 Budget 스토리)
- [ ] **Computational Thinking(S03)** 반드시 포함(CT→구현 매핑)

---

# 1) 발표 구조(30분) → 웹페이지 챕터(13 섹션)
> 발표자는 스크롤/키보드로 “다음 장면”만 넘기면 됩니다.

| Section | Title | Time |
|---|---:|---:|
| S01 | Hero | 0:50 |
| S02 | Problem + Philosophy | 2:10 |
| S03 | Computational Thinking (★핵심) | 2:30 |
| S04 | Solution (Role-based + Modular) | 2:00 |
| S05 | Demo Teaser | 0:40 |
| S06 | Live Demo | 8:00 |
| S07 | Tech Moat (3 pillars) | 2:10 |
| S08 | Architecture | 2:20 |
| S09 | AI System (LangGraph + RAG) | 2:40 |
| S10 | Reliability & Performance | 2:10 |
| S11 | Scale & Proof | 1:40 |
| S12 | Cost Discipline (★핵심) | 2:00 |
| S13 | Business + Roadmap + Ask | 2:10 |

---

# 2) IA & Routing (단일 페이지 + 모드)
- `/pitch.html` : 발표용 단일 페이지 (기본 Investor)
- `/pitch.html?audience=dev` : Developer 레이어 기본 ON
- `/pitch.html?presenter=1` : Presenter Mode(노트/타이머/다음 섹션)
- `/app` : 실제 MVP (데모 버튼으로 새 탭 오픈)

> 단일 파일 구현이므로 실제 라우팅은 “쿼리 파라미터 파싱 + DOM 토글”로 처리.

---

# 3) 글로벌 디자인 시스템 (Dark Mode Base + Glassmorphism)
> 사용자 제공 가이드 반영: **Dark Mode Base** + **Glassmorphism 패널** + radius 24px 이상.

## 3-1) Fonts
- 영문: `Inter`
- 한글: `Pretendard`
- fallback: `system-ui, -apple-system, Apple SD Gothic Neo`

## 3-2) Design Tokens
```yaml
tokens:
  theme_default: "dark"   # 기본은 다크
  colors:
    bg: "#0B0F1A"
    text: "#F5F5F7"
    subtext: "#86868B"
    accent_blue: "#0071E3"
    accent_cyan: "#6AE4FF"
    accent_green: "#30E7A9"  # 효율/비용 절감 강조용
    border: "rgba(245,245,247,0.14)"
    glass:
      surface: "rgba(255,255,255,0.06)"
      surface2: "rgba(255,255,255,0.10)"
      blur: 18
  radius:
    card: 24
    pill: 999
  shadow:
    card: "0 12px 40px rgba(0,0,0,0.35)"
  typography:
    h1: { size: 64, weight: 800, tracking: -0.03em }
    h2: { size: 42, weight: 750, tracking: -0.02em }
    body: { size: 18, weight: 450, lh: 1.65 }
```

## 3-3) UI 컴포넌트 규칙
- 카드/패널: `backdrop-filter: blur(18px)` + `rgba(255,255,255,0.06)` + border 1px
- 라운드: 24px 이상
- 그래프/다이어그램: 라벨 텍스트는 **웹에서 오버레이**(이미지에 텍스트 넣지 않기)

---

# 4) Core UX: Dual-Layer Toggle (Investor ↔ Developer)
> 핵심은 “같은 섹션이 Developer 모드에서 ‘X-ray’로 변한다”는 경험.

## 4-1) UI
- 우측 상단 Sticky Segmented Control: `[ Investor | Developer ]`
- 토글 상태는 `localStorage`에 저장 + `?audience=dev`로 초기값 세팅 가능

## 4-2) 레이어 구조(권장 DOM 규칙)
- 각 섹션 내부는 아래처럼 **동일 레이아웃**에 “두 레이어”를 겹쳐 둔다.

```html
<section class="slide" id="s03">
  <div class="layer layer--investor"> ... </div>
  <div class="layer layer--developer"> ... </div>

  <div class="visual visual--investor"> ... </div>
  <div class="visual visual--developer"> ... </div>
</section>
```

## 4-3) 전환(필수)
- Developer ON:
  - `.layer--investor`, `.visual--investor` → `opacity:0; pointer-events:none; filter: blur(6px);`
  - `.layer--developer`, `.visual--developer` → `opacity:1; transform: translateY(0);`
- 전환 모션: `opacity + transform + blur`를 200~300ms로

---

# 5) Navigation & Motion
## 5-1) Scroll Snap
```css
main { scroll-snap-type: y mandatory; overflow-y: auto; height: 100svh; }
.slide { scroll-snap-align: start; min-height: 100svh; }
```

## 5-2) Scroll-triggered Reveal
- `IntersectionObserver`로 `.reveal` 요소에 `is-visible` class 부여
- 기본 모션: 아래→위 12px + fade-in

## 5-3) Count-up
- 숫자 요소에 `data-count-to="597000"` 같은 속성 부여
- 섹션 진입 시 1회 실행(중복 실행 방지)

---

# 6) 구현 지침 (단일 파일: HTML/CSS/JS)
> 사용자 가이드의 “단일 파일” 요구를 강제 반영.

## 6-1) Deliverable
- `pitch.html` 한 파일만 제출
  - `<style>` 내부에 모든 CSS
  - `<script>` 내부에 모든 JS
  - 폰트는 (선택) `<link rel="preconnect">` + Google Fonts 또는 CDN, 실패 시 fallback

## 6-2) Toggle Logic (필수)
```js
const setAudience = (mode) => {
  document.documentElement.dataset.audience = mode; // 'investor' | 'developer'
  localStorage.setItem('audience', mode);
};

document.querySelector('#audienceToggle').addEventListener('click', (e) => {
  const mode = e.target?.dataset?.mode;
  if (mode) setAudience(mode);
});

// CSS: html[data-audience="developer"] .layer--investor { opacity:0; ... }
```

## 6-3) 코드 스니펫 스타일링(VS Code Dark 느낌)
- 외부 하이라이터 없이 CSS만으로:
  - 배경: `#111827` 계열
  - 토큰: `.kw`, `.fn`, `.str`, `.com` 클래스만 단순 적용
- 라인 수 20줄 제한 + Copy 버튼 제공

---

# 7) 섹션별 상세 스펙(Investor/Dev + Visual Swap)
> 섹션 내용은 v4를 유지하되, “다크/글라스/단일파일/레이어 전환” 지침을 추가로 반영.

---

## S01 Hero — 서막
### Investor Layer
- H1: **법률 정보의 민주화**
- Sub: **검색을 넘어, 사건을 ‘행동’으로 바꾸는 법률 AI 플랫폼**
- Metric chips(글라스 pill):
  - `$100 Budget MVP`
  - `Used $20`
  - `Forecast $55 (1-day demo)`
  - `Buffer $25`
- CTA: `Watch Demo` / `See Architecture`

### Visual(Investor)
- 통합 AI 채팅 UI(출처 카드 강조) + **은은한 파티클 배경**
  - 파티클은 canvas로 30~60개 점(저속)만: “과장 금지”

### Developer Layer
- Stack: `Next.js 14 + FastAPI + LangGraph + PostgreSQL 17 + LanceDB` fileciteturn0file3
- 2 bullets:
  - “단일 챗봇이 아닌 **모듈 오케스트레이션** 플랫폼”
  - “Elasticsearch 없이 **BM25 + 하이브리드**로 복잡도/비용 최소화” fileciteturn0file3

### Visual(Developer)
- X-ray 다이어그램 카드: “Chat Router → Modules → Data(FTS+Vector)” (텍스트는 웹 오버레이)

---

## S02 Problem + Philosophy — 철학
### Investor Layer
- Headline: **법률 문제는 ‘검색’이 아니라 ‘결정’과 ‘절차’다.**
- 3 cards: 이해 / 근거 / 실행(단절)

### Developer Layer
- Evidence-first UX: `sources` 이벤트를 구조적으로 제공
- Workflow-first State Machine: 질문→근거→타임라인/문서화
- Role-based entry: 온보딩 분기

---

## S03 Computational Thinking (★핵심)
### Investor Layer
- 메시지: **컴퓨팅 사고로 법률 문제를 워크플로우로 완벽히 분해하다.**
- 4-step Stepper: Decomposition / Pattern / Abstraction / Algorithm

### Developer Layer(CT→구현 매핑)
- Decomposition → Evidence Upload / Timeline / Wizard
- Pattern → LangGraph 서브그래프(소액소송/스토리보드/모의법정) fileciteturn0file3
- Abstraction → `CaseState` 코드 스니펫
- Algorithm → RAG 5단계 파이프라인(하이브리드→RRF→리랭킹) fileciteturn0file3

---

## S04 Solution — Role-based Modular Platform
### Investor Layer
- User vs Lawyer 분기 + 공통 코어(통합 AI 채팅) fileciteturn0file3

### Developer Layer
- 통합 채팅이 라우터: NAVIGATE 액션으로 모듈 이동 fileciteturn0file3

---

## S05 Demo Teaser / S06 Live Demo
### Investor Layer
- 3장면 티저 + 라이브 데모 버튼
- Live/Replay 토글(네트워크 대비)

### Developer Layer
- 데모 체크포인트:
  - SSE streaming + rAF 렌더링 안정 fileciteturn0file3
  - `sources` 이벤트 확인
  - interrupt 체크포인터(소액소송) fileciteturn0file3

---

## S07 Tech Moat — 3 Pillars
### Investor Layer
- “차별점은 모델이 아니라 시스템”
  1) Evidence-first RAG
  2) Stateful Orchestration
  3) Streaming UX + Reliability

### Developer Layer
- 일반 챗봇 vs 우리 비교(테이블)

---

## S08 Architecture / S09 AI System
### Investor Layer
- Next.js + FastAPI + Postgres/LanceDB + RAG 5단계 fileciteturn0file3

### Developer Layer
- SSE 프록시 분리(Next rewrites 버퍼링 우회) fileciteturn0file3
- HttpOnly 세션 격리 fileciteturn0file3
- `asyncio.gather` 병렬 검색 fileciteturn0file3

---

## S10 Reliability & Performance
### Investor Layer
- 4개 Engineering Guarantees 카드

### Developer Layer(Decision Cards)
- ONNX 품질 게이트 자동 폴백 fileciteturn0file3
- requestAnimationFrame 스트리밍 최적화 fileciteturn0file3

---

## S11 Scale & Proof
### Investor Layer
- Stats count-up:
  - 문서 597k / 벡터 3.16M / FTS 425k / 코드 142k fileciteturn0file3

### Developer Layer
- 21개 타입 인제스트 파이프라인 구조 설명 fileciteturn0file3

---

## S12 Cost Discipline (★핵심)
### Investor Layer
- Key story:
  - **$300+ 예상 → 1일 배포 + 데이터 50GB→15GB + Elasticsearch 제거(BM25 전환) → $55**
- BudgetWaterfall:
  - Start 100 → Spent -20 → AWS -15 → LLM -40 → Buffer 25
- Before/After 인프라 비교 다이어그램(텍스트는 웹 오버레이)

### Developer Layer(재현 가능한 비용 절감 로직)
- Cost Decomposition: Compute / Storage / Search
- 데이터 요약 적재로 50GB→15GB
- BM25(Postgres FTS) 전환으로 별도 검색 클러스터 비용 제거 fileciteturn0file3
- 서버는 ARM Graviton(t4g) 고려 시 런레이트 절감(데모 기간에서 효과 큼) fileciteturn0file2
- “시간 자체가 비용”: 3주 24/7 기준 ~$60 수준 산정 → 1일 운영은 더 낮은 런레이트로 통제 가능 fileciteturn0file1

---

## S13 Business + Roadmap + Ask
### Investor Layer
- B2C/B2B 2트랙 + 로드맵 + Ask(투자/PoC/파트너십) CTA

### Developer Layer
- Q&A 포인트:
  - Gradio 평가 자동화 확장 fileciteturn0file3
  - 컴플라이언스/감사/모니터링
  - 멀티모달 파이프라인 고도화

---

# 8) 에셋 매니페스트(필수)
> 실제 MVP 스크린샷이 1순위. AI 생성 에셋은 “배경/다이어그램/아이콘/컨셉” 위주.

- A01_HERO_KEYVISUAL (16:9, dark base, subtle particle)
- A02_PROBLEM_TRIPTYCH (3 cards)
- A03_CT_STEPPER (4 icons + background)
- A04_ROLE_SPLIT_DIAGRAM (16:9)
- A07_MOAT_ICONSET_3 (3 icons)
- A08_SYSTEM_ARCH_DIAGRAM (16:9)
- A09_LANGGRAPH_ROUTING_MAP (16:9)
- A09_RAG_PIPELINE_5STEP (16:9)
- A10_GUARANTEE_ICONSET_4 (4 icons)
- A11_DATA_PIPELINE_MINI_DIAGRAM (wide)
- A12_BUDGET_WATERFALL (16:9)
- A12_COST_BEFORE_AFTER (16:9)
- A12_DATA_REDUCTION_50_TO_15 (16:9)
- A12_SEARCH_STACK_CHANGE (16:9)
- A13_ROADMAP_TIMELINE_BG (16:9)
- A13_CLOSING_BG (16:9)

---

# 9) QA/리허설 체크(발표 전날/당일)
- Live/Replay 토글 정상
- 데모 프리셋 데이터(복붙) + Reset 버튼
- Presenter Mode 타이머/노트 확인
- Developer 모드에서 Appendix 링크(선택) 정상 작동
- `prefers-reduced-motion`에서도 정보 전달 유지

---
끝.
