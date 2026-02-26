# 트렌드 분석 시스템 문제점 및 개선방향 보고서

> **분석 일자**: 2026-02-25
> **분석 팀**: 에이전트 4명 + Gemini CLI 레드팀 + Claude Lead Manager
> **대상 시스템**: 콘텐츠 마케팅 트렌드 수집 파이프라인 v3.0

---

## 1. 현황 요약

| 항목 | 관측값 | 기대값 |
|------|--------|--------|
| 1위 종합점수 | 31.3 / 100 | 70+ / 100 |
| 10위 종합점수 | 24.4 / 100 | 30+ / 100 |
| 점수 분포 범위 | 20~35점 집중 | 30~85점 분포 |
| 검색 결과 구성 | 루리웹 게임 콘텐츠 지배 | 카테고리 관련 법률/사회 이슈 |
| 카테고리 필터 효과 | 형사 선택해도 게임 결과 | 형사 관련 뉴스만 수집 |
| 설계-구현 일치도 | 73% (경고) | 90%+ (양호) |

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### 2.1 원인 1: Tavily 도메인 하드코딩 (심각도: Critical)

**위치**: `backend/app/tools/trend/sources/tavily_source.py` L66-71

```python
include_domains=[
    "https://bbs.ruliweb.com/community/board/300561",  # 루리웹 게임 게시판
    "https://bada.us",
    "http://www.todayhumor.co.kr/board/list.php?table=bestofbest",
    "https://www.dcinside.com"
],
```

**영향**: Stage 1에서 게임/유머 커뮤니티만 수집 → Stage 2에서 게임 키워드 추출 → Stage 3에서 게임 뉴스 검색 (연쇄 오염)

**Gemini 레드팀 의견**: "수도꼭지의 수원(Source)을 바꿔야 한다. 쿼리를 바꿔도 소스가 루리웹이면 무용지물."

### 2.2 원인 2: 카테고리 필터 미연결 (심각도: Critical)

**위치**: `backend/app/tools/trend/collector.py` L75-100

`request.category`가 캐시 키에만 사용되고 검색 쿼리 구성에 전혀 반영되지 않음. `_CATEGORY_SEARCH_TERMS` 딕셔너리(L34-42)가 존재하지만 `persona.specialty_areas`를 통해서만 접근 가능.

| 위치 | category 사용 | 내용 |
|------|:---:|------|
| `_make_cache_key()` | O | 캐시 키 구성만 |
| `_build_search_query()` | X | persona만 참조, category 무시 |
| `_collect_community()` | X | category 미전달 |
| `_collect_news()` | X | Stage 3 쿼리에 category 미반영 |
| `scorer._calculate_fitness_score()` | 간접 | 수집 후 스코어링에서만 사용 |

### 2.3 원인 3: 스코어링 공식 구조 결함 (심각도: High)

**위치**: `backend/app/tools/trend/scorer.py` L238

```python
# Gate 통과 공식
combined = analysis.legal_score * base * 100
# base = 0.30*M + 0.25*C + 0.15*S + 0.30*F

# legal_score=0.35, base=0.895 → combined = 31.3 (관측된 1위 점수)
```

`legal_score`가 **곱셈 승수(Multiplier)**로 작동하여 점수 상한선을 제한:
- legal_score=0.3 → 최대 30점
- legal_score=0.5 → 최대 50점
- legal_score=0.9 → 최대 90점

### 2.4 원인 4: mention_score 분모 과잉 (심각도: High)

**위치**: `backend/app/tools/trend/scorer.py` L137

```python
return min(len(group_items) / max(total_items * 0.3, 1), 1.0)
```

50건 수집 시 특정 이슈 5건이면 `5 / 15 = 0.33`. 현실적으로 0.15~0.35 범위에 집중.

### 2.5 원인 5: Legal Gate 임계값 설계 불일치 (심각도: Medium)

**위치**: `backend/app/core/config.py` L121

| 항목 | 설계 문서 | 실제 구현 |
|------|----------|----------|
| TREND_LEGAL_THRESHOLD | 0.3 | 0.1 |

0.1로 설정하여 대부분의 이슈가 Gate를 통과하지만, Gate-passed 공식에서 낮은 legal_score가 곱셈자로 작동하여 오히려 점수를 깎아냄.

---

## 3. 에이전트별 분석 요약

### Agent 1: 스코어링 공식 분석가

| 발견 | 심각도 | 위치 |
|------|:------:|------|
| legal_score 곱셈 승수 효과 → 점수 상한 30점대 | 치명 | scorer.py L238 |
| mention_score 분모(total*0.3) → 현실적 max 0.35 | 치명 | scorer.py L137 |
| spread_score 분모 3.0 하드코딩 (실제 소스 7개) | 경고 | scorer.py L387 |
| LLM 프롬프트 legal_score 가이드 → 0.3 미만 유도 | 치명 | scorer.py L293 |

**핵심 제안**: legal_score를 독립 가중합 차원으로 전환, mention_score 로그 스케일 적용

### Agent 2: 파이프라인 매핑 분석가

| 발견 | 심각도 | 위치 |
|------|:------:|------|
| Tavily include_domains 게임 커뮤니티 고정 | 치명 | tavily_source.py L66-71 |
| request.category 검색 쿼리 미반영 | 치명 | collector.py L75-100 |
| 게임 키워드 Stage 1→2→3 연쇄 오염 | 치명 | collector.py L173-177 |
| _CATEGORY_SEARCH_TERMS persona 경유만 가능 | 높음 | collector.py L88-89 |

**핵심 제안**: tavily_source.py 도메인 교체, _build_search_query에 category 파라미터 추가

### Agent 3: 검색 품질 개선안 분석가

| 개선안 | 변경 파일 | 핵심 내용 |
|--------|----------|----------|
| A. 도메인 전략 | tavily_source.py | 카테고리별 동적 도메인 매핑 |
| B. 카테고리 필터 | collector.py, sources/__init__.py | SourceConfig에 category 추가 |
| C. 키워드 추출 | keyword_extractor.py | 카테고리 컨텍스트 주입 + 게임 제외 강화 |
| D. 스코어링 재설계 | scorer.py | 가산 방식 가중합 (legal 0.15 등) |

### Agent 4: 설계-구현 갭 분석가

| 갭 | 심각도 | 설계 의도 | 실제 동작 |
|----|:------:|----------|----------|
| 카테고리 필터링 | High | 카테고리별 맞춤 수집 | 모든 카테고리 동일 결과 |
| 페르소나 개인화 | High | 3차원 적합도 (분야+토픽+시청자) | audience_relevance 누락 |
| Legal Gate 임계값 | High | 0.3 (유의미한 필터링) | 0.1 (Gate 무력화) |
| Tavily 소스 | High | 사회적 이슈 수집 | 게임 커뮤니티 고정 |
| 점수 분포 | Medium | 0-100 의미있는 분포 | 20-35 집중 |

---

## 4. Gemini CLI 레드팀 검증 의견

### 4.1 에이전트 분석에 대한 반론

| 에이전트 제안 | 레드팀 반론 |
|-------------|-----------|
| 점수 공식을 가산 방식으로 변경 (Agent 1) | "게임 데이터에 높은 점수를 주면 **오탐(False Positive) 양산**. 점수가 낮은 게 아니라 낮아야 할 데이터만 들어오는 것이 문제" |
| 커뮤니티 도메인을 클리앙/펨코로 변경 (Agent 3) | "일반 커뮤니티는 잡담 비중 90%+. **전문 도메인 소스(법률신문, 로톡뉴스)**가 완전히 누락됨" |
| 키워드 추출기에 카테고리 컨텍스트 주입 (Agent 3) | "게임 글에 '형사법 관점'으로 키워드 추출 강제하면 **LLM 환각(Hallucination)** 유발 위험" |

### 4.2 에이전트들이 놓친 추가 문제점

| 문제 | 설명 |
|------|------|
| 시의성(Recency) 평가 부재 | 데이터 생성 시점, 증가 속도(Velocity)에 대한 고려 없음 |
| 토큰 비용 효율성 | 모든 Raw Data를 LLM 파이프라인에 태움. Rule-based 1차 필터링 없음 |

### 4.3 레드팀 대안 제안: 이원화 파이프라인

```
Track A: 전문 정보 트랙 (High Quality, Low Noise)
├── 소스: 법률신문, 로톡뉴스, 언론사 법조/사회 섹션
├── 처리: 요약 + 법적 쟁점 추출 중심
└── 목적: 신뢰할 수 있는 법률 트렌드 (Base Data)

Track B: 소셜 리스닝 트랙 (High Noise, High Speed)
├── 소스: 커뮤니티(클리앙, 펨코, 블라인드), 트위터
├── 처리: Strict Filtering (법률 핵심 키워드 50개 필터)
└── 목적: 대중의 법 감정 및 화제성 파악
```

---

## 5. Lead Manager 종합 의견 및 수정 우선순위

에이전트 분석과 레드팀 반론을 종합하여, 다음과 같이 **실행 가능한 수정 우선순위**를 결정합니다.

### 5.1 수정 우선순위

| 순위 | 영역 | 변경 파일 | 변경 내용 | 근거 |
|:----:|------|----------|----------|------|
| **1** | 소스 도메인 | `tavily_source.py` L66-71 | include_domains 제거 또는 동적 매핑 | 모든 에이전트 + 레드팀 동의. 근본 원인 |
| **2** | 카테고리 필터 | `collector.py` L75-100, L133-148, L161-177 | `_build_search_query`에 category 파라미터 추가 | Agent 2,3,4 동의. 카테고리 무용지물 해소 |
| **3** | 키워드 추출 | `keyword_extractor.py` L16-31 | 카테고리 컨텍스트 추가 + 비법률 강력 제외 | Agent 3 제안. 레드팀 환각 주의 반영하여 제한적 적용 |
| **4** | 스코어링 공식 | `scorer.py` L238, L137, L387 | legal_score 독립화 + mention_score 로그 스케일 | Agent 1 제안. 레드팀 반론 반영하여 Legal Gate 유지 |
| **5** | Legal Gate | `config.py` L121 | TREND_LEGAL_THRESHOLD 0.1→0.3 복원 | Agent 4 제안 (설계 원래값) |
| **6** | SourceConfig | `sources/__init__.py` | category 필드 추가 | Agent 3 제안. 도메인 동적 매핑 지원 |

### 5.2 스코어링 공식 권장안

레드팀의 "Legal Gate는 관문으로 유지" 의견과 에이전트 1의 "가산 방식" 제안을 절충합니다.

```python
# 권장 공식: Legal Gate 유지 + 가중합 방식 (Hybrid)

if legal_gate_passed:
    # Step 1: 가중합 (legal_score 포함, 곱셈 아닌 독립 차원)
    raw = (
        0.20 * analysis.legal_score      # 법률 관련성 (독립 차원)
        + 0.25 * mention                  # 언급 빈도
        + 0.20 * analysis.controversy_score  # 논란도
        + 0.10 * spread                   # 확산도
        + 0.25 * fitness                  # 적합도
    ) * 100  # max 100

    # Step 2: Legal Gate 보너스 (통과 시 기본 보정)
    combined = max(raw, 30.0)  # Gate 통과 최소 보장 30점
else:
    # Gate 미통과: 화제성만 반영, 페널티 적용
    combined = (
        0.4 * mention
        + 0.3 * analysis.controversy_score
        + 0.3 * spread
    ) * 100 * 0.6  # 40% 페널티
```

**예상 점수 분포 (개선 후)**:

| 시나리오 | legal | mention | controversy | spread | fitness | 예상 점수 |
|---------|:-----:|:-------:|:-----------:|:------:|:-------:|:---------:|
| 법률 직접 이슈 | 0.9 | 0.6 | 0.7 | 0.5 | 0.8 | **72** |
| 사회 법률 관련 | 0.5 | 0.4 | 0.5 | 0.4 | 0.6 | **48** |
| 법률 간접 관련 | 0.3 | 0.3 | 0.4 | 0.3 | 0.5 | **35** |
| Gate 미통과 (화제) | 0.1 | 0.5 | 0.6 | 0.4 | - | **29** |
| Gate 미통과 (비관련) | 0.05 | 0.2 | 0.2 | 0.2 | - | **12** |

### 5.3 Tavily 도메인 권장안

레드팀의 "전문 도메인 소스" 제안을 수용하되, 현재 Tavily API의 한계를 고려합니다.

```python
# 즉시 적용 가능한 방안: include_domains 제거 → 전체 웹 검색 허용
# 사용자가 직접 지정한 커뮤니티는 별도 SourceConfig로 관리

# 중기 방안: 카테고리별 도메인 + 법률 전문 소스 추가
_CATEGORY_DOMAINS = {
    "all": [],  # 제한 없음 (전체 웹 검색)
    "criminal": ["news.naver.com", "www.lawtimes.co.kr"],
    "civil": ["news.naver.com", "www.lawtimes.co.kr"],
    # ...
}
```

---

## 6. 결론

현재 트렌드 분석 시스템은 **두 가지 독립적 버그의 복합 작용**으로 기능이 마비된 상태입니다:

1. **입력 오염**: Tavily가 게임 커뮤니티만 수집 → 전체 파이프라인 오염
2. **출력 왜곡**: 스코어링 공식이 점수를 구조적으로 억압

수정 순서는 **"입력 정화(소스 교체) → 필터 연결(카테고리) → 공식 보정(스코어링)"** 순으로 진행해야 하며, 공식만 수정하면 오탐이 양산된다는 레드팀의 경고를 반영해야 합니다.

---

## 7. 추가 발견 사항 (Agent 4 갭 분석 보완)

Agent 4의 상세 갭 분석에서 추가로 발견된 항목입니다.

### 7.1 Auth IDOR 취약점 (심각도: High - 보안)

**위치**: `backend/app/modules/content_marketing/router/__init__.py` L46-48

```python
TEMP_USER_ID = "temp_user_001"  # 모든 사용자가 동일한 user_id 사용
```

5개 Persona API 엔드포인트가 `TEMP_USER_ID`를 하드코딩하여 모든 사용자가 같은 페르소나를 공유. 설계(§2.1.1)에서는 Auth Dependency(Bearer Token)에서 user_id를 추출하도록 명시.

### 7.2 BackgroundTrendWorker 미구현 (심각도: Medium)

설계(§2.3.1)에서는 매 1시간 크론으로 글로벌 캐시를 갱신하고, API 호출 시 캐시에서 읽도록 의도. 현재는 API 호출마다 전체 LLM 파이프라인(수집+스코어링+요약)을 실행하여 비용과 응답 시간 모두 비효율적.

### 7.3 설계-구현 종합 매칭 비율

| 분류 | 설계 항목 | 구현 완료 | 불일치 | 미구현 |
|------|:--------:|:--------:|:------:|:------:|
| 트렌드 수집 파이프라인 | 8 | 4 | 3 | 1 |
| 스코어링 공식 | 6 | 3 | 2 | 1 |
| API 및 보안 | 5 | 3 | 1 | 1 |
| **합계** | **19** | **10 (53%)** | **6** | **3** |

---

## 부록: 변경 대상 파일 목록

| # | 파일 | 변경 내용 | 우선순위 |
|---|------|----------|:--------:|
| 1 | `backend/app/tools/trend/sources/tavily_source.py` | include_domains 제거/교체 | P0 |
| 2 | `backend/app/tools/trend/collector.py` | _build_search_query에 category 추가 | P0 |
| 3 | `backend/app/tools/trend/sources/__init__.py` | SourceConfig에 category 필드 추가 | P1 |
| 4 | `backend/app/tools/trend/keyword_extractor.py` | 카테고리 컨텍스트 + 비법률 제외 강화 | P1 |
| 5 | `backend/app/tools/trend/scorer.py` | 점수 공식 재설계 (L238, L137, L387) | P1 |
| 6 | `backend/app/core/config.py` | TREND_LEGAL_THRESHOLD 0.1→0.3 | P2 |
| 7 | `backend/app/modules/content_marketing/router/__init__.py` | Auth Dependency 구현 (TEMP_USER_ID 제거) | P2 (보안) |
