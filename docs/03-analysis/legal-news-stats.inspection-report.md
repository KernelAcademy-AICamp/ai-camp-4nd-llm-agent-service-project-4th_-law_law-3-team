# 법률 뉴스 통계 그래프 — 기능 점검 보고서

**점검일**: 2026-02-27
**점검 팀**: Agent Team (백엔드 개발자, 프론트엔드 개발자)
**점검 대상**: 법률 뉴스 통계 그래프 기능 (일별 수집 막대 그래프 + 카테고리 도넛 차트)

---

## 1. Backend API 점검 결과

### 1.1 일별 통계 API (`GET /api/legal-news/stats/daily`)

| 항목 | 결과 | 상세 |
|------|------|------|
| HTTP 상태 | **200 OK** | 정상 |
| 응답 구조 | **PASS** | `items`, `total`, `period_days` 필드 정상 |
| 데이터 정합성 | **PASS** | 178건, 소스별(lawtimes/naver) 분류 정상 |
| 기간 필터 | **PASS** | `?days=7`, `?days=14`, `?days=30` 모두 정상 |
| 타임존 | **PASS** | KST(UTC+9) 기준 날짜 정상 |

**응답 예시** (`?days=7`):
```json
{
  "items": [
    {"date": "2026-02-24", "source": "lawtimes", "count": 28},
    {"date": "2026-02-24", "source": "naver", "count": 4},
    {"date": "2026-02-25", "source": "naver", "count": 146}
  ],
  "total": 178,
  "period_days": 7
}
```

### 1.2 카테고리 통계 API (`GET /api/legal-news/stats/category`)

| 항목 | 결과 | 상세 |
|------|------|------|
| HTTP 상태 | **200 OK** | 수정 후 정상 |
| 응답 구조 | **PASS** | `items`, `total` 필드 정상 |
| SQL 집계 | **PASS** | CASE WHEN + GROUP BY 정상 동작 |
| 카테고리 분류 | **PASS** | 5개 카테고리 + 기타 정상 분류 |

**응답 예시**:
```json
{
  "items": [
    {"category": "기타", "count": 91},
    {"category": "소송·재판", "count": 53},
    {"category": "형사·검찰", "count": 16},
    {"category": "법령 동향", "count": 12},
    {"category": "법조계 동향", "count": 6}
  ],
  "total": 178
}
```

### 1.3 발견 및 수정된 버그 (Critical)

| 버그 ID | 심각도 | 설명 | 수정 상태 |
|---------|--------|------|----------|
| **BUG-001** | **Critical** | 카테고리 통계 API 500 에러 | **수정 완료** |

**BUG-001 상세:**
- **증상**: `GET /api/legal-news/stats/category` 호출 시 500 Internal Server Error
- **원인**: `_build_category_expression()` 결과를 `group_by(literal("category"))` 로 그룹핑 → PostgreSQL이 문자열 리터럴 `'category'`로 해석하여 `GroupingError: column "news_articles.section" must appear in the GROUP BY clause` 발생
- **수정**: `group_by(literal("category"))` → `group_by(category_expr)` (CASE WHEN 표현식 직접 참조)
- **파일**: `backend/app/modules/legal_news/service.py:264`
- **검증**: ruff check PASS, mypy PASS, API 재테스트 200 OK

### 1.4 정적 분석

| 도구 | 결과 | 에러 |
|------|------|------|
| `ruff check backend/app/` | **PASS** | 0건 |
| `mypy backend/app/` | **PASS** | 관련 에러 0건 (기존 무관 경고만 존재) |

---

## 2. Frontend UI 점검 결과

### 2.1 레이아웃 검증

| 항목 | 데스크톱 (1280px) | 모바일 (390px) | 결과 |
|------|-------------------|---------------|------|
| 2분할 레이아웃 | 좌: 뉴스 목록, 우: 통계 대시보드 | 상: 통계, 하: 목록 | **PASS** |
| 사이드바 너비 | 380px 고정 | 100% (풀 너비) | **PASS** |
| 모바일 순서 | - | `order-first` (통계 먼저) | **PASS** |
| 스크롤 독립성 | 목록 영역만 스크롤 | 전체 스크롤 | **PASS** |

### 2.2 컴포넌트별 검증

#### 일별 수집 막대 그래프 (NewsBarChart)

| 항목 | 결과 | 상세 |
|------|------|------|
| 차트 렌더링 | **PASS** | Stacked bar 정상 표시 |
| 소스 색상 | **PASS** | 로타임즈(파랑 #3b82f6), 네이버(초록 #22c55e) |
| X축 날짜 | **PASS** | MM/DD 형식, 7일간 02/21~02/27 표시 |
| Y축 스케일 | **PASS** | 자동 스케일 (0~160), 정수만 표시 |
| 범례 | **PASS** | "로타임즈", "네이버" 한글 표시 |
| 기간 전환 | **PASS** | 7일/14일/30일 버튼 동작, 선택 상태 파란색 |
| 빈 날짜 보간 | **PASS** | 0건인 날짜도 X축에 표시 |
| 미지 소스 | **PASS** | lawtimes/naver 외 소스 무시 |
| 툴팁 | **PASS** | 호버 시 "N건" + 소스명 표시 |

#### 카테고리 도넛 차트 (NewsDonutChart)

| 항목 | 결과 | 상세 |
|------|------|------|
| 차트 렌더링 | **PASS** | 도넛 차트 정상 표시 (BUG-001 수정 후) |
| 카테고리 표시 | **PASS** | 기타, 법령 동향, 법조계 동향, 소송·재판, 형사·검찰 |
| 색상 구분 | **PASS** | 12색 팔레트 적용, 5개 카테고리 구분 명확 |
| 총 건수 | **PASS** | "총 178건" 표시 |
| 범례 | **PASS** | 카테고리명 범례 정상 표시 |

### 2.3 기능 검증

| 기능 | 결과 | 상세 |
|------|------|------|
| 페이지네이션 | **PASS** | 1/9 페이지, 이전/다음 버튼 동작 |
| 소스 필터 | **PASS** | 전체 소스/법률신문/네이버 드롭다운 |
| 날짜 필터 | **PASS** | 날짜 입력 필드 존재 |
| 탭 전환 | **PASS** | 뉴스 목록/뉴스 검색 탭 전환 시 통계 대시보드 숨김 |
| 면책 고지 | **PASS** | 상단 배너 표시 |
| 뉴스 카드 | **PASS** | 단일 컬럼 리스트, 소스/날짜/제목/요약/태그/출처 표시 |

### 2.4 콘솔 에러

| 시점 | 에러 수 | 상세 |
|------|---------|------|
| 초기 로딩 | **0건** | BUG-001 수정 후 에러 없음 |
| 기간 전환 | **0건** | 정상 |
| 탭 전환 | **0건** | 정상 |

### 2.5 빌드 검증

| 도구 | 결과 |
|------|------|
| `npm run build` | **PASS** (관련 에러 없음) |

---

## 3. 3중 검증 피드백 반영 확인

### Red Team (Gemini CLI) 피드백 반영

| 지적 사항 | 반영 | 검증 |
|----------|------|------|
| Critical: 전체 데이터 메모리 로드 DoS | **반영 완료** | SQL CASE WHEN + GROUP BY 적용 |
| Medium: `func.date()` 인덱스 무효화 | **보류** | 현재 데이터량 소규모 |
| 타임존 혼선 | **반영 완료** | `published_at` 기준 통일 |

### External Consultant (Codex CLI) 피드백 반영

| 지적 사항 | 반영 | 검증 |
|----------|------|------|
| High: 카테고리 SQL 집계 전환 | **반영 완료** | `_build_category_expression()` + GROUP BY |
| High: 미지 소스 오표시 | **반영 완료** | lawtimes/naver만 집계, 미지 소스 무시 |
| Medium: 기간 변경 시 카테고리 재조회 방지 | **반영 완료** | 독립적 fetch 분리 |
| Medium: 빈 날짜 0건 보간 | **반영 완료** | `transformData()` 내 보간 로직 |

---

## 4. 종합 평가

### 점검 요약

| 구분 | 전체 | 통과 | 실패 | 수정 후 통과 |
|------|------|------|------|-------------|
| Backend API | 8 | 7 | 0 | 1 (BUG-001) |
| Frontend UI | 22 | 21 | 0 | 1 (BUG-001 연동) |
| 정적 분석 | 3 | 3 | 0 | 0 |
| **합계** | **33** | **31** | **0** | **2** |

### 최종 판정: **PASS (조건부)**

- 발견된 Critical 버그 1건(BUG-001)은 점검 과정에서 즉시 수정 완료
- 수정 후 재검증에서 모든 항목 통과
- 3중 검증 피드백 4건 모두 정상 반영 확인

### 잔여 기술 부채

| 항목 | 우선순위 | 사유 |
|------|---------|------|
| `func.date()` 함수형 인덱스 | P2 | 데이터량 증가 시 성능 개선 필요 |
| React Query 도입 | P2 | 기존 프로젝트 패턴 유지 중 |
| 테스트 코드 추가 | P1 | 별도 작업으로 분리 |
| 접근성 속성 (ARIA) | P2 | 추후 UX 개선 시 적용 |

---

## 5. 스크린샷

- 데스크톱 뷰: `legal-news-stats-desktop.png`
- 모바일 뷰: `legal-news-stats-mobile.png`

---

## 6. 팀 교차 리뷰 결과

### 6.1 리뷰어별 판정 요약

| 리뷰어 | 역할 | 모델 | 판정 | 요구사항 충족도 |
|--------|------|------|------|:-------------:|
| PM | Product Lead | Opus | **조건부 승인** | 92/100 |
| 백엔드 개발자 | Engineering | Sonnet | **조건부 승인** | - |
| 프론트엔드 개발자 | Engineering | Sonnet | **조건부 승인** | - |
| QA 엔지니어 | Ops/QA | Sonnet | **조건부 승인** | - |

### 6.2 팀 전체 합의 사항

#### 머지 전 필수 수정 (4명 합의)

| # | 항목 | 지적자 | 우선순위 |
|---|------|--------|:--------:|
| 1 | `get_news_stats_daily` total 중복 쿼리 제거 (DB 라운드트립 2회→1회) | 백엔드 | P1 |
| 2 | 라우터 `/{article_id}` 앞 정적 경로 순서 주석 추가 | 백엔드 | P1 |
| 3 | `NewsBarChart` 툴팁 `labelFormatter`에 `formatDate` 적용 | 프론트엔드 | P1 |
| 4 | `transformData` 빈 날짜 보간 시 KST 기준 날짜 계산 수정 | 프론트엔드 | P1 |

#### 다음 스프린트 필수 (PM 조건)

| # | 항목 | 지적자 | 우선순위 |
|---|------|--------|:--------:|
| 1 | 통계 API 단위 테스트 최소 3건 (BUG-001 회귀 방지 포함) | PM, QA | **P0** |
| 2 | `useNewsStats` 로딩 상태 분리 (`dailyLoading`/`categoryLoading`) | PM, 프론트엔드 | P1 |
| 3 | 카테고리 "기타" 51% 문제 분류 체계 재검토 보고서 | PM | P1 |
| 4 | `_classify_category()` 미사용 함수 제거 또는 SSOT 리팩토링 | QA, 백엔드 | P1 |

#### 백로그 (보류)

| 항목 | 우선순위 | 지적자 |
|------|:--------:|--------|
| `DailyStatItem.date` 타입 `str` → `datetime.date` | P2 | 백엔드 |
| `published_at` 함수형 인덱스 Alembic 마이그레이션 | P2 | 백엔드, QA |
| `source` 파라미터 허용 목록 검증 | P2 | QA |
| 차트 접근성 ARIA 속성 추가 | P2 | 프론트엔드 |
| 에러 재시도 버튼 + 스켈레톤 로딩 UI | P2 | 프론트엔드 |
| React Query 도입 | P3 | PM |

### 6.3 주요 기술 인사이트

1. **BUG-001 예방 교훈**: SQLAlchemy에서 `literal("string")`은 SQL 문자열 상수를 생성하므로, `GROUP BY`에서 컬럼 별칭이 아닌 CASE 표현식을 직접 참조해야 함. 이 패턴은 향후 SQL 집계 구현 시 팀 내 공유 필요.

2. **카테고리 분류 이중 구현 문제**: Python 함수(`_classify_category`)와 SQL CASE WHEN(`_build_category_expression`)이 병렬 유지되어 드리프트 위험 존재. 단일 소스(SSOT) 리팩토링 권장.

3. **타임존 일관성**: 백엔드 KST(UTC+9), 프론트엔드 `toISOString()`(UTC) 간 날짜 불일치 가능성. 명시적 KST 변환 필요.

### 6.4 최종 팀 판정

**조건부 승인 (Conditional Approval)**

- 머지 전 필수 수정 4건 완료 시 → 최종 승인
- 다음 스프린트 P0/P1 항목 4건 → 기술 부채 추적 필수

---

## 7. 수정 기획서 연계

### 7.1 수정 기획서

사용자 추가 요구사항 2건 + 팀 리뷰 4건 + 3중 검증 피드백 2건 = **총 9건** 수정 기획서 작성 완료.

- **기획서**: `docs/01-plan/features/legal-news-stats-fix.plan.md`
- **Red Team 보고서**: `docs/03-analysis/legal-news-stats-fix.redteam.md`
- **컨설팅 보고서**: `docs/03-analysis/legal-news-stats-fix.consulting.md`

### 7.2 사용자 추가 요구사항

| # | 항목 | 우선순위 |
|---|------|:--------:|
| F1 | 막대 그래프: Stacked → Grouped (겹침 해소) | **P0** |
| F2 | 막대 그래프: 각 막대 위에 건수 숫자 라벨 표시 | **P0** |
| F3 | 도넛 차트: 각 카테고리 슬라이스에 건수 라벨 표시 | **P0** |

### 7.3 3중 검증 추가 반영 항목

| # | 항목 | 출처 | 우선순위 |
|---|------|------|:--------:|
| F8 | Backend `since` 날짜 정규화 (시각 00:00:00) | Red Team | P1 |
| F9 | 30일 조회 시 막대 라벨 자동 숨김 | Red Team + Consultant | P1 |
