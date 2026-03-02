# 외부 컨설팅 코드 리뷰 보고서 (v2)

> **Date**: 2026-02-26
> **Reviewer**: Codex CLI (External Consultant, gpt-5.3-codex)
> **Target**: `frontend/src/features/legal-news/` + `frontend/src/app/legal-news/page.tsx`
> **Context**: 갭 분석 피드백 반영 후 최종 버전

## 종합 점수: 8/10

## 1. 아키텍처 분석

1. **[High] ID 계약 불일치 가능성**: 목록은 `id`로 상세 조회, 검색은 `doc_id`로 상세 조회 수행. 상세 API가 어느 값을 요구하는지 명확히 확인 필요. 불일치 시 검색 상세 조회가 404 발생 가능.
   - `NewsCard.tsx:17`, `SearchResultCard.tsx:15`, `services/index.ts:36`
2. **[Medium] 오버레이 에러 상태 미포함**: `selectedArticle || detailLoading` 조건에 `detailError` 미포함. 에러 시 패널은 열리지만 배경이 클릭 가능한 비모달 상태.
   - `NewsListPanel.tsx:149`, `SearchPanel.tsx:151`
3. **[Low] 상세 패널 제어 로직 중복**: 두 패널에서 동일한 오버레이+패널 구조 반복. 공통 컴포넌트 추출 권장.

## 2. 테스트 가능성

1. **[Medium] 훅 단위 테스트 부재**: MSW 기반 훅 테스트 필요.
2. **[Low] 유틸 함수 테스트 부재**: 경계값 테스트 용이, 회귀 방지 효과 큼.

## 3. 타입 안전성

1. **[Medium] NewsSourceBadge 타입 느슨**: `source: string` → `NewsSource` 변경 필요.
2. **[Medium] 서비스 계층 타입 불일치**: `source?: string | null` → `NewsSource | null` 변경 필요.
3. **[Low] SOURCE_OPTIONS 빈 문자열 캐스팅**: 타입 가드 권장.

## 4. 확장성/모듈화

1. **[Medium] 상세 패널 오버레이 중복**: 공통 "DetailOverlay" 컴포넌트 추출 권장.
2. **[Low] 필터 상수/옵션 분리는 양호**.

## 5. 모범 사례 대비 분석

1. **[Medium] Focus Trap 부재**: `aria-modal`/포커스 트랩 추가 권장.
2. **[Low] 런타임 API 응답 검증 없음**: Zod 등 런타임 검증 권장.

## 6. 에러 핸들링

1. **[High] 상세 조회 시작 시 이전 기사 미클리어**: `selectedArticle` 초기화 누락으로 실패 시 이전 기사 잔류.
2. **[Medium] 에러 상태에서 오버레이 미표시**: 에러 시에도 모달 컨텍스트 유지 필요.

## 7. 최종 평가 및 권장 사항

### 우선 순위 권장 사항

1. **검색/목록 상세 조회 ID 계약 확정**: `id` vs `doc_id` 확인 후 타입 및 호출 일원화
2. **상세 패널 상태 일관성**: 오버레이 조건에 `detailError` 포함, 새 조회 시 `selectedArticle` 초기화
3. **NewsSource 타입 전파**: 서비스/컴포넌트 전반에 `NewsSource` 적용

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1 | 2026-02-26 | 초기 코드 리뷰 (6.8/10) |
| v2 | 2026-02-26 | 갭 분석 피드백 반영 후 재리뷰 (8/10) |
