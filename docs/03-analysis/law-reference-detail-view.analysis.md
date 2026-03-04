# Gap Analysis: law-reference-detail-view

## 분석 요약

| 항목 | 값 |
|------|------|
| 설계 문서 | `docs/02-design/features/law-reference-detail-view.design.md` |
| 분석일 | 2026-03-03 |
| Match Rate (초기) | 86.1% (31/36) |
| Match Rate (수정 후) | 97.2% (35/36) |
| 총 체크 항목 | 36 |
| 갭 수 (초기) | 5 (Medium 2, Low 3) |
| 갭 수 (수정 후) | 0 |

## 카테고리별 매칭

| 카테고리 | 항목 수 | 매칭 | Match Rate |
|----------|---------|------|-----------|
| Backend (retrieval, pipeline, format_utils) | 8 | 8 | 100% |
| Frontend Types + Utils | 4 | 4 | 100% |
| ReferenceDetailHeader | 4 | 4 | 100% |
| PrecedentDetailUser | 4 | 4 | 100% |
| PrecedentDetailLawyer | 3 | 3 | 100% |
| LawDetailUser | 5 | 5 | 100% |
| LawDetailLawyer | 4 | 4 | 100% |
| UserView 리팩토링 | 4 | 4 | 100% |

## 식별된 갭 (모두 수정 완료)

### Gap-01 [Medium] — docTypeUtils.ts `legislation` 라벨 (수정됨)

- **설계**: `legislation: '입법예고'`
- **구현 (수정 전)**: `legislation: '법령해석'`
- **수정**: `docTypeUtils.ts:6` — `'법령해석'` → `'입법예고'`

### Gap-02 [Low] — ReferenceDetailHeader 미사용 prop (수정됨)

- **설계**: `isLawyer` prop 정의하되 내부 미사용
- **수정**: `ReferenceDetailHeader.tsx` — `isLawyer` prop 인터페이스 및 파라미터에서 제거, `UserView.tsx` 호출부에서도 제거

### Gap-03 [Medium] — LawDetailLawyer 타입 호환성 (수정됨)

- **설계**: CaseDetailPanel에서 `PrecedentDetail` 타입으로 전달
- **구현 (수정 전)**: `ChatSource` 단일 타입 → `as unknown as ChatSource` 캐스팅 필요
- **수정**:
  - `LawDetailLawyer.tsx` — props를 `ChatSource | PrecedentDetail` 유니온으로 변경
  - `PrecedentDetail` 타입에 법령 optional 필드 추가 (`law_name`, `law_type`, `article_number`, `article_title`, `ministry`, `cited_statutes`, `similar_cases`)
  - `CaseDetailPanel.tsx` — `as unknown as ChatSource` 캐스팅 제거

### Gap-04 [Low] — LawDetailUser 경고 아이콘 누락 (수정됨)

- **설계**: "조문 정보 없음" 헤딩에 ⚠️ 아이콘 포함
- **구현 (수정 전)**: 아이콘 없이 텍스트만 표시
- **수정**: `LawDetailUser.tsx:62` — SVG 경고 아이콘 추가

### Gap-05 [Low] — SearchPanel 빈 메시지 중복 (수정됨)

- **설계**: 헤더와 콘텐츠 영역 메시지 역할 분리
- **구현 (수정 전)**: 헤더 "챗봇에게 질문하면 관련 문서가 표시됩니다" + 콘텐츠 "챗봇에게 법률 질문을 하면..." 중복
- **수정**: 헤더 — 결과 없을 때 "관련 문서" (중립), 결과 있을 때 "관련 문서 (N건)"; 콘텐츠 영역 — 기존 안내 메시지 유지

## 검증 결과

- `ruff check app/services/rag/retrieval.py pipeline.py format_utils.py` — All checks passed
- `npx tsc --noEmit` — 0 case-precedent errors (vis-timeline 기존 이슈만 존재)
