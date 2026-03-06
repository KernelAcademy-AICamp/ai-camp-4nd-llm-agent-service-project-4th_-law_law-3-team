# 출처 클릭 시 원본 문서 텍스트 하이라이팅 기능

## 요약

채팅 답변의 참조 출처를 클릭하면, 원본 문서(판례 전문/법령 전문)에서 **RAG가 실제로 참고한 부분**을 노란색으로 하이라이팅하고 자동 스크롤하는 기능을 추가한다.

## 현재 상태

### 법령 (이미 조문 단위 하이라이팅 구현됨)
- `LawDetailUser.tsx`, `LawDetailLawyer.tsx`에서 `article_number` 매칭으로 해당 조문에 파란 배경 + "검색된 조문" 뱃지 표시
- 법령 전문 보기 아코디언에서도 동일 하이라이팅 적용
- **부족한 점**: 조문 내 텍스트 단위 하이라이팅은 없음 (조문 전체가 강조됨)

### 판례 (하이라이팅 전혀 없음)
- `PrecedentDocumentViewer.tsx`의 `SectionContent`가 순수 텍스트만 렌더링
- `PrecedentFullTextViewer.tsx`에 하이라이팅 관련 props 없음
- `PrecedentDetailUser.tsx`에서 판결요지/주문은 별도 카드로 표시하지만, 전문 보기에서 해당 부분이 어디인지 알 수 없음

### 데이터 가용성
- `ChatSource.content`: RAG 청크 텍스트 (판례: ruling+reasoning 합침, 법령: 조문 본문)
- 판례 전문: `full_text`, `full_reason`, `reasoning`, `ruling` 등 섹션별 필드
- 법령 전문: `articles[].article_content` (조문별 본문)
- **LanceDB에 `start_offset` 없음** → 문자열 매칭 방식 필수

## MVP 범위

### 포함
1. **판례 원문 텍스트 하이라이팅**: `PrecedentDocumentViewer`에서 RAG 청크 텍스트(`content`)를 원문 섹션 내에서 찾아 `<mark>` 태그로 강조
2. **판례 자동 스크롤**: 상세 뷰 진입 시 하이라이팅된 첫 번째 위치로 자동 스크롤
3. **법령 텍스트 하이라이팅 강화**: 기존 조문 단위 강조에 추가로, 조문 본문 내에서 RAG 청크 텍스트의 해당 부분을 `<mark>` 태그로 세부 강조
4. **텍스트 매칭 유틸리티**: 공백 정규화 + 부분 문자열 매칭 함수

### 제외
- 챗봇 답변 본문 내 인라인 출처 링크 (현재는 참조 카드 목록 방식 유지)
- 복수 청크 하이라이팅 (동일 문서에서 여러 청크가 검색된 경우 — 현재 중복 제거로 문서당 1개)
- 하이라이팅 색상 커스터마이징

## 기술 설계

### 1. 텍스트 매칭 유틸리티 (`highlightUtils.ts`)

```typescript
// frontend/src/features/case-precedent/utils/highlightUtils.ts

/**
 * 원본 텍스트에서 검색 텍스트가 포함된 구간을 찾아 {start, end}[] 반환
 * - 공백 정규화 (연속 공백 → 단일 공백)
 * - 줄바꿈 차이 무시
 * - 매칭 실패 시 빈 배열 반환
 */
export function findHighlightRanges(
  fullText: string,
  searchText: string,
  options?: { minLength?: number }
): Array<{ start: number; end: number }>

/**
 * 텍스트를 하이라이트 구간에 따라 React 요소 배열로 분할
 * 매칭 구간은 <mark> 태그로 감싸고, 첫 번째 매칭에 ref 부여 (스크롤용)
 */
export function splitByHighlights(
  text: string,
  ranges: Array<{ start: number; end: number }>,
  firstMatchRef?: React.RefObject<HTMLElement>
): React.ReactNode[]
```

**매칭 전략**:
1. 양쪽 텍스트의 공백/줄바꿈을 정규화하여 비교
2. `content` 전체가 매칭되면 그 구간 반환
3. 전체 매칭 실패 시, `content`를 문장 단위(`. ` 기준)로 분리하여 각 문장을 개별 매칭 (긴 청크의 부분 매칭)
4. 최소 30자 이상인 문장만 매칭 시도 (짧은 문장은 오탐 위험)

### 2. PrecedentDocumentViewer 확장

```typescript
// Props 추가
interface PrecedentDocumentViewerProps {
  // ... 기존 props
  /** RAG 검색 청크 텍스트 (하이라이팅 대상) */
  highlightContent?: string
}
```

- `SectionContent` 컴포넌트에 `highlightContent` prop 전달
- `SectionContent`가 `findHighlightRanges()` → `splitByHighlights()`로 매칭 구간을 `<mark>` 렌더링
- 매칭이 없는 섹션은 기존과 동일 (순수 텍스트)
- 첫 번째 `<mark>` 요소에 `ref`를 부여하고, 마운트 후 `scrollIntoView({ behavior: 'smooth', block: 'center' })` 실행

### 3. Props 전달 경로

```
ChatSource.content (RAG 청크)
  ↓
PrecedentFullTextViewer  → highlightContent prop 추가
  ↓
PrecedentDocumentViewer  → highlightContent prop 추가
  ↓
SectionContent           → highlightContent 받아서 텍스트 매칭 + <mark> 렌더링
```

### 4. 법령 텍스트 하이라이팅 강화

- `LawDetailUser.tsx`의 조문 본문 렌더링 부분에서 `source.content`와 `article.article_content`를 매칭
- `ReactMarkdown` 대신 `highlightUtils`로 하이라이팅 적용 (ReactMarkdown과 공존 어려움 → 조문 본문은 마크다운이 단순하므로 plain text 매칭 후 `<mark>` 주입)
- **대안**: ReactMarkdown의 `components` prop을 활용하여 `p`, `li` 등 텍스트 노드에 하이라이팅 적용 — 이 방식이 마크다운 구조를 유지하면서 하이라이팅 가능

### 5. 자동 스크롤

- `useRef`로 첫 번째 `<mark>` 요소 참조
- `useEffect`에서 ref가 설정되면 `scrollIntoView({ behavior: 'smooth', block: 'center' })` 호출
- 스크롤 대상: 상세 뷰의 `overflow-y-auto` 컨테이너 내부

## 목표 파일 구조

### 프론트엔드 (신규 1파일 + 기존 5파일 수정)

```
frontend/src/features/case-precedent/
├── utils/
│   └── highlightUtils.ts                  # [신규] 텍스트 매칭 + React 요소 분할 유틸
├── components/
│   ├── PrecedentDocumentViewer.tsx         # [수정] highlightContent prop 추가, SectionContent 확장
│   ├── PrecedentFullTextViewer.tsx         # [수정] highlightContent prop 추가 + 전달
│   ├── PrecedentDetailUser.tsx            # [수정] source.content를 PrecedentFullTextViewer에 전달
│   ├── PrecedentDetailLawyer.tsx          # [수정] source.content 전달
│   └── LawDetailUser.tsx                  # [수정] 조문 본문 내 텍스트 하이라이팅 추가
```

### 백엔드 (변경 없음)
- `content` 필드는 이미 SSE sources 이벤트로 전달되고 있음
- 추가 API나 데이터 변경 불필요

## 구현 순서

### Phase 1: 텍스트 매칭 유틸리티

**Step 1-1.** `highlightUtils.ts` 신규 생성
- `normalizeWhitespace()`: 공백/줄바꿈 정규화
- `findHighlightRanges()`: 정규화된 텍스트에서 매칭 구간 검색
- `splitByHighlights()`: 매칭 구간을 React 요소로 분할 (`<mark>` 태그)

### Phase 2: 판례 하이라이팅

**Step 2-1.** `PrecedentDocumentViewer.tsx` 수정
- `PrecedentDocumentViewerProps`에 `highlightContent?: string` 추가
- `SectionContent`를 `highlightContent` 인식하도록 확장
- 첫 번째 `<mark>`에 `ref` 부여 + `useEffect`로 자동 스크롤

**Step 2-2.** `PrecedentFullTextViewer.tsx` 수정
- `PrecedentFullTextViewerProps`에 `highlightContent?: string` 추가
- `PrecedentDocumentViewer`에 `highlightContent` 전달

**Step 2-3.** `PrecedentDetailUser.tsx` 수정
- `PrecedentFullTextViewer`에 `highlightContent={source.content}` 전달

**Step 2-4.** `PrecedentDetailLawyer.tsx` 수정
- `PrecedentFullTextViewer`에 `highlightContent={source.content}` 전달

### Phase 3: 법령 텍스트 하이라이팅 강화

**Step 3-1.** `LawDetailUser.tsx` 수정
- 검색된 조문(`isCurrentArticle`)의 본문 렌더링에 `highlightUtils` 적용
- `ReactMarkdown`의 `components` prop으로 `p` 요소 내에서 `source.content` 매칭 하이라이팅
- 또는 조문 본문이 단순 텍스트인 경우 `splitByHighlights()` 직접 사용

### Phase 4: 정적 검증

**Step 4-1.** `cd frontend && npm run build`

## 데이터 흐름

```
[사용자] 챗봇에 질문
    ↓
[백엔드] RAG 파이프라인 → 벡터 검색 → 리랭킹
    ↓
[백엔드] format_precedent_sources() / format_law_sources()
    ↓  content = RAG 청크 텍스트 (검색된 원문 일부)
[SSE] sources 이벤트 → ChatSource[] (content 포함)
    ↓
[프론트엔드] UserView → 출처 카드 목록 표시
    ↓  사용자가 출처 카드 클릭
[프론트엔드] selectedRef = ChatSource (content 포함)
    ↓
[상세 뷰] PrecedentDetailUser / LawDetailUser
    ↓  source.content를 highlightContent로 전달
[PrecedentDocumentViewer / LawDetailUser 전문 보기]
    ↓  findHighlightRanges(sectionText, source.content)
    ↓  splitByHighlights() → <mark> 태그 렌더링
[자동 스크롤] 첫 번째 <mark> 요소로 scrollIntoView
```

## <mark> 스타일

```css
/* Tailwind 기반 인라인 또는 globals.css */
mark {
  background-color: #fef08a;  /* yellow-200 */
  padding: 1px 2px;
  border-radius: 2px;
}
```

## 테스트 케이스

1. **판례 상세 (일반인)**: 출처 클릭 → "판결문 전체 보기" 아코디언 펼침 → 판결요지/이유 섹션에서 검색된 부분이 노란색 하이라이팅
2. **판례 상세 (변호사)**: 출처 클릭 → 전문 뷰에서 해당 부분 하이라이팅 + 자동 스크롤
3. **법령 상세 (일반인)**: 출처 클릭 → 기존 조문 파란 배경 유지 + 조문 본문 내 검색된 텍스트 노란색 하이라이팅
4. **자동 스크롤**: 상세 뷰 진입 시 하이라이팅 위치로 부드럽게 스크롤
5. **매칭 실패**: `content`가 전문에서 찾을 수 없는 경우 → 하이라이팅 없이 기존과 동일 렌더링 (graceful degradation)
6. **빈 content**: `source.content`가 없는 경우 → 하이라이팅 시도 안 함
7. **공백 차이**: DB 원문과 RAG 청크 간 공백/줄바꿈 차이가 있어도 매칭 성공
8. **`npm run build` 통과**

## 가정/기본값

1. `ChatSource.content`에 RAG 검색된 청크 텍스트가 항상 들어온다 (현재 `format_utils.py`에서 보장)
2. 판례 `content`는 `ruling + reasoning`의 합이므로, `reasoning` 또는 `ruling` 섹션에서 매칭 가능
3. 법령 `content`는 해당 조문 본문이므로, `article_content`와 대부분 일치
4. 문자열 매칭은 정규화 후 `indexOf` 기반으로 충분 (정규식 불필요)
5. 동일 문서에 복수 청크가 있는 경우는 현재 중복 제거 로직으로 문서당 1개만 표시

## 리스크

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| 공백/줄바꿈 차이로 매칭 실패 | 중 | 정규화 함수로 연속 공백/줄바꿈 통합 |
| 매우 짧은 content로 오탐 | 낮 | 30자 미만 content는 하이라이팅 건너뛰기 |
| 판례 전문이 없는 경우 (full_text 비어있음) | 낮 | 하이라이팅 없이 기존 동작 유지 |
| ReactMarkdown과 `<mark>` 충돌 | 중 | ReactMarkdown `components` prop 사용 또는 plain text 모드로 fallback |
