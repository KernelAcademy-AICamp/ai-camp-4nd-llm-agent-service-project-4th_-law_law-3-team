# API 스키마 수정 - ChatSource content 필드 누락 해결

## 날짜: 2026-01-29

## 문제

프론트엔드 챗봇에서 참조 자료(판례/법령) 목록이 표시될 때, 내용 미리보기가 나타나지 않고 "클릭하여 상세 내용을 확인하세요."라는 fallback 메시지만 표시됨.

### 증상

- `UserView.tsx`에서 `ref.content`가 `undefined`로 나타남
- 백엔드 API가 `content` 필드를 반환하지 않는 경우 발생

### 원인 분석

1. **에이전트별 응답 형식 불일치**: `case_search` 에이전트는 `content`를 포함하지만, `small_claims` 에이전트는 누락
2. **세션 라우팅**: 이전 대화 컨텍스트에 따라 다른 에이전트로 라우팅되어 응답 형식이 달라짐

### 디버깅 과정

1. curl로 백엔드 API 직접 테스트 → content 정상 반환
2. 브라우저 콘솔에서 API 응답 확인 → content 없음
3. `agent_used` 필드 확인 → `small_claims` 에이전트 사용 확인
4. `small_claims_agent.py` 코드 분석 → sources 구성 시 content 필드 누락

## 해결

### 1. `small_claims_agent.py` 수정

```python
# 수정 전
sources = [
    {
        "case_name": doc.get("metadata", {}).get("case_name", ""),
        "case_number": doc.get("metadata", {}).get("case_number", ""),
        "doc_type": doc.get("metadata", {}).get("doc_type", ""),
        "similarity": round(doc.get("similarity", 0), 3),
    }
    for doc in related_docs
]

# 수정 후
sources = [
    {
        "case_name": doc.get("metadata", {}).get("case_name", ""),
        "case_number": doc.get("metadata", {}).get("case_number", ""),
        "doc_type": doc.get("metadata", {}).get("doc_type", ""),
        "similarity": round(doc.get("similarity", 0), 3),
        "content": doc.get("content", ""),  # 추가
    }
    for doc in related_docs
]
```

### 2. 프론트엔드 ChatSource 인터페이스 확장

`frontend/src/components/ChatWidget.tsx`와 `frontend/src/features/case-precedent/types/index.ts`:

```typescript
interface ChatSource {
  case_name?: string       // optional (법령은 없음)
  case_number?: string     // optional (법령은 없음)
  doc_type: string         // required
  similarity: number       // required
  summary?: string         // optional
  content?: string         // optional - 추가됨
  law_name?: string        // optional (판례는 없음)
  law_type?: string        // optional (판례는 없음)
  cited_statutes?: string[]  // optional (그래프 보강)
  similar_cases?: string[]   // optional (그래프 보강)
}
```

## 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/modules/multi_agent/agents/small_claims_agent.py` | sources에 content 필드 추가 |
| `frontend/src/components/ChatWidget.tsx` | ChatSource 인터페이스 확장 |
| `frontend/src/features/case-precedent/types/index.ts` | ChatSource 타입 정의 확장 |
| `frontend/src/features/case-precedent/components/UserView.tsx` | 디버그 코드 제거 |

## 검증

```bash
# 백엔드 API 테스트
curl -s -X POST http://localhost:8000/api/multi-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "손해배상 판례", "user_role": "user"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print('content' in d['sources'][0])"
# 출력: True
```

## 교훈

1. **일관된 API 스키마**: 모든 에이전트가 동일한 응답 형식을 반환하도록 통일 필요
2. **타입 정의 공유**: 프론트엔드/백엔드 간 타입 정의를 공유하는 방안 검토 (예: OpenAPI 스키마 생성)
3. **디버깅 방법**: 브라우저 콘솔에서 전체 API 응답을 JSON으로 출력하여 필드 누락 확인

## 관련 이슈

- 프론트엔드-백엔드 스키마 불일치
- 멀티 에이전트 응답 형식 통일
