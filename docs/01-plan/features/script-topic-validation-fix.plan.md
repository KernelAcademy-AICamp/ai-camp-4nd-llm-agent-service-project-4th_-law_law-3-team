# 대본 생성 버튼 비활성화 버그 수정 계획

## 개요

- **기능**: 콘텐츠 마케팅 - 키워드 기반 뉴스 선택 후 대본 생성
- **증상**: 뉴스 기사를 선택한 후 대본 생성 탭으로 전환 시, "대본 생성하기" 버튼이 비활성화되어 클릭 불가
- **영향도**: 키워드 → 뉴스 → 대본 생성 플로우 전체 차단 (핵심 기능 장애)

## 근본 원인 분석

### 사용자 플로우

```
1. 키워드 탐색 → 키워드 수집
2. 키워드 선택 → 관련 뉴스 검색
3. 뉴스 기사 체크박스 선택
4. "선택한 뉴스로 대본 생성" 클릭
5. → 대본 생성 탭으로 전환 (topic = 키워드 문자열)
6. → "대본 생성하기" 버튼 disabled (topic.length < 5)  ← 버그!
```

### 원인 코드

| 파일 | 라인 | 코드 | 문제점 |
|------|------|------|--------|
| `backend/.../schema/__init__.py` | 298 | `topic: str = Field(min_length=5)` | 한국어 키워드 2~4자 거부 |
| `frontend/.../ScriptGenerator.tsx` | 142 | `disabled={topic.trim().length < 5}` | 버튼 비활성화 |
| `frontend/.../ScriptGenerator.tsx` | 51 | `if (topic.trim().length < 5) return` | 생성 요청 차단 |

### 한국어 키워드 길이 분석

| 키워드 예시 | 글자 수 | 현재 통과 여부 |
|------------|---------|--------------|
| "형법" | 2 | X |
| "민법" | 2 | X |
| "사기죄" | 3 | X |
| "부동산" | 3 | X |
| "부동산법" | 4 | X |
| "형법 개정" | 5 (공백 포함) | O |
| "이재명 판결" | 6 (공백 포함) | O |

## 수정 계획

### 변경 사항 (3개 파일, 3개 지점)

**1. Backend: `backend/app/modules/content_marketing/schema/__init__.py`**

```python
# Before (line 298)
topic: str = Field(min_length=5, max_length=500, ...)

# After
topic: str = Field(min_length=2, max_length=500, ...)
```

**2. Frontend: `frontend/src/features/content-marketing/components/ScriptGenerator.tsx`**

```tsx
// Before (line 51)
if (!topic.trim() || topic.trim().length < 5) return

// After
if (!topic.trim() || topic.trim().length < 2) return

// Before (line 142)
disabled={!topic.trim() || topic.trim().length < 5}

// After
disabled={!topic.trim() || topic.trim().length < 2}
```

**3. Frontend: placeholder 텍스트 업데이트**

```tsx
// Before (line 78)
placeholder="대본 주제를 입력하세요 (최소 5자)"

// After
placeholder="대본 주제를 입력하세요 (최소 2자)"
```

### 최소 길이 2자 근거

- 한국어 법률 키워드 최소 단위: 2자 (형법, 민법, 상법 등)
- 1자는 "법", "죄" 등 의미가 불명확 → 2자 최소가 적절
- 영어도 2자 이상이면 합리적 (예: "AI", "IP")
- Backend-Frontend 동기화 유지

## 리스크 평가

| 리스크 | 영향도 | 대응 |
|--------|--------|------|
| 너무 짧은 주제로 품질 저하 | 낮음 | LLM이 자체적으로 주제 확장 가능 |
| 기존 API 호출과 호환성 | 없음 | min_length 완화는 하위호환 보장 |
| 빈 문자열 입력 | 없음 | `!topic.trim()` 조건으로 빈 입력 방어 |

## 검증 계획

1. Frontend build 통과 확인 (`npm run build`)
2. Backend lint/type 통과 확인 (`ruff check`, `mypy`)
3. 2자 키워드로 대본 생성 버튼 활성화 확인
4. 기존 5자+ 주제 입력이 여전히 동작하는지 확인
