# Backend Python 코딩 표준 수정 보고서

> 작성일: 2026-01-23
> 기준: Python Coding Standards (`.claude/skills/python-coding-standards`)

## 개요

Python 코딩 표준 점검을 통해 발견된 즉시 수정이 필요한 이슈들을 수정했습니다.

### 수정된 파일 목록

| 파일 | 수정 항목 |
|------|----------|
| `app/modules/small_claims/router/__init__.py` | Import 정렬, 로깅, 하드코딩 제거, 에러 메시지 보안 |
| `app/common/chat_service.py` | Import 정렬, 로깅, 전역 변수 → lru_cache, 타입 힌트 |
| `app/modules/storyboard/router/__init__.py` | 로깅 추가, 에러 메시지 보안 |
| `app/modules/storyboard/service/image_generation.py` | 로깅 추가 |
| `app/modules/lawyer_finder/router/__init__.py` | Import 정렬, 로깅 추가 |
| `app/modules/case_precedent/router/__init__.py` | Import 정렬, 로깅 추가, 에러 메시지 보안 |

---

## 1. 하드코딩된 설정값 제거

### 문제점
`small_claims/router/__init__.py`에서 OpenAI 모델명이 하드코딩되어 있었습니다.

### 수정 내용

**Before:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",  # 하드코딩
    messages=[...],
)
```

**After:**
```python
from app.core.config import settings

response = client.chat.completions.create(
    model=settings.OPENAI_MODEL,  # 환경 설정 사용
    messages=[...],
)
```

### 적용 파일
- `app/modules/small_claims/router/__init__.py:387`

---

## 2. 로깅 추가

### 문제점
예외 발생 시 로깅 없이 에러만 반환하여 디버깅이 어려웠습니다.

### 수정 내용

모든 라우터 파일에 로깅을 추가했습니다:

```python
import logging

logger = logging.getLogger(__name__)
```

예외 핸들러에 로깅 추가:

**Before:**
```python
except Exception as e:
    raise HTTPException(status_code=500, detail=f"서류 생성 실패: {str(e)}")
```

**After:**
```python
except Exception as e:
    logger.error(f"서류 생성 실패: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="서류 생성 중 오류가 발생했습니다")
```

### 적용 파일
- `app/modules/small_claims/router/__init__.py` (2개 핸들러)
- `app/modules/storyboard/router/__init__.py` (7개 핸들러)
- `app/modules/storyboard/service/image_generation.py` (1개 핸들러)
- `app/modules/case_precedent/router/__init__.py` (6개 핸들러)
- `app/modules/lawyer_finder/router/__init__.py` (로거 설정)

---

## 3. 에러 메시지 보안 강화

### 문제점
예외 메시지에 `str(e)`를 포함하여 내부 시스템 정보가 클라이언트에 노출될 위험이 있었습니다.

### 수정 내용

**Before:**
```python
raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")
# 잠재적 노출: "검색 실패: connection refused to database at 192.168.1.100:5432"
```

**After:**
```python
logger.error(f"검색 실패: {e}", exc_info=True)  # 서버 로그에만 기록
raise HTTPException(status_code=500, detail="검색 중 오류가 발생했습니다")
# 클라이언트에는 일반적인 메시지만 표시
```

### 적용된 엔드포인트

| 모듈 | 엔드포인트 | 수정된 에러 메시지 |
|------|-----------|-------------------|
| small_claims | POST /generate-document | "서류 생성 중 오류가 발생했습니다" |
| small_claims | GET /related-cases/{type} | "관련 판례 조회 중 오류가 발생했습니다" |
| storyboard | POST /extract | "타임라인 추출 중 오류가 발생했습니다" |
| storyboard | POST /validate | "유효성 검사에 실패했습니다" |
| storyboard | POST /transcribe | "음성 변환 중 오류가 발생했습니다" |
| storyboard | POST /analyze-image | "이미지 분석 중 오류가 발생했습니다" |
| storyboard | POST /generate-image | "이미지 생성 중 오류가 발생했습니다" |
| storyboard | POST /generate-images-batch | "일괄 이미지 생성을 시작할 수 없습니다" |
| storyboard | POST /generate-video | "영상 생성 중 오류가 발생했습니다" |
| case_precedent | POST /chat | "챗봇 응답 생성 중 오류가 발생했습니다" |
| case_precedent | POST /search | "검색 중 오류가 발생했습니다" |
| case_precedent | GET /precedents | "판례 검색 중 오류가 발생했습니다" |
| case_precedent | GET /precedents/{id} | "판례 조회 중 오류가 발생했습니다" |
| case_precedent | POST /precedents/{id}/ask | "질문 처리 중 오류가 발생했습니다" |

---

## 4. 전역 변수 → @lru_cache 전환

### 문제점
`chat_service.py`에서 전역 mutable 변수를 사용하여 모델을 캐싱하고 있었습니다.

### 수정 내용

**Before:**
```python
_local_model = None

def get_local_model():
    """sentence-transformers 모델 로드"""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _local_model = SentenceTransformer(
            settings.LOCAL_EMBEDDING_MODEL,
            cache_folder=str(cache_dir)
        )
    return _local_model
```

**After:**
```python
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_local_model() -> "SentenceTransformer":
    """sentence-transformers 모델 로드 (캐싱)"""
    from sentence_transformers import SentenceTransformer
    cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        settings.LOCAL_EMBEDDING_MODEL,
        cache_folder=str(cache_dir)
    )
```

### 장점
- 전역 변수 제거로 사이드 이펙트 감소
- `@lru_cache`는 스레드 안전
- 타입 힌트 추가로 IDE 지원 개선
- 코드 간결화

### 적용 파일
- `app/common/chat_service.py:28-40`

---

## 5. Import 정렬 (PEP 8)

### 규칙
```
1. 표준 라이브러리 (import os, from datetime import ...)
2. 서드파티 패키지 (from fastapi import ...)
3. 로컬 모듈 (from app.core import ...)
```

### 수정 예시

**Before (`small_claims/router/__init__.py`):**
```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.common.chat_service import search_relevant_documents
```

**After:**
```python
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.common.chat_service import search_relevant_documents
from app.core.config import settings

logger = logging.getLogger(__name__)
```

### 적용 파일
- `app/modules/small_claims/router/__init__.py`
- `app/common/chat_service.py`
- `app/modules/lawyer_finder/router/__init__.py`
- `app/modules/case_precedent/router/__init__.py`

---

## 6. 타입 힌트 추가

### 수정 내용

**Before:**
```python
def get_local_model():
    ...
```

**After:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

def get_local_model() -> "SentenceTransformer":
    ...
```

### 적용 파일
- `app/common/chat_service.py:31`

---

## 향후 개선 권장 사항 (P2/P3)

다음 항목들은 시간이 있을 때 추가 리팩토링을 권장합니다:

### P2 - 파일 분할 필요
| 파일 | 현재 줄 수 | 권장 조치 |
|------|-----------|----------|
| `small_claims/router/__init__.py` | 508줄 | 템플릿/스키마 분리 |
| `lawyer_finder_agent.py` | 503줄 | 상수(좌표/키워드) 분리 |
| `chat_service.py` | 425줄 | 기능별 모듈 분리 |
| `lawyer_finder/service/__init__.py` | 413줄 | 지오코딩 함수 분리 |

### P3 - 추가 개선
- 모든 라우터 함수에 반환 타입 힌트 추가
- 테스트 코드 작성
- mypy strict 모드 적용

---

## 버그 수정 (2차)

> 작성일: 2026-01-23

### 7. Path Traversal 보안 취약점 수정 (High)

**파일:** `app/modules/storyboard/service/video_generation.py`

**문제:** 문자열 prefix 비교로 경로 검증 시 우회 가능

**Before:**
```python
if not str(source_path).startswith(str(media_dir_resolved)):
    return False
```

**After:**
```python
try:
    source_path.relative_to(media_dir_resolved)
except ValueError:
    logger.warning(f"경로 순회 시도 감지: {url}")
    return False
```

---

### 8. IndexError 방지 - API 응답 검증

**문제:** `response.choices[0]` 접근 시 빈 리스트면 IndexError 발생

**수정된 파일:**
- `app/modules/small_claims/router/__init__.py`
- `app/modules/case_precedent/router/__init__.py`
- `app/modules/storyboard/service/__init__.py`

**Before:**
```python
generated_body = response.choices[0].message.content
```

**After:**
```python
if not response.choices:
    raise HTTPException(status_code=503, detail="AI 응답이 없습니다")
generated_body = response.choices[0].message.content
```

---

### 9. IndexError 방지 - JSON 파싱

**파일:** `app/modules/storyboard/service/vision.py`

**문제:** split 결과가 예상보다 적을 때 IndexError

**Before:**
```python
if "```json" in content:
    content = content.split("```json")[1].split("```")[0]
```

**After:**
```python
if "```json" in content:
    parts = content.split("```json")
    if len(parts) > 1:
        inner_parts = parts[1].split("```")
        content = inner_parts[0] if inner_parts else parts[1]
```

---

### 10. 예외 무시 → 로깅 추가

**수정된 파일:**
- `app/modules/storyboard/service/job_manager.py`
- `app/modules/storyboard/service/video_generation.py`
- `app/modules/multi_agent/agents/small_claims_agent.py`

**Before:**
```python
except Exception:
    failed.append(item["id"])
```

**After:**
```python
except Exception as e:
    logger.error(f"이미지 생성 실패 (item_id={item['id']}): {e}", exc_info=True)
    failed.append(item["id"])
```

---

### 11. Race Condition 수정

**파일:** `app/modules/storyboard/service/job_manager.py`

**문제:** dict 조회와 제거 사이에 다른 코루틴이 수정할 수 있음

**Before:**
```python
finally:
    if job_id in self._subscribers:
        self._subscribers[job_id].remove(queue)
```

**After:**
```python
finally:
    try:
        subscribers = self._subscribers.get(job_id)
        if subscribers and queue in subscribers:
            subscribers.remove(queue)
    except (KeyError, ValueError):
        pass  # 이미 제거됨
```

---

### 12. JSON 파일 로드 에러 처리

**파일:** `app/modules/lawyer_finder/service/__init__.py`

**문제:** 파일이 손상되었을 때 JSONDecodeError 미처리

**Before:**
```python
if LAWYERS_FILE.exists():
    with open(LAWYERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
```

**After:**
```python
for file_path in files_to_try:
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류 ({file_path}): {e}")
            continue
        except UnicodeDecodeError as e:
            logger.error(f"인코딩 오류 ({file_path}): {e}")
            continue
```

---

### 13. VectorStore IndexError 방지

**파일:** `app/common/vectorstore/base.py`

**문제:** documents/metadatas가 존재하지만 빈 리스트일 때 IndexError

**Before:**
```python
"content": result["documents"][0] if result.get("documents") else "",
```

**After:**
```python
documents = result.get("documents", [])
"content": documents[0] if documents else "",
```

---

## 버그 수정 요약

| 버그 유형 | 심각도 | 수정 파일 수 |
|----------|--------|-------------|
| Path Traversal | High | 1 |
| IndexError (API 응답) | High | 3 |
| IndexError (JSON 파싱) | Medium | 1 |
| 예외 무시 (로깅 없음) | Medium | 3 |
| Race Condition | Medium | 1 |
| JSON 파일 로드 | Medium | 1 |
| VectorStore IndexError | Medium | 1 |

---

## 검증

```bash
# 문법 검사 통과 (모든 수정 파일)
python3 -m py_compile \
  app/modules/small_claims/router/__init__.py \
  app/common/chat_service.py \
  app/modules/storyboard/router/__init__.py \
  app/modules/storyboard/service/image_generation.py \
  app/modules/storyboard/service/video_generation.py \
  app/modules/storyboard/service/vision.py \
  app/modules/storyboard/service/__init__.py \
  app/modules/storyboard/service/job_manager.py \
  app/modules/lawyer_finder/router/__init__.py \
  app/modules/lawyer_finder/service/__init__.py \
  app/modules/case_precedent/router/__init__.py \
  app/modules/multi_agent/agents/small_claims_agent.py \
  app/common/vectorstore/base.py
# ✅ 모든 파일 통과
```
