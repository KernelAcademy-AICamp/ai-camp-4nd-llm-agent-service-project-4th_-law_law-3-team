# compression-mecab-force-tokens Planning Document

> **Summary**: MeCab userdic 기반 동적 force_tokens 추출로 LLMLingua-2 한국어 압축 품질 개선
>
> **Project**: Law Platform (RAG Pipeline)
> **Author**: Claude
> **Date**: 2026-02-26
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

LLMLingua-2의 SentencePiece 토크나이저가 한국어 법률 복합명사를 과도하게 서브워드 분할하여 압축 시 핵심 법률 용어가 탈락하는 문제를 해결한다.

**현재 문제:**
- `손해배상` → `▁손`, `해`, `배`, `상` (4토큰) → 일부 서브워드 탈락
- `대법원` → `대` + `법원` → `대` 탈락
- `민법 제750조` → `민법` + `제` + `750` + `조` → `조` 탈락

### 1.2 Background

- LLMLingua-2는 XLM-RoBERTa 기반 토큰 분류 모델로, SentencePiece(Unigram) 토크나이저 사용
- SentencePiece는 100개 언어 공유 어휘(250K)를 사용하므로 한국어 할당량이 제한적
- 현재 정적 `FORCE_TOKENS` (~25개)로는 법률 도메인의 다양한 복합명사 커버 불가
- 프로젝트에 이미 MeCab userdic (37,366+ 법률 복합명사)이 구축되어 있음

### 1.3 Related Documents

- `docs/01-plan/features/llmlingua-context-compression.plan.md` (LLMLingua-2 기본 구현)
- `backend/app/services/rag/compression.py` (현재 구현)
- `backend/app/tools/vectorstore/mecab_tokenizer.py` (MeCab 토크나이저)

---

## 2. Scope

### 2.1 In Scope

- [x] 압축 전 MeCab으로 입력 텍스트에서 법률 복합명사 동적 추출
- [x] 추출된 명사를 해당 텍스트의 force_tokens에 동적 추가
- [x] MeCab 미설치 시 RuntimeError 발생 (프로젝트 필수 의존성과 일관)
- [x] 테스트 스크립트 업데이트 (동적 추출 검증)

### 2.2 Out of Scope

- MeCab 토크나이저 자체 수정
- LLMLingua-2 모델 파인튜닝
- 새로운 환경변수/Feature Flag 추가 (기존 `ENABLE_CONTEXT_COMPRESSION`으로 충분)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | compress_field() 호출 전 MeCab으로 텍스트에서 명사 추출 | High | Pending |
| FR-02 | 추출된 명사를 force_tokens에 동적 병합 (정적 + 동적) | High | Pending |
| FR-03 | MeCab 미설치/userdic 미빌드 시 RuntimeError 발생 | High | Pending |
| FR-04 | 테스트에서 `민법 제750조`, `대법원` 보존 확인 | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | MeCab 추출 오버헤드 <5ms/문서 | 테스트 스크립트 시간 측정 |
| Reliability | MeCab 미설치 시 명확한 에러 메시지 | RuntimeError 발생 확인 |
| Quality | 핵심 용어 보존율 4/6 → 6/6 개선 | 테스트 스크립트 검증 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [x] `_extract_legal_nouns()` 함수 구현
- [x] `compress_field()`에 동적 force_tokens 통합
- [x] MeCab 연동 테스트 통과
- [x] 테스트 스크립트에서 `민법 제750조`, `대법원` 보존 확인
- [x] ruff check + mypy 통과

### 4.2 Quality Criteria

- [x] 기존 테스트 결과 유지 (ruling 보존, 법조문 번호 5/5)
- [x] 핵심 용어 보존율 향상 (목표: 6/6)
- [x] 압축 소요 시간 증가 <5ms

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MeCab 미설치 환경 | Medium | Low | 프로젝트 필수 의존성 (FTS도 MeCab 필수), RuntimeError로 명확한 안내 |
| 동적 토큰 과다 → 압축률 저하 | Low | Low | MeCab morphs()가 이미 명사만 반환 (10~30개/문서) |
| 스레드 안전성 | Medium | Low | 기존 _get_thread_tokenizer() 활용 (threading.local) |

---

## 6. Architecture

### 6.1 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `app/services/rag/compression.py` | `_extract_legal_nouns()` 추가, `compress_field()` 수정 |
| `scripts/test_compression.py` | 동적 force_tokens 검증 추가 |

### 6.2 데이터 흐름

```
입력 텍스트
    ↓
MeCab 형태소 분석 (<1ms)
    ↓
법률 복합명사 추출 (명사만, 2자 이상)
    ↓
정적 FORCE_TOKENS + 동적 명사 병합
    ↓
LLMLingua-2 압축 (force_tokens 적용)
    ↓
압축된 텍스트
```

### 6.3 핵심 구현

```python
def _extract_legal_nouns(text: str) -> list[str]:
    """MeCab userdic으로 텍스트에서 법률 복합명사 추출.

    MeCab 미설치 시 RuntimeError 발생 (프로젝트 필수 의존성).
    """
    from app.tools.vectorstore.lancedb import _get_thread_tokenizer
    tokenizer = _get_thread_tokenizer()
    return tokenizer.morphs(text)

def compress_field(self, text, policy):
    # ... 기존 스킵 로직 ...
    dynamic_tokens = _extract_legal_nouns(text)
    all_force_tokens = FORCE_TOKENS + list(policy.force_tokens) + dynamic_tokens
    # ... LLMLingua-2 압축 ...
```

---

## 7. Next Steps

1. [x] Design 문서 작성 (생략 가능 - 단순 변경)
2. [x] 구현
3. [x] 테스트 검증
4. [x] ruff + mypy 통과

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-26 | Initial draft | Claude |
