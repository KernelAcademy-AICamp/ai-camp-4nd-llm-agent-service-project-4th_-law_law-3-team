---
name: prompt-engineering
description: |
  LangGraph 멀티에이전트 법률 상담 시스템의 프롬프트 작성, 최적화, 관리 가이드.
  시스템 프롬프트 설계, 역할 기반 프롬프트, RAG 컨텍스트 구성, 쿼리 리라이팅, 온도 조절.
  에이전트 프롬프트 수정, 새 에이전트 추가, 프롬프트 품질 개선, 응답 품질 문제 해결 시 반드시 사용.
  모의 법정 역할 프롬프트, 소액소송 단계별 메시지, 법률 검색 쿼리 최적화 시에도 사용.
---

# Prompt Engineering

법률 서비스 멀티에이전트의 프롬프트 설계/최적화/관리 패턴.

## 현재 프롬프트 아키텍처

```
┌─────────────────────────────────────────────┐
│ RulesRouter (의도 감지 → 에이전트 선택)       │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┬────────┐
    ▼          ▼          ▼          ▼        ▼
 Legal     SmallClaims  Storyboard  LawStudy  MockTrial
 Search    (상태 머신)  (타임라인)  (교육)   (역할 기반)
 (RAG+LLM)                                  (5개 역할)
```

### 프롬프트 저장 위치

| 위치 | 내용 | 에이전트 |
|------|------|---------|
| `multi_agent/agents/*.py` | 모듈 내 `_SYSTEM_PROMPT` 상수 | LegalSearch, Storyboard, LawStudy |
| `multi_agent/subgraphs/mock_trial_prompts.py` | 역할별 프롬프트 dict | MockTrial (5역할) |
| `multi_agent/agents/small_claims_agent.py` | `STEP_MESSAGES` dict | SmallClaims |
| `tools/prompts/__init__.py` | 레거시 프롬프트 풀 | (일부 참조) |

---

## 1. 시스템 프롬프트 설계 원칙

### 구조 템플릿

```python
_SYSTEM_PROMPT = """당신은 {역할}입니다.

## 핵심 임무
{에이전트가 수행해야 할 핵심 작업 1-2문장}

## 답변 규칙
1. {구체적인 행동 지침}
2. {금지 사항}
3. {형식 요구사항}

## 도메인 지식
- {법률 용어 사용 규칙}
- {인용 형식}

## 면책 고지
이 답변은 법률 자문이 아닌 일반적인 정보 제공 목적입니다.
실제 법률 문제는 변호사와 상담하시기 바랍니다."""
```

### 설계 원칙

| 원칙 | 설명 | 예시 |
|------|------|------|
| **역할 명확화** | 첫 문장에서 역할 정의 | "당신은 한국 법률 판례 검색 전문 AI입니다" |
| **행동 구체화** | 모호한 지시 대신 구체적 행동 | "판례번호를 반드시 포함하세요" (O) vs "자세히 답하세요" (X) |
| **제약 조건** | 하면 안 되는 것 명시 | "확인되지 않은 법조문을 인용하지 마세요" |
| **출력 형식** | 기대하는 응답 구조 | "1. 관련 법령 → 2. 판례 → 3. 실무 조언" |
| **면책 고지** | 법률 서비스 필수 | 모든 에이전트에 포함 |

---

## 2. 프롬프트 패턴별 가이드

### Pattern A: RAG + LLM 조합 (LegalSearch, LawStudy)

```python
# 메시지 구성 순서
messages = [
    ("system", _SYSTEM_PROMPT),           # 1. 시스템 프롬프트
    *history_to_messages(history),         # 2. 대화 히스토리
    ("user", f"""참고 자료:                # 3. RAG 컨텍스트 + 사용자 질문
{context}

사용자 질문: {message}"""),
]
response = model.invoke(messages)
```

**핵심 규칙:**
- 시스템 프롬프트와 사용자 입력은 항상 분리 (Prompt Injection 방어)
- RAG 컨텍스트는 사용자 메시지 내에 포함 (시스템 프롬프트에 삽입 금지)
- 히스토리는 시스템과 사용자 사이에 위치

**컨텍스트 구성 예시:**

```python
def format_context(results: list[dict]) -> str:
    """검색 결과를 LLM 친화적 컨텍스트로 변환"""
    parts = []
    for i, r in enumerate(results, 1):
        doc_type = r.get("doc_type", "기타")
        title = r.get("title", "제목 없음")
        content = r.get("content", "")[:2000]  # 길이 제한

        parts.append(f"[참고자료 {i}] ({doc_type}) {title}\n{content}")

    return "\n\n---\n\n".join(parts)
```

### Pattern B: 역할 기반 프롬프트 (MockTrial)

```python
# mock_trial_prompts.py 구조
SYSTEM_PROMPTS: dict[tuple[str, str], str] = {
    ("criminal", "judge"):      JUDGE_SYSTEM_PROMPT,
    ("criminal", "prosecutor"): PROSECUTOR_CRIMINAL_PROMPT,
    ("criminal", "attorney"):   ATTORNEY_CRIMINAL_PROMPT,
    ("criminal", "defendant"):  DEFENDANT_PERSON_PROMPT,
    ("criminal", "clerk"):      CLERK_SYSTEM_PROMPT,
    ("civil", "judge"):         JUDGE_SYSTEM_PROMPT,
    ("civil", "prosecutor"):    PLAINTIFF_CIVIL_PROMPT,
    ("civil", "attorney"):      DEFENDANT_CIVIL_PROMPT,
    ("civil", "defendant"):     DEFENDANT_PERSON_PROMPT,
    ("civil", "clerk"):         CLERK_SYSTEM_PROMPT,
}
```

**역할별 설계:**

| 역할 | 핵심 특성 | 온도 | 이유 |
|------|----------|------|------|
| 판사 | 공정, 중립, 절차 준수 | 0.3 | 일관된 결정 |
| 검사/원고 | 논리적 주장, 증거 제시 | 0.7 | 균형 잡힌 논리 |
| 변호인/피고 대리인 | 방어 논리, 반론 | 0.7 | 균형 잡힌 논리 |
| 피고인 | 감정적, 진술 | 0.8 | 자연스러운 표현 |
| 서기 | 기록, 정리 | 0.2 | 객관적 기록 |

**역할 프롬프트 작성 팁:**
- 행동 양식 구체화: "발언은 간결하고 권위 있게 합니다" (판사)
- 절차 준수 명시: "형사소송법에 따른 절차를 엄격히 준수합니다"
- 역할 일관성: 한 역할이 다른 역할의 발언을 하지 않도록 제약

### Pattern C: 상태 머신 프롬프트 (SmallClaims)

```python
# 단계별 메시지 정의
STEP_MESSAGES = {
    SmallClaimsStep.INIT: "소액소송 절차를 안내해드리겠습니다...",
    SmallClaimsStep.GATHER_INFO: "다음 정보를 알려주세요...",
    SmallClaimsStep.ANALYZE: "입력하신 정보를 분석합니다...",
    SmallClaimsStep.GUIDE: "소송 절차를 안내합니다...",
    SmallClaimsStep.DOCUMENT: "필요한 서류를 안내합니다...",
}
```

**설계 규칙:**
- 각 단계 프롬프트는 독립적으로 이해 가능해야 함
- 다음 단계로의 전이 조건을 명확히
- 사용자 입력 부족 시 재요청 메시지 포함

---

## 3. 쿼리 리라이팅

### 현재 구현 (`services/rag/query_rewrite.py`)

```python
# LLM 기반 쿼리 확장
prompt = f"""다음 법률 검색 쿼리를 {num_queries - 1}개의 다른 표현으로 바꿔주세요.
각 쿼리는 같은 의미를 가지되 다른 단어나 표현을 사용해야 합니다.
법률 용어와 일상 용어를 적절히 혼용해주세요.

원본 쿼리: {query}

다음 형식으로 출력하세요 (번호와 쿼리만, 설명 없이):
1. [확장된 쿼리 1]
2. [확장된 쿼리 2]
..."""
```

### 리라이팅 최적화 팁

| 기법 | 설명 | 예시 |
|------|------|------|
| **법률 용어 확장** | 일상어 → 법률 용어 | "해고" → "부당해고", "근로계약 해지" |
| **일상어 변환** | 법률 용어 → 일상어 | "채무불이행" → "빚 안 갚음" |
| **구체화** | 추상적 → 구체적 | "이혼" → "이혼 재산분할 기준", "양육권 결정" |
| **문맥 보존** | 대화형 쿼리 처리 | "그건 어떻게 되나요?" → (히스토리 참조) 독립 쿼리로 |

### 쿼리 리라이팅 (`rewrite_query`)

```python
from app.services.rag.query_rewrite import rewrite_query

# LLM 기반: 일상 표현 → 법률 용어 변환, 자연스러운 문장 형태
queries = rewrite_query("당근마켓에서 사기당했어", use_llm=True)
# → ["중고거래 사기 피해에 대한 형사고소 및 손해배상청구 절차"]

# 키워드 기반 (LLM 미사용)
queries = rewrite_query("사기당했어", use_llm=False)
# → ["사기당했어 사기 소송"]
```

---

## 4. 온도(Temperature) 설계

### 에이전트별 권장 온도

| 에이전트 | 온도 | 근거 |
|---------|------|------|
| LegalSearch | 0.3 | 사실 기반 정확한 인용 |
| LawStudy | 0.5 | 교육적 설명 + 정확성 |
| Storyboard | 0.5 | 타임라인 구성 + 창의성 |
| SmallClaims | 0.3 | 절차 안내 정확성 |
| MockTrial 판사 | 0.3 | 일관된 결정 |
| MockTrial 검사/변호인 | 0.7 | 논리 + 적절한 변형 |
| MockTrial 피고인 | 0.8 | 자연스러운 감정 표현 |
| MockTrial 서기 | 0.2 | 객관적 기록 |
| Query Rewrite | 0.3 | 안정적인 쿼리 변환 |
| SimpleChatAgent | 0.7 | 일반 대화 |

### 온도 선택 기준

```
정확성 중요 ← 0.0 ─ 0.3 ─ 0.5 ─ 0.7 ─ 1.0 → 창의성 중요
                 │         │         │
            법조문 인용  교육/설명  일반 대화
            판결문 작성  논리 구성  캐릭터 연기
```

---

## 5. 프롬프트 품질 체크리스트

### 새 에이전트 프롬프트 작성 시

- [ ] 역할이 첫 문장에 명확히 정의됨
- [ ] 핵심 임무가 1-2문장으로 요약됨
- [ ] 구체적인 행동 지침 (모호한 표현 없음)
- [ ] 금지 사항 명시 (하면 안 되는 것)
- [ ] 출력 형식 정의 (구조화된 응답)
- [ ] 법률 면책 고지 포함
- [ ] 온도 설정 근거 명확

### 프롬프트 수정 시

- [ ] 기존 응답 품질 문제 구체적 파악
- [ ] 수정 전/후 비교 가능한 테스트 쿼리 준비
- [ ] 엣지 케이스 테스트 (짧은 질문, 모호한 질문, 다국어)
- [ ] 다른 에이전트와의 일관성 확인

### 응답 품질 문제 진단

| 증상 | 원인 | 해결 |
|------|------|------|
| 너무 긴 응답 | 길이 제약 없음 | "300자 이내로 요약하세요" 추가 |
| 환각 (없는 판례 인용) | 컨텍스트 무시 | "제공된 참고 자료만 인용하세요" 강화 |
| 역할 이탈 | 프롬프트 약함 | 역할 제약 강화 + 온도 낮춤 |
| 법률 용어 과다 | 대상 독자 미명시 | "법률 비전문가도 이해할 수 있게" 추가 |
| 면책 고지 누락 | 프롬프트에 없음 | 시스템 프롬프트 마지막에 필수 포함 |

---

## 6. 새 에이전트 프롬프트 작성 절차

### Step 1: 역할 정의

```python
_SYSTEM_PROMPT = """당신은 한국 {전문 분야} 전문 AI 어시스턴트입니다.
{에이전트가 제공하는 핵심 가치 1문장}"""
```

### Step 2: 행동 규칙 정의

```python
## 답변 규칙
1. 제공된 참고 자료를 기반으로 답변하세요
2. 관련 법령이나 판례가 있다면 구체적으로 인용하세요
3. 법률 용어는 괄호 안에 쉬운 설명을 추가하세요
4. 확실하지 않은 내용은 "확인이 필요합니다"로 명시하세요
5. 모든 답변 마지막에 면책 고지를 포함하세요
```

### Step 3: 온도 설정

```python
# tools/llm/__init__.py 의 get_chat_model() 사용
model = get_chat_model(temperature=0.3)  # 정확성 우선
```

### Step 4: 메시지 구성

```python
async def _generate_response(
    self,
    message: str,
    context: str,
    history: list[dict],
) -> str:
    messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]

    # 히스토리 추가 (최근 N턴)
    for h in history[-10:]:
        messages.append(("user", h["user"]))
        messages.append(("assistant", h["assistant"]))

    # RAG 컨텍스트 + 사용자 질문
    user_msg = f"참고 자료:\n{context}\n\n사용자 질문: {message}"
    messages.append(("user", user_msg))

    response = await model.ainvoke(messages)
    return response.content
```

---

## 7. LLM 프로바이더 설정

### 환경변수

```python
# backend/app/core/config.py
LLM_PROVIDER: str = "openai"      # openai | anthropic | google
OPENAI_MODEL: str = "gpt-4o-mini"
ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
GOOGLE_MODEL: str = "gemini-3-flash-preview"
LLM_TIMEOUT_SECONDS: int = 60
AGENT_TIMEOUT_SECONDS: int = 120
```

### 모델 선택 가이드

| 용도 | 권장 모델 | 이유 |
|------|----------|------|
| 법률 검색 응답 | gpt-4o-mini | 비용 효율 + 충분한 품질 |
| 복잡한 법률 분석 | gpt-4o / claude-3-5-sonnet | 높은 추론 능력 |
| 쿼리 리라이팅 | gpt-4o-mini | 단순 변환, 비용 절감 |
| 모의 법정 | gpt-4o | 역할 연기 + 법률 지식 |

---

## 8. 프롬프트 디버깅

### 응답 품질 문제 추적

```python
import logging
logger = logging.getLogger(__name__)

# 프롬프트 입출력 로깅
logger.debug(f"System prompt length: {len(_SYSTEM_PROMPT)}")
logger.debug(f"Context length: {len(context)}")
logger.debug(f"History turns: {len(history)}")
logger.debug(f"User message: {message[:100]}...")
logger.debug(f"Response length: {len(response)}")
```

### 컨텍스트 윈도우 관리

| 구성 요소 | 예상 토큰 | 관리 방법 |
|----------|----------|----------|
| 시스템 프롬프트 | 500~1000 | 간결하게 유지 |
| 히스토리 | 가변 | 최근 10턴 제한 |
| RAG 컨텍스트 | 2000~4000 | 청크 크기 + top-K 제한 |
| 사용자 메시지 | ~500 | 길이 제한 적용 |
| **합계** | ~6000 | 모델 한도의 50% 이하 권장 |
