# LanceDB Test Suite Design Document

> **Summary**: LanceDB 벡터 DB 테스트 스위트의 구체적 구현 설계 - 스키마, 저장소, MeCab FTS, 하이브리드 검색
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-05
> **Status**: Draft
> **Plan Reference**: `docs/01-plan/features/lancedb-test.plan.md` (v0.2)

---

## 1. Overview

### 1.1 Purpose

Plan 문서(FR-01~FR-28)에 정의된 테스트 요구사항을 **구체적인 코드 수준 설계**로 변환한다. 각 테스트 파일의 구현 상세, 픽스처 설계, MeCab 토크나이저 유틸리티 인터페이스, FTS 통합 패턴을 명세한다.

### 1.2 Scope

| 산출물 | 파일 경로 | 대상 FR |
|--------|----------|---------|
| 스키마 단위 테스트 | `backend/tests/unit/test_vectorstore_schema.py` | FR-01~04, 18 |
| 저장소 단위 테스트 | `backend/tests/unit/test_lancedb_store.py` | FR-05~16, 19 |
| MeCab 토크나이저 테스트 | `backend/tests/unit/test_mecab_tokenizer.py` | FR-20~22, 27 |
| 통합 테스트 | `backend/tests/integration/test_lancedb_integration.py` | FR-17, 23~26, 28 |
| MeCab 유틸리티 | `backend/app/tools/vectorstore/mecab_tokenizer.py` | 신규 모듈 |
| conftest 업데이트 | `backend/tests/conftest.py` | 픽스처 + 마커 추가 |

---

## 2. Architecture

### 2.1 테스트 대상 모듈 의존관계

```
backend/app/tools/vectorstore/
├── base.py          ← SearchResult, VectorStoreBase (ABC)
├── schema_v2.py     ← LEGAL_CHUNKS_SCHEMA, LegalChunk, create_*_chunk()
├── lancedb.py       ← LanceDBStore(VectorStoreBase)
├── mecab_tokenizer.py ← MeCabTokenizer (신규)
└── __init__.py      ← get_vector_store() 팩토리
```

### 2.2 테스트 아키텍처

```
backend/tests/
├── conftest.py                          # 공유 픽스처 (기존 + LanceDB + MeCab 추가)
├── unit/
│   ├── test_vectorstore_schema.py       # 순수 함수 테스트 (DB 불필요)
│   ├── test_lancedb_store.py            # 실제 LanceDB (tmp_path)
│   └── test_mecab_tokenizer.py          # MeCab 유틸리티 테스트
└── integration/
    └── test_lancedb_integration.py      # E2E + FTS + 하이브리드
```

### 2.3 테스트 격리 전략

| 전략 | 적용 대상 | 구현 방법 |
|------|----------|----------|
| 임시 디렉토리 | LanceDBStore | `tmp_path` + `monkeypatch`로 `LANCEDB_URI` 오버라이드 |
| 랜덤 벡터 (고정 시드) | 모든 벡터 데이터 | `numpy.random.seed(42)`, `VECTOR_DIM=1024` |
| 테이블명 격리 | LanceDB 테이블 | `test_legal_chunks_{uuid4().hex[:8]}` |
| 조건부 skip | MeCab 미설치 | `@pytest.mark.requires_mecab` + `importlib` 체크 |
| 조건부 skip | FTS 테스트 | `@pytest.mark.requires_fts` |

---

## 3. Data Model

### 3.1 테스트용 샘플 데이터 스키마

#### 샘플 법령 데이터 (3건, 각 2청크 = 6레코드)

```python
SAMPLE_LAW_DATA = [
    {
        "source_id": "010719",
        "title": "민법",
        "contents": [
            "[법령] 민법 제750조(불법행위의 내용): 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
            "[법령] 민법 제751조(재산 이외의 손해의 배상): 타인의 신체, 자유 또는 명예를 해하거나 기타 정신상 고통을 가한 자는 재산 이외의 손해에 대하여도 배상할 책임이 있다.",
        ],
        "enforcement_date": "2023-08-08",
        "department": "법무부",
        "law_type": "법률",
        "promulgation_date": "20230808",
        "promulgation_no": "19592",
        "article_nos": ["제750조", "제751조"],
    },
    {
        "source_id": "010720",
        "title": "상법",
        "contents": [
            "[법령] 상법 제1조(상사에 대한 통칙): 상사에 관하여 본법에 규정이 없으면 상관습법에 의하고 상관습법이 없으면 민법의 규정에 의한다.",
            "[법령] 상법 제2조(상인): 자기 명의로 상행위를 하는 자를 상인이라 한다.",
        ],
        "enforcement_date": "2023-01-01",
        "department": "법무부",
        "law_type": "법률",
        "promulgation_date": "20230101",
        "promulgation_no": "19000",
        "article_nos": ["제1조", "제2조"],
    },
    {
        "source_id": "010721",
        "title": "도로교통법",
        "contents": [
            "[법령] 도로교통법 제1조(목적): 이 법은 도로에서 일어나는 교통상의 모든 위험과 장해를 방지하고 제거하여 안전하고 원활한 교통을 확보함을 목적으로 한다.",
            "[법령] 도로교통법 제2조(정의): 이 법에서 사용하는 용어의 뜻은 다음과 같다.",
        ],
        "enforcement_date": "2024-01-01",
        "department": "경찰청",
        "law_type": "법률",
        "promulgation_date": "20240101",
        "promulgation_no": "20000",
        "article_nos": ["제1조", "제2조"],
    },
]
```

#### 샘플 판례 데이터 (3건, 각 2청크 = 6레코드)

```python
SAMPLE_PRECEDENT_DATA = [
    {
        "source_id": "76396",
        "title": "손해배상(기)",
        "contents": [
            "[판례] 손해배상청구사건 - 불법행위로 인한 손해배상의 범위에 관한 판결. 피고인의 고의 또는 과실로 인한 불법행위가 인정되므로 손해배상 책임이 있다.",
            "[판례] 판결요지 - 민법 제750조에 의하여 불법행위의 성립요건으로서 가해행위와 손해 사이에 인과관계가 인정되어야 한다.",
        ],
        "decision_date": "2023-05-15",
        "court_name": "대법원",
        "case_number": "2023다12345",
        "case_type": "민사",
        "judgment_type": "판결",
        "judgment_status": "확정",
        "reference_provisions": "민법 제750조, 제751조",
        "reference_cases": None,
    },
    {
        "source_id": "76397",
        "title": "교통사고처리특례법위반",
        "contents": [
            "[판례] 교통사고처리특례법위반 사건 - 피고인이 도로교통법을 위반하여 교통사고를 발생시킨 사건.",
            "[판례] 판결요지 - 도로교통법 위반 여부와 과실 비율에 관한 판단.",
        ],
        "decision_date": "2023-07-20",
        "court_name": "서울중앙지방법원",
        "case_number": "2023고단6789",
        "case_type": "형사",
        "judgment_type": "판결",
        "judgment_status": "확정",
        "reference_provisions": "도로교통법 제1조, 제2조",
        "reference_cases": "2023다12345",
    },
    {
        "source_id": "76398",
        "title": "매매대금반환",
        "contents": [
            "[판례] 매매대금반환 청구사건 - 상법상 상행위에 해당하는 매매계약의 해제와 원상회복 의무.",
            "[판례] 판결요지 - 상법 제1조에 따른 상관습법의 적용 범위에 관한 판단.",
        ],
        "decision_date": "2024-01-10",
        "court_name": "서울고등법원",
        "case_number": "2023나45678",
        "case_type": "민사",
        "judgment_type": "판결",
        "judgment_status": "미확정",
        "reference_provisions": "상법 제1조",
        "reference_cases": None,
    },
]
```

### 3.2 벡터 생성 전략

```python
import numpy as np

VECTOR_DIM = 1024

def make_vector(seed: int = 42) -> list[float]:
    """고정 시드 1024차원 랜덤 벡터 (L2 정규화)"""
    rng = np.random.RandomState(seed)
    vec = rng.randn(VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # 정규화
    return vec.tolist()

def make_similar_vector(base_seed: int = 42, noise: float = 0.1) -> list[float]:
    """base_seed 벡터와 유사한 벡터 생성 (코사인 유사도 검증용)"""
    rng_base = np.random.RandomState(base_seed)
    base = rng_base.randn(VECTOR_DIM).astype(np.float32)
    rng_noise = np.random.RandomState(base_seed + 1000)
    perturbation = rng_noise.randn(VECTOR_DIM).astype(np.float32) * noise
    vec = base + perturbation
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()
```

---

## 4. MeCab Tokenizer Design (신규 모듈)

### 4.1 파일 위치

`backend/app/tools/vectorstore/mecab_tokenizer.py`

### 4.2 인터페이스

```python
"""
MeCab 기반 한국어 형태소 분석기 (LanceDB FTS 사전 토크나이징용)

LanceDB는 한국어 네이티브 FTS 토크나이저를 미지원(PR #2855 미머지)하므로,
MeCab으로 사전 토크나이징한 텍스트를 FTS 인덱싱하는 전략을 채택한다.

Usage:
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    tokenizer = MeCabTokenizer()
    tokenized = tokenizer.tokenize("손해배상청구")
    # → "손해 배상 청구"

    # FTS 검색 쿼리 사전 토크나이징
    query = tokenizer.tokenize_query("불법행위 손해배상")
    # → "불법 행위 손해 배상"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# MeCab 설치 여부 확인
_MECAB_AVAILABLE = False
try:
    import MeCab
    _MECAB_AVAILABLE = True
except ImportError:
    pass


def is_mecab_available() -> bool:
    """MeCab 설치 여부 확인"""
    return _MECAB_AVAILABLE


class MeCabTokenizer:
    """
    MeCab 한국어 형태소 분석기

    사전 토크나이징 전략:
    1. content → MeCab 형태소 분석 → 공백 구분 토큰 문자열
    2. content_tokenized 컬럼에 저장
    3. LanceDB FTS 인덱스는 content_tokenized에 생성
    4. 검색 쿼리도 동일하게 토크나이징 후 FTS 검색

    MeCab 미설치 시 공백 분리 fallback 동작.
    """

    def __init__(self) -> None:
        self._tagger: Optional["MeCab.Tagger"] = None
        if _MECAB_AVAILABLE:
            self._tagger = MeCab.Tagger()

    @property
    def is_available(self) -> bool:
        """MeCab 토크나이저 사용 가능 여부"""
        return self._tagger is not None

    def morphs(self, text: str) -> list[str]:
        """
        형태소 분석 결과를 리스트로 반환

        Args:
            text: 분석할 한국어 텍스트

        Returns:
            형태소 리스트 (예: ["손해", "배상", "청구"])
            MeCab 미설치 시 공백 분리 결과 반환
        """
        if not text or not text.strip():
            return []

        if self._tagger is None:
            logger.warning("MeCab 미설치: 공백 분리 fallback 사용")
            return text.strip().split()

        # MeCab 형태소 분석
        parsed = self._tagger.parse(text)
        morphs_list: list[str] = []
        for line in parsed.strip().split("\n"):
            if line == "EOS" or line == "":
                continue
            token = line.split("\t")[0]
            if token.strip():
                morphs_list.append(token)

        return morphs_list

    def tokenize(self, text: str) -> str:
        """
        텍스트를 형태소 분석하여 공백 구분 문자열로 반환

        Args:
            text: 원본 텍스트

        Returns:
            공백 구분 형태소 문자열 (예: "손해 배상 청구")
        """
        return " ".join(self.morphs(text))

    def tokenize_query(self, query: str) -> str:
        """
        검색 쿼리를 형태소 분석 (tokenize의 별칭, 의미 구분용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            공백 구분 형태소 문자열
        """
        return self.tokenize(query)
```

### 4.3 `__init__.py` export 추가

```python
# backend/app/tools/vectorstore/__init__.py에 추가
from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer, is_mecab_available
```

---

## 5. Fixture Design (conftest.py 추가분)

### 5.1 새 마커 등록

```python
# conftest.py pytest_configure()에 추가
config.addinivalue_line(
    "markers",
    "requires_mecab: marks tests that require MeCab installation"
)
config.addinivalue_line(
    "markers",
    "requires_fts: marks tests that require FTS index support"
)
```

### 5.2 LanceDB 픽스처

```python
import numpy as np
from app.tools.vectorstore.schema_v2 import VECTOR_DIM


@pytest.fixture
def random_vector() -> list[float]:
    """고정 시드 1024차원 정규화 벡터"""
    rng = np.random.RandomState(42)
    vec = rng.randn(VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def make_random_vector():
    """시드를 지정하여 랜덤 벡터를 생성하는 팩토리 픽스처"""
    def _make(seed: int = 42) -> list[float]:
        rng = np.random.RandomState(seed)
        vec = rng.randn(VECTOR_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    return _make


@pytest.fixture
def lancedb_store(tmp_path, monkeypatch):
    """
    임시 디렉토리 기반 격리된 LanceDBStore 인스턴스

    monkeypatch로 settings를 오버라이드하여 테스트 격리.
    테스트 종료 후 tmp_path가 자동 삭제됨.
    """
    from app.core.config import settings

    lancedb_dir = tmp_path / "lancedb_test"
    lancedb_dir.mkdir()

    monkeypatch.setattr(settings, "LANCEDB_URI", str(lancedb_dir))
    monkeypatch.setattr(settings, "LANCEDB_TABLE_NAME", "test_legal_chunks")

    from app.tools.vectorstore.lancedb import LanceDBStore
    store = LanceDBStore()
    yield store

    # 정리: reset() 호출로 테이블 삭제
    store.reset()


@pytest.fixture
def populated_store(lancedb_store, make_random_vector):
    """
    샘플 데이터가 미리 삽입된 LanceDBStore

    법령 3건(6청크) + 판례 3건(6청크) = 총 12레코드
    """
    # 법령 3건 추가
    lancedb_store.add_law_documents(
        source_ids=["010719", "010719", "010720", "010720", "010721", "010721"],
        chunk_indices=[0, 1, 0, 1, 0, 1],
        embeddings=[make_random_vector(i) for i in range(6)],
        titles=["민법", "민법", "상법", "상법", "도로교통법", "도로교통법"],
        contents=[
            "[법령] 민법 제750조: 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
            "[법령] 민법 제751조: 타인의 신체, 자유 또는 명예를 해한 자는 재산 이외의 손해에 대하여도 배상할 책임이 있다.",
            "[법령] 상법 제1조: 상사에 관하여 본법에 규정이 없으면 상관습법에 의한다.",
            "[법령] 상법 제2조: 자기 명의로 상행위를 하는 자를 상인이라 한다.",
            "[법령] 도로교통법 제1조: 도로에서 일어나는 교통상의 모든 위험과 장해를 방지한다.",
            "[법령] 도로교통법 제2조: 이 법에서 사용하는 용어의 뜻은 다음과 같다.",
        ],
        enforcement_dates=["2023-08-08"] * 2 + ["2023-01-01"] * 2 + ["2024-01-01"] * 2,
        departments=["법무부"] * 4 + ["경찰청"] * 2,
        total_chunks_list=[2, 2, 2, 2, 2, 2],
        law_types=["법률"] * 6,
    )

    # 판례 3건 추가
    lancedb_store.add_precedent_documents(
        source_ids=["76396", "76396", "76397", "76397", "76398", "76398"],
        chunk_indices=[0, 1, 0, 1, 0, 1],
        embeddings=[make_random_vector(i + 100) for i in range(6)],
        titles=[
            "손해배상(기)", "손해배상(기)",
            "교통사고처리특례법위반", "교통사고처리특례법위반",
            "매매대금반환", "매매대금반환",
        ],
        contents=[
            "[판례] 불법행위로 인한 손해배상의 범위에 관한 판결.",
            "[판례] 민법 제750조에 의하여 불법행위의 성립요건으로서 인과관계가 인정되어야 한다.",
            "[판례] 교통사고처리특례법위반 사건 - 도로교통법 위반.",
            "[판례] 도로교통법 위반 여부와 과실 비율에 관한 판단.",
            "[판례] 상법상 상행위에 해당하는 매매계약의 해제와 원상회복 의무.",
            "[판례] 상법 제1조에 따른 상관습법의 적용 범위에 관한 판단.",
        ],
        decision_dates=["2023-05-15"] * 2 + ["2023-07-20"] * 2 + ["2024-01-10"] * 2,
        court_names=["대법원"] * 2 + ["서울중앙지방법원"] * 2 + ["서울고등법원"] * 2,
        total_chunks_list=[2, 2, 2, 2, 2, 2],
        case_numbers=["2023다12345"] * 2 + ["2023고단6789"] * 2 + ["2023나45678"] * 2,
        case_types=["민사"] * 2 + ["형사"] * 2 + ["민사"] * 2,
    )

    return lancedb_store
```

### 5.3 MeCab 픽스처

```python
@pytest.fixture
def mecab_tokenizer():
    """
    MeCab 토크나이저 (미설치 시 테스트 skip)

    @pytest.mark.requires_mecab과 함께 사용
    """
    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer, is_mecab_available
    if not is_mecab_available():
        pytest.skip("MeCab이 설치되지 않았습니다")
    return MeCabTokenizer()


@pytest.fixture
def sample_legal_texts() -> dict[str, str]:
    """법률 도메인 샘플 텍스트 (MeCab 토크나이징 테스트용)"""
    return {
        "tort": "불법행위로 인한 손해배상청구권의 소멸시효",
        "article_ref": "민법 제750조에 의한 손해배상 책임",
        "case_number": "대법원 2023다12345 판결",
        "mixed": "OWASP Top 10 보안 취약점 분석",
        "traffic": "도로교통법 위반으로 인한 교통사고처리특례법 적용",
    }
```

---

## 6. Test File Specifications

### 6.1 `test_vectorstore_schema.py` (15 tests)

**대상**: `backend/app/tools/vectorstore/schema_v2.py`
**의존성**: 없음 (순수 함수 테스트, DB 불필요)

```python
"""
schema_v2.py 단위 테스트

대상:
- create_law_chunk(): 법령 청크 생성 헬퍼
- create_precedent_chunk(): 판례 청크 생성 헬퍼
- LegalChunk: Pydantic 검증 모델
- 스키마 상수: LEGAL_CHUNKS_SCHEMA, COLUMN_GROUPS
"""

import pytest
import numpy as np
from app.tools.vectorstore.schema_v2 import (
    LEGAL_CHUNKS_SCHEMA,
    VECTOR_DIM,
    TABLE_NAME,
    COMMON_COLUMNS,
    LAW_COLUMNS,
    PRECEDENT_COLUMNS,
    ALL_COLUMNS,
    LegalChunk,
    create_law_chunk,
    create_precedent_chunk,
)
```

#### 테스트 상세

| # | 함수명 | FR | 검증 내용 | 어설션 |
|---|--------|-----|----------|--------|
| 1 | `test_create_law_chunk_required_fields` | FR-01 | create_law_chunk 결과에 공통 필드 존재 | `result["id"] == "010719_0"`, `result["data_type"] == "법령"`, `result["source_id"] == "010719"` |
| 2 | `test_create_law_chunk_law_specific_fields` | FR-01 | 법령 전용 필드 매핑 | `result["law_type"] == "법률"`, `result["article_no"] == "제750조"` |
| 3 | `test_create_law_chunk_precedent_fields_null` | FR-04 | 법령 청크에서 판례 필드 모두 None | `all(result[col] is None for col in PRECEDENT_COLUMNS)` |
| 4 | `test_create_precedent_chunk_required_fields` | FR-02 | create_precedent_chunk 결과에 공통 필드 존재 | `result["data_type"] == "판례"` |
| 5 | `test_create_precedent_chunk_precedent_specific_fields` | FR-02 | 판례 전용 필드 매핑 | `result["case_number"] == "2023다12345"`, `result["case_type"] == "민사"` |
| 6 | `test_create_precedent_chunk_law_fields_null` | FR-04 | 판례 청크에서 법령 필드 모두 None | `all(result[col] is None for col in LAW_COLUMNS)` |
| 7 | `test_legal_chunk_validation_valid_law` | FR-03 | 유효한 법령 LegalChunk 통과 | `chunk.validate_by_type() is True` |
| 8 | `test_legal_chunk_validation_valid_precedent` | FR-03 | 유효한 판례 LegalChunk 통과 | `chunk.validate_by_type() is True` |
| 9 | `test_legal_chunk_validation_law_with_precedent_fields` | FR-04 | 법령에 판례 필드 설정 시 에러 | `pytest.raises(ValueError, match="판례 필드")` |
| 10 | `test_legal_chunk_validation_precedent_with_law_fields` | FR-04 | 판례에 법령 필드 설정 시 에러 | `pytest.raises(ValueError, match="법령 필드")` |
| 11 | `test_id_format_law` | FR-18 | 법령 ID 형식 | `result["id"] == f"{source_id}_{chunk_index}"` |
| 12 | `test_id_format_precedent` | FR-18 | 판례 ID 형식 | `result["id"] == f"{source_id}_{chunk_index}"` |
| 13 | `test_schema_column_count` | - | 스키마 총 20컬럼 | `len(LEGAL_CHUNKS_SCHEMA) == 20` |
| 14 | `test_vector_dimension` | - | VECTOR_DIM == 1024 | `VECTOR_DIM == 1024` |
| 15 | `test_column_groups` | - | 컬럼 그룹 개수 확인 | `len(COMMON_COLUMNS) == 10`, `len(LAW_COLUMNS) == 4`, `len(PRECEDENT_COLUMNS) == 6` |

### 6.2 `test_lancedb_store.py` (21 tests)

**대상**: `backend/app/tools/vectorstore/lancedb.py`, `__init__.py`
**의존성**: `lancedb`, `pyarrow`, `numpy`, `pandas`
**픽스처**: `lancedb_store`, `populated_store`, `make_random_vector`

```python
"""
LanceDBStore 단위 테스트

대상:
- LanceDBStore.__init__(): 초기화
- add_law_documents(), add_precedent_documents(): 문서 추가
- search(), search_by_type(): 벡터 검색
- get_by_source_id(), delete_by_source_id(): CRUD
- count(), count_by_type(): 카운트
- reset(): 초기화
- _escape_sql(), _build_filter_conditions(): 내부 헬퍼
- get_vector_store(): 팩토리 함수

모든 테스트는 tmp_path 기반 실제 LanceDB를 사용.
"""

import pytest
from app.tools.vectorstore.base import SearchResult
from app.tools.vectorstore.lancedb import LanceDBStore
```

#### 테스트 상세

| # | 함수명 | FR | 검증 내용 |
|---|--------|-----|----------|
| 1 | `test_store_initialization_default` | FR-05 | `lancedb_store.table_name == "test_legal_chunks"`, `lancedb_store.db is not None` |
| 2 | `test_store_initialization_custom_collection` | FR-05 | `LanceDBStore(collection_name="custom")` → `.table_name == "custom"` |
| 3 | `test_add_law_documents` | FR-06 | 법령 6청크 추가 후 `store.count() == 6` |
| 4 | `test_add_precedent_documents` | FR-07 | 판례 6청크 추가 후 `store.count() == 6` |
| 5 | `test_add_law_documents_metadata` | FR-06 | `get_by_ids(["010719_0"])` 메타데이터에 `data_type == "법령"` |
| 6 | `test_add_precedent_documents_metadata` | FR-07 | `get_by_ids(["76396_0"])` 메타데이터에 `case_number == "2023다12345"` |
| 7 | `test_search_returns_search_result` | FR-08,19 | `isinstance(result, SearchResult)` |
| 8 | `test_search_vector_similarity` | FR-08 | 삽입된 벡터와 동일 벡터로 검색 시 해당 문서가 최상위 |
| 9 | `test_search_n_results` | FR-08 | `len(result.ids[0]) <= n_results` |
| 10 | `test_search_by_type_law` | FR-09 | 결과 모든 메타데이터의 `data_type == "법령"` |
| 11 | `test_search_by_type_precedent` | FR-09 | 결과 모든 메타데이터의 `data_type == "판례"` |
| 12 | `test_count_total` | FR-10 | `populated_store.count() == 12` (법령 6 + 판례 6) |
| 13 | `test_count_by_type` | FR-10 | `count_by_type("법령") == 6`, `count_by_type("판례") == 6` |
| 14 | `test_get_by_source_id` | FR-11 | `get_by_source_id("010719")` → `len(ids) == 2`, chunk_index 정렬 확인 |
| 15 | `test_get_by_source_id_not_found` | FR-11 | `get_by_source_id("nonexistent")` → `{"ids": [], ...}` |
| 16 | `test_delete_by_source_id` | FR-12 | 삭제 전 2청크 → 삭제 후 0청크, 총 count 감소 |
| 17 | `test_escape_sql` | FR-13 | `_escape_sql("it's")` → `"it''s"` |
| 18 | `test_build_filter_conditions_string` | FR-14 | `{"data_type": "법령"}` → `["data_type = '법령'"]` |
| 19 | `test_build_filter_conditions_list` | FR-14 | `{"data_type": ["법령", "판례"]}` → `IN` 절 생성 |
| 20 | `test_reset` | FR-15 | `reset()` 후 `count() == 0` |
| 21 | `test_get_vector_store_returns_lancedb` | FR-16 | `monkeypatch` VECTOR_DB="lancedb" → `isinstance(store, LanceDBStore)` |

#### `test_search_result_dict_access` (FR-19)

```python
def test_search_result_dict_access():
    """SearchResult의 dict-like 접근 (get, __getitem__)"""
    result = SearchResult(
        ids=[["id1", "id2"]],
        distances=[[0.1, 0.2]],
        documents=[["doc1", "doc2"]],
        metadatas=[[{"key": "val1"}, {"key": "val2"}]],
    )
    # dict-like 접근
    assert result["ids"] == [["id1", "id2"]]
    assert result.get("distances") == [[0.1, 0.2]]
    assert result.get("nonexistent", "default") == "default"
```

### 6.3 `test_mecab_tokenizer.py` (11 tests)

**대상**: `backend/app/tools/vectorstore/mecab_tokenizer.py`
**의존성**: `mecab-python3`, `mecab-ko-dic`
**마커**: `@pytest.mark.requires_mecab` (MeCab 의존 테스트)

```python
"""
MeCab 한국어 형태소 분석기 단위 테스트

대상:
- MeCabTokenizer: 초기화, morphs(), tokenize(), tokenize_query()
- is_mecab_available(): 설치 여부 확인
- Fallback: MeCab 미설치 시 공백 분리
"""

import pytest
from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer, is_mecab_available
```

#### 테스트 상세

| # | 함수명 | FR | 마커 | 검증 내용 |
|---|--------|-----|------|----------|
| 1 | `test_mecab_initialization` | FR-20 | requires_mecab | `tokenizer.is_available is True` |
| 2 | `test_mecab_morphs_basic` | FR-20 | requires_mecab | `"손해"` in `morphs("손해배상")` |
| 3 | `test_mecab_legal_terms_tokenization` | FR-21 | requires_mecab | `"손해"`, `"배상"`, `"청구"` all in `morphs("손해배상청구")` |
| 4 | `test_mecab_article_reference` | FR-21 | requires_mecab | `morphs("민법 제750조")` 결과에 `"민법"`, `"제"`, `"750"`, `"조"` 포함 여부 검증 |
| 5 | `test_mecab_case_number` | FR-21 | requires_mecab | `morphs("2023다12345")` 가 에러 없이 처리되고 비어있지 않음 |
| 6 | `test_pretokenize_content_for_fts` | FR-22 | requires_mecab | `tokenize("불법행위로 인한 손해배상")` 결과가 공백 구분 문자열 |
| 7 | `test_pretokenize_preserves_searchability` | FR-22 | requires_mecab | `tokenize()` 결과에 `"손해"`, `"배상"` 문자열 포함 |
| 8 | `test_pretokenize_query` | FR-22 | requires_mecab | `tokenize_query("손해배상")` 결과가 `tokenize("손해배상")`과 동일 |
| 9 | `test_mecab_not_installed_fallback` | FR-27 | (마커 없음) | `MeCabTokenizer` mock으로 `_tagger=None` → 공백 분리 동작 확인 |
| 10 | `test_mecab_empty_string` | FR-20 | (마커 없음) | `tokenizer.morphs("")` → `[]`, `tokenizer.tokenize("")` → `""` |
| 11 | `test_mecab_mixed_korean_english` | FR-20 | requires_mecab | `tokenize("OWASP 보안 취약점")` 결과에 `"OWASP"` 포함 |

#### Fallback 테스트 구현 패턴

```python
def test_mecab_not_installed_fallback():
    """MeCab 미설치 시 공백 분리 fallback"""
    tokenizer = MeCabTokenizer()
    # _tagger를 None으로 강제 설정하여 fallback 테스트
    tokenizer._tagger = None

    result = tokenizer.morphs("손해 배상 청구")
    assert result == ["손해", "배상", "청구"]

    result_str = tokenizer.tokenize("손해 배상 청구")
    assert result_str == "손해 배상 청구"
```

### 6.4 `test_lancedb_integration.py` (15 tests)

**대상**: E2E 벡터 검색 + FTS + 하이브리드 검색 흐름
**의존성**: `lancedb`, `numpy`, `mecab-python3`
**마커**: `requires_lancedb`, `requires_mecab` (FTS 테스트), `requires_fts`

#### Section A: 벡터 검색 E2E (6 tests)

| # | 함수명 | FR | 검증 흐름 |
|---|--------|-----|----------|
| 1 | `test_e2e_add_and_search_law` | FR-17 | 법령 추가 → 동일 벡터 검색 → `ids[0]` 확인 |
| 2 | `test_e2e_add_and_search_precedent` | FR-17 | 판례 추가 → 검색 → 판례 결과 확인 |
| 3 | `test_e2e_mixed_search_filter_by_type` | FR-17 | 법령+판례 추가 → `search_by_type("법령")` → 법령만 반환 |
| 4 | `test_e2e_source_id_extraction_flow` | FR-17 | 검색 → `metadatas[0][0]["source_id"]` 추출 → 문자열 확인 |
| 5 | `test_e2e_multi_chunk_document` | FR-17 | 3청크 문서 추가 → `get_by_source_id()` → 3청크 반환, chunk_index 순서 |
| 6 | `test_e2e_add_search_delete_lifecycle` | FR-17 | 추가 → 검색 확인 → `delete_by_source_id` → 재검색 → 결과 없음 |

#### Section B: FTS E2E (6 tests)

```python
@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestFTSIntegration:
    """FTS 인덱스 + MeCab 사전 토크나이징 통합 테스트"""
```

| # | 함수명 | FR | 검증 흐름 |
|---|--------|-----|----------|
| 7 | `test_fts_index_creation` | FR-23 | `content_tokenized` 컬럼으로 FTS 인덱스 생성 → 에러 없음 |
| 8 | `test_fts_keyword_search` | FR-24 | MeCab 토크나이징 쿼리 → FTS 검색 → 결과 존재 |
| 9 | `test_fts_search_accuracy` | FR-25 | "손해배상" 검색 → 손해배상 관련 문서가 상위 |
| 10 | `test_fts_match_query` | FR-28 | 단일 키워드 "손해" → 매칭 결과 확인 |
| 11 | `test_fts_phrase_query` | FR-28 | 구문 "손해 배상" → 해당 토큰 연속 포함 문서 |
| 12 | `test_fts_boolean_query` | FR-28 | "손해" AND "배상" → 두 토큰 모두 포함 문서 |

#### Section C: 하이브리드 검색 E2E (3 tests)

```python
@pytest.mark.requires_mecab
@pytest.mark.requires_fts
class TestHybridSearch:
    """벡터 + FTS 하이브리드 검색 (RRF) 통합 테스트"""
```

| # | 함수명 | FR | 검증 흐름 |
|---|--------|-----|----------|
| 13 | `test_hybrid_search_rrf` | FR-26 | 벡터 Top-K + FTS Top-K → RRF 결합 → 최종 순위 |
| 14 | `test_hybrid_search_vector_only_fallback` | FR-26 | FTS 결과 없는 쿼리 → 벡터 결과만 반환 |
| 15 | `test_hybrid_search_fts_boost` | FR-26 | 정확한 키워드 쿼리 → FTS 결과가 순위에 기여 |

#### FTS 테스트 데이터 구성

```python
@pytest.fixture
def fts_store(tmp_path, monkeypatch, mecab_tokenizer):
    """FTS 인덱스가 구성된 LanceDBStore"""
    from app.core.config import settings

    lancedb_dir = tmp_path / "lancedb_fts"
    lancedb_dir.mkdir()
    monkeypatch.setattr(settings, "LANCEDB_URI", str(lancedb_dir))
    monkeypatch.setattr(settings, "LANCEDB_TABLE_NAME", "test_fts_chunks")

    from app.tools.vectorstore.lancedb import LanceDBStore
    store = LanceDBStore()

    # content_tokenized 포함 데이터 추가
    # → MeCab으로 content를 사전 토크나이징하여 content_tokenized 컬럼에 저장
    contents = [
        "불법행위로 인한 손해배상청구권의 소멸시효",
        "민법 제750조에 의한 손해배상 책임",
        "도로교통법 위반으로 인한 교통사고",
    ]
    tokenized_contents = [mecab_tokenizer.tokenize(c) for c in contents]

    # add_documents로 content_tokenized 포함하여 추가
    # (스키마 확장 필요 - 구현 단계에서 schema_v2에 content_tokenized 추가)
    # ...

    # FTS 인덱스 생성
    table = store._ensure_table()
    table.create_fts_index("content_tokenized")

    yield store
    store.reset()
```

#### RRF (Reciprocal Rank Fusion) 구현 패턴

```python
def reciprocal_rank_fusion(
    vector_ids: list[str],
    fts_ids: list[str],
    k: int = 60,
) -> list[str]:
    """
    벡터 검색 + FTS 검색 결과를 RRF로 결합

    Args:
        vector_ids: 벡터 검색 결과 ID 리스트 (순위순)
        fts_ids: FTS 검색 결과 ID 리스트 (순위순)
        k: RRF 파라미터 (기본값: 60)

    Returns:
        RRF 점수 내림차순 정렬된 ID 리스트
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(fts_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=lambda x: scores[x], reverse=True)
```

---

## 7. Error Handling

### 7.1 테스트 환경 에러 처리

| 상황 | 처리 방법 |
|------|----------|
| LanceDB 미설치 | `@pytest.mark.requires_lancedb` + conftest에서 import 체크 |
| MeCab 미설치 | `@pytest.mark.requires_mecab` + `is_mecab_available()` 체크 |
| tmp_path 권한 문제 | pytest 자체 처리 (OS 기본 tmp) |
| FTS 인덱스 생성 실패 | `try/except` → `pytest.skip("FTS index creation failed")` |
| settings import 에러 | `monkeypatch`로 환경변수 사전 설정 |

### 7.2 MeCab Fallback 동작

```
MeCab 설치 여부 확인
    │
    ├─ 설치됨 → MeCab.Tagger() 사용 → 형태소 분석
    │
    └─ 미설치 → 경고 로그 출력 → 공백 분리 (str.split())
```

---

## 8. Test Plan Summary

### 8.1 전체 테스트 수

| 파일 | 테스트 수 | 마커 의존성 |
|------|----------|------------|
| `test_vectorstore_schema.py` | 15 | 없음 |
| `test_lancedb_store.py` | 21 | `requires_lancedb` (lancedb 패키지) |
| `test_mecab_tokenizer.py` | 11 | `requires_mecab` (7개), 없음 (4개) |
| `test_lancedb_integration.py` | 15 | `requires_lancedb` + `requires_mecab`/`requires_fts` (9개) |
| **합계** | **62** | |

### 8.2 마커별 실행 가이드

```bash
# 전체 실행
uv run pytest tests/unit/test_vectorstore_schema.py tests/unit/test_lancedb_store.py tests/unit/test_mecab_tokenizer.py tests/integration/test_lancedb_integration.py -v

# MeCab 없이 실행 (46 tests)
uv run pytest -m "not requires_mecab" -v

# FTS 없이 실행 (53 tests)
uv run pytest -m "not requires_fts" -v

# 스키마 테스트만 (외부 의존성 없음, 15 tests)
uv run pytest tests/unit/test_vectorstore_schema.py -v

# 저장소 테스트만 (LanceDB만 필요, 21 tests)
uv run pytest tests/unit/test_lancedb_store.py -v
```

### 8.3 CI 환경 고려사항

| 환경 | LanceDB | MeCab | 실행 가능 테스트 |
|------|---------|-------|----------------|
| 로컬 (macOS, brew install mecab) | O | O | 62/62 |
| 로컬 (MeCab 미설치) | O | X | 46/62 |
| CI (Ubuntu, apt install mecab) | O | O | 62/62 |
| CI (최소 환경) | O | X | 46/62 |

---

## 9. Implementation Guide

### 9.1 구현 순서

```
Step 1: mecab_tokenizer.py 생성
    └─ MeCabTokenizer 클래스, is_mecab_available()

Step 2: conftest.py 업데이트
    └─ 새 마커 등록 (requires_mecab, requires_fts)
    └─ 새 픽스처 추가 (random_vector, lancedb_store, populated_store, mecab_tokenizer 등)

Step 3: test_vectorstore_schema.py 작성
    └─ 순수 함수 테스트 (DB 불필요)

Step 4: test_lancedb_store.py 작성
    └─ LanceDBStore CRUD + 팩토리 테스트

Step 5: test_mecab_tokenizer.py 작성
    └─ MeCab 형태소 분석 + fallback 테스트

Step 6: schema_v2.py에 content_tokenized 컬럼 추가
    └─ LEGAL_CHUNKS_SCHEMA 확장 (21번째 컬럼)

Step 7: test_lancedb_integration.py 작성
    └─ E2E 벡터 검색 + FTS + 하이브리드

Step 8: 린트 및 타입 체크
    └─ uv run ruff check backend/
    └─ uv run mypy backend/app/tools/vectorstore/
```

### 9.2 의존성 추가

```bash
# 개발 의존성으로 추가
cd backend
uv add mecab-python3

# OS 레벨 MeCab 설치 (macOS)
brew install mecab mecab-ko-dic

# OS 레벨 MeCab 설치 (Ubuntu)
sudo apt-get install -y mecab libmecab-dev mecab-ko-dic
```

### 9.3 schema_v2.py 변경 사항 (Step 6)

```python
# LEGAL_CHUNKS_SCHEMA에 추가할 필드
pa.field("content_tokenized", pa.utf8()),  # MeCab 사전 토크나이징 결과

# COMMON_COLUMNS에 추가
COMMON_COLUMNS = [
    "id", "source_id", "data_type", "title", "content",
    "vector", "date", "source_name", "chunk_index", "total_chunks",
    "content_tokenized",  # 추가
]
```

> **주의**: `content_tokenized`는 기존 데이터에는 NULL이 될 수 있으므로 nullable 유지.
> 기존 임베딩 데이터 재생성 불필요 (새 문서 추가 시에만 MeCab 토크나이징 적용).

---

## 10. Coding Conventions

### 10.1 테스트 코드 규칙

| 규칙 | 적용 |
|------|------|
| 함수명 | `test_<행위>_<조건>_<기대결과>` (가능한 범위에서) |
| Docstring | 각 테스트 함수에 한국어 1줄 설명 |
| Assert | 하나의 테스트에서 하나의 논리적 검증 (관련 assert는 묶어도 OK) |
| 픽스처 | conftest.py에 공유, 파일 로컬 픽스처는 최소화 |
| 마커 | 외부 의존성 있으면 반드시 마커 부착 |
| Import | 표준 → 서드파티 → 로컬 순서 (isort 규칙) |

### 10.2 타입 힌트

```python
# 테스트 파일에서도 타입 힌트 권장
def test_example(lancedb_store: "LanceDBStore") -> None:
    """테스트 설명"""
    result: SearchResult = lancedb_store.search(...)
    assert isinstance(result, SearchResult)
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-05 | Plan 기반 초기 설계 작성 | Claude |
