"""
Pytest 공통 설정 및 픽스처

테스트 환경 설정 및 공유 픽스처 정의
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from typing import Callable, Generator

import numpy as np
import pytest
import pytest_asyncio
import tempfile


# ============================================================================
# Async 테스트용 이벤트 루프 설정
# ============================================================================
# 문제: SQLAlchemy async engine이 첫 사용 시 이벤트 루프에 바인딩됨
#       pytest-asyncio 기본값은 각 테스트마다 새 루프 생성 → 충돌 발생
# 해결: 전체 테스트 세션 동안 같은 이벤트 루프 공유

@pytest.fixture(scope="session")
def event_loop():
    """세션 스코프 이벤트 루프 (asyncpg 연결 풀 공유용)"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """세션 스코프 DB 엔진 (테스트 종료 시 정리)"""
    from app.core.database import engine
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """프로젝트 루트 경로"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def evaluation_module():
    """evaluation 모듈 임포트 픽스처"""
    import evaluation
    return evaluation


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """임시 디렉토리 픽스처"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_eval_query():
    """샘플 평가 쿼리 데이터"""
    return {
        "question": "손해배상 청구 요건은 무엇인가요?",
        "source_documents": [
            {"doc_id": "76396", "doc_type": "precedent"},
            {"doc_id": "010719", "doc_type": "law", "article": "제750조"},
        ],
        "category": "민사",
        "query_type": "개념검색",
        "difficulty": "medium",
        "key_points": ["불법행위", "손해", "인과관계"],
        "required_citations": ["민법 제750조"],
    }


@pytest.fixture
def sample_search_results():
    """샘플 검색 결과 데이터"""
    return [
        {
            "doc_id": "76396",
            "chunk_id": "76396_0",
            "score": 0.92,
            "title": "손해배상청구사건",
            "content": "불법행위로 인한 손해배상...",
        },
        {
            "doc_id": "010719",
            "chunk_id": "010719_5",
            "score": 0.85,
            "title": "민법",
            "content": "제750조(불법행위의 내용)...",
        },
        {
            "doc_id": "12345",
            "chunk_id": "12345_0",
            "score": 0.78,
            "title": "다른 판례",
            "content": "관련 내용...",
        },
    ]


@pytest.fixture
def sample_metrics():
    """샘플 메트릭 데이터"""
    return {
        "recall_at_5": 0.72,
        "recall_at_10": 0.84,
        "mrr": 0.68,
        "hit_rate": 0.92,
        "ndcg_at_10": 0.78,
        "latency_p50_ms": 150.0,
        "latency_p95_ms": 450.0,
    }


# 마커 등록
def pytest_configure(config):
    """pytest 마커 등록"""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_lancedb: marks tests that require LanceDB data"
    )
    config.addinivalue_line(
        "markers",
        "requires_postgres: marks tests that require PostgreSQL data"
    )
    config.addinivalue_line(
        "markers",
        "requires_mecab: marks tests that require MeCab installation"
    )
    config.addinivalue_line(
        "markers",
        "requires_fts: marks tests that require FTS index support"
    )


# ============================================================================
# LanceDB 테스트 픽스처
# ============================================================================

VECTOR_DIM = 1024


@pytest.fixture
def random_vector() -> list[float]:
    """고정 시드 1024차원 정규화 벡터"""
    rng = np.random.RandomState(42)
    vec = rng.randn(VECTOR_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def make_random_vector() -> Callable[[int], list[float]]:
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
    store.reset()


@pytest.fixture
def populated_store(lancedb_store, make_random_vector):
    """
    샘플 데이터가 미리 삽입된 LanceDBStore

    법령 3건(6청크) + 판례 3건(6청크) = 총 12레코드
    """
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


# ============================================================================
# MeCab 테스트 픽스처
# ============================================================================

@pytest.fixture
def mecab_tokenizer():
    """MeCab 토크나이저 (미설치 시 테스트 skip)"""
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
