"""
Pytest 공통 설정 및 픽스처

테스트 환경 설정 및 공유 픽스처 정의
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import tempfile
from typing import Generator


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
