"""
인제스트 파이프라인 통합 테스트 (첫 행 기반)

19개 데이터 타입의 인제스트 설정을 실제 JSON 첫 행으로 검증한다.

테스트 구성:
- TestIngestConfig: 19타입 × 4함수 = 76개 (외부 의존성 없음)
- TestFtsIndexingPipeline: 19타입 × 3함수 = 57개 (MeCab 필요)
- test_all_configs_registered: 1개
- 총 134개 (MeCab 미설치 시 57개 skip)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# 프로젝트 경로 설정
_BACKEND_ROOT = Path(__file__).parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import scripts.ingest.types  # noqa: E402, F401
from app.tools.vectorstore.mecab_tokenizer import (  # noqa: E402
    MeCabTokenizer,
    is_mecab_available,
)
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_config,
    list_configs,
)

# 전체 19개 타입 이름
ALL_CONFIG_NAMES = sorted(list_configs())

# 벡터 차원
VECTOR_DIM = 1024

# FTS 메타데이터 필수 키
FTS_REQUIRED_KEYS = {"source_id", "data_type", "title", "date", "source_name", "case_number"}

# 벡터 메타데이터 필수 키
VECTOR_REQUIRED_KEYS = {
    "id", "source_id", "data_type", "title", "content",
    "vector", "source_name", "chunk_index", "total_chunks", "date",
}

# search_text 토큰 형식: 공백 구분 문자열 (예: "손해 배상 청구")
# 각 토큰이 비어있지 않은 문자열인지 확인
_SEARCH_TEXT_TOKEN_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# 첫 행 로드 헬퍼
# ---------------------------------------------------------------------------


def _load_first_row_single_file(file_path: Path) -> dict[str, Any] | None:
    """단일 JSON 파일에서 첫 항목을 ijson 스트리밍으로 추출"""
    try:
        import ijson
    except ImportError:
        pytest.skip("ijson 패키지가 설치되지 않았습니다")
        return None  # unreachable

    if not file_path.exists():
        return None

    with open(file_path, "rb") as f:
        # 최상위 배열 또는 items 키
        for item in ijson.items(f, "item"):
            return dict(item)

    return None


def _load_first_row_directory(dir_path: Path) -> dict[str, Any] | None:
    """디렉토리 타입에서 첫 JSON 파일의 첫 항목 추출 + __source_group__ 주입"""
    import re

    if not dir_path.exists() or not dir_path.is_dir():
        return None

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        return None

    first_file = json_files[0]

    # 그룹명 추출 (db_writer._extract_group_from_filename 로직 재현)
    m = re.search(r"(?:dec_comm|intp_min|sadm_case)_(.+?)_v\d+\.json", first_file.name)
    group_name = m.group(1) if m else first_file.stem

    try:
        import ijson

        with open(first_file, "rb") as f:
            for item in ijson.items(f, "item"):
                row = dict(item)
                row["__source_group__"] = group_name
                return row
    except ImportError:
        # fallback: json.load
        with open(first_file, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("items", [])
        if items:
            row = dict(items[0])
            row["__source_group__"] = group_name
            return row

    return None


def _load_first_row(config: IngestConfig) -> dict[str, Any] | None:
    """설정에 따라 첫 행 로드 (단일 파일 / 디렉토리 자동 분기)"""
    source = config.source_path

    if source.is_dir():
        return _load_first_row_directory(source)

    return _load_first_row_single_file(source)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def first_rows() -> dict[str, dict[str, Any]]:
    """19개 타입의 첫 행 데이터를 세션 스코프로 캐싱"""
    rows: dict[str, dict[str, Any]] = {}
    for name in ALL_CONFIG_NAMES:
        cfg = get_config(name)
        row = _load_first_row(cfg)
        if row is not None:
            rows[name] = row
    return rows


@pytest.fixture
def dummy_vector() -> list[float]:
    """테스트용 더미 벡터 (1024차원, 모두 0.1)"""
    return [0.1] * VECTOR_DIM


@pytest.fixture(scope="session")
def mecab_tokenizer() -> MeCabTokenizer:
    """MeCab 토크나이저 (userdic 필수). MeCab 또는 userdic 미설치 시 skip."""
    if not is_mecab_available():
        pytest.skip("MeCab이 설치되지 않았습니다")

    from app.core.config import settings

    dic_path = Path(settings.MECAB_USERDIC_PATH)
    if not dic_path.exists():
        pytest.skip(f"userdic 미빌드: {dic_path}")

    decomp_path = dic_path.parent / "decomposition_map.json"
    decomposition_map: dict[str, list[str]] = {}
    if decomp_path.exists():
        with open(decomp_path, encoding="utf-8") as f:
            decomposition_map = json.load(f)

    return MeCabTokenizer(
        userdic_path=str(dic_path),
        decomposition_map=decomposition_map,
    )


# ---------------------------------------------------------------------------
# 테스트: 등록 확인
# ---------------------------------------------------------------------------


def test_all_configs_registered() -> None:
    """19개 타입이 모두 등록되어 있는지 확인"""
    configs = list_configs()
    assert len(configs) == 19, (
        f"등록된 타입 수: {len(configs)}, 기대: 19. "
        f"등록된 타입: {configs}"
    )


# ---------------------------------------------------------------------------
# 파라미터화 테스트 (19개 타입 × 4개 함수)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
class TestIngestConfig:
    """인제스트 설정 파이프라인 함수 검증"""

    def _get_row(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """첫 행 데이터 가져오기. 없으면 skip."""
        if config_name not in first_rows:
            pytest.skip(
                f"소스 JSON 파일이 없습니다: {get_config(config_name).source_path}"
            )
        return first_rows[config_name]

    # --- 1. ORM Factory ---

    def test_orm_factory(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
    ) -> None:
        """orm_factory_fn: ORM 인스턴스 생성 + ID 속성 검증"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        result = config.orm_factory_fn(row)

        # ORM 클래스 인스턴스인지 확인
        assert isinstance(result, config.orm_class), (
            f"반환 타입: {type(result).__name__}, "
            f"기대: {config.orm_class.__name__}"
        )

        # orm_id_attr 값이 존재하는지 확인
        id_value = getattr(result, config.orm_id_attr, None)
        assert id_value is not None, (
            f"ORM 인스턴스의 {config.orm_id_attr} 속성이 None입니다"
        )
        assert str(id_value).strip(), (
            f"ORM 인스턴스의 {config.orm_id_attr} 속성이 빈 문자열입니다"
        )

    # --- 2. Fulltext ---

    def test_fulltext_fn(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
    ) -> None:
        """fulltext_fn: 비어있지 않은 문자열 반환 검증"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        result = config.fulltext_fn(row)

        assert isinstance(result, str), (
            f"반환 타입: {type(result).__name__}, 기대: str"
        )
        assert len(result) > 0, "fulltext_fn이 빈 문자열을 반환했습니다"

    # --- 3. FTS Metadata ---

    def test_fts_metadata_fn(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
    ) -> None:
        """fts_metadata_fn: 필수 키 6개 존재 + source_id/data_type 검증"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        result = config.fts_metadata_fn(row)

        assert isinstance(result, dict), (
            f"반환 타입: {type(result).__name__}, 기대: dict"
        )

        # 필수 키 6개 존재 확인
        missing = FTS_REQUIRED_KEYS - result.keys()
        assert not missing, (
            f"FTS 메타데이터에 누락된 키: {missing}. "
            f"반환된 키: {set(result.keys())}"
        )

        # source_id가 non-empty string
        source_id = result["source_id"]
        assert isinstance(source_id, str) and source_id.strip(), (
            f"source_id가 비어있습니다: {source_id!r}"
        )

        # data_type이 config.data_type_label과 일치
        assert result["data_type"] == config.data_type_label, (
            f"data_type 불일치: {result['data_type']!r} != "
            f"{config.data_type_label!r}"
        )

    # --- 4. Vector Metadata ---

    def test_vector_metadata_fn(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
        dummy_vector: list[float],
    ) -> None:
        """vector_metadata_fn: 10개 키 + 벡터 차원 + data_type 검증"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        result = config.vector_metadata_fn(row, dummy_vector)

        assert isinstance(result, dict), (
            f"반환 타입: {type(result).__name__}, 기대: dict"
        )

        # 필수 키 10개 존재 확인
        missing = VECTOR_REQUIRED_KEYS - result.keys()
        assert not missing, (
            f"벡터 메타데이터에 누락된 키: {missing}. "
            f"반환된 키: {set(result.keys())}"
        )

        # vector 차원 확인
        vec = result["vector"]
        assert isinstance(vec, list), f"vector 타입: {type(vec).__name__}"
        assert len(vec) == VECTOR_DIM, (
            f"vector 차원: {len(vec)}, 기대: {VECTOR_DIM}"
        )

        # data_type 일치 확인
        assert result["data_type"] == config.data_type_label, (
            f"data_type 불일치: {result['data_type']!r} != "
            f"{config.data_type_label!r}"
        )

        # source_id가 non-empty string
        source_id = result["source_id"]
        assert isinstance(source_id, str) and source_id.strip(), (
            f"source_id가 비어있습니다: {source_id!r}"
        )

        # id 형식 확인 (source_id_chunkIndex)
        chunk_id = result["id"]
        assert isinstance(chunk_id, str) and chunk_id.strip(), (
            f"id가 비어있습니다: {chunk_id!r}"
        )


# ---------------------------------------------------------------------------
# FTS 역인덱싱 파이프라인 테스트 (19개 타입 × 3개 함수, MeCab 필요)
#
# fulltext_fn → MeCab.morphs → search_text 전체 흐름 검증
# ---------------------------------------------------------------------------


@pytest.mark.requires_mecab
@pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
class TestFtsIndexingPipeline:
    """FTS 역인덱싱 파이프라인: fulltext → MeCab 토크나이징 → search_text 생성"""

    def _get_row(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """첫 행 데이터 가져오기. 없으면 skip."""
        if config_name not in first_rows:
            pytest.skip(
                f"소스 JSON 파일이 없습니다: {get_config(config_name).source_path}"
            )
        return first_rows[config_name]

    # --- 1. MeCab 토크나이징 ---

    def test_mecab_tokenize(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
        mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """fulltext → MeCab morphs: 비어있지 않은 토큰 리스트 반환"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        fulltext = config.fulltext_fn(row)
        tokens = mecab_tokenizer.morphs(fulltext)

        assert isinstance(tokens, list), (
            f"반환 타입: {type(tokens).__name__}, 기대: list"
        )
        assert len(tokens) > 0, (
            f"MeCab이 빈 토큰 리스트를 반환했습니다. "
            f"fulltext 길이: {len(fulltext)}"
        )
        # 모든 토큰이 non-empty 문자열인지 확인
        for i, token in enumerate(tokens):
            assert isinstance(token, str) and token.strip(), (
                f"토큰[{i}]이 빈 문자열입니다: {token!r}"
            )

    # --- 2. search_text 생성 ---

    def test_search_text_pipeline(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
        mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """fulltext → morphs → search_text: 유효한 공백 구분 토큰 문자열 생성"""
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        fulltext = config.fulltext_fn(row)
        tokens = mecab_tokenizer.morphs(fulltext)
        search_text = " ".join(tokens)

        assert isinstance(search_text, str), (
            f"반환 타입: {type(search_text).__name__}, 기대: str"
        )
        assert len(search_text) > 0, (
            f"search_text가 빈 문자열입니다. 토큰 수: {len(tokens)}"
        )

        # search_text 형식 검증: 공백 구분 토큰
        search_tokens = _SEARCH_TEXT_TOKEN_RE.findall(search_text)
        assert len(search_tokens) > 0, (
            f"search_text에서 유효한 토큰을 찾을 수 없습니다: {search_text[:200]!r}"
        )

        # 토큰 수가 일치하는지 확인
        assert len(search_tokens) == len(tokens), (
            f"search_text 토큰 수 불일치: {len(search_tokens)} != {len(tokens)}"
        )

    # --- 3. FTS 메타데이터 + search_text 통합 ---

    def test_fts_record_assembly(
        self,
        config_name: str,
        first_rows: dict[str, dict[str, Any]],
        mecab_tokenizer: MeCabTokenizer,
    ) -> None:
        """fts_metadata + search_text 조합: fts_index 레코드 완성 검증

        db_writer.py의 실제 파이프라인을 재현:
        1. fulltext_fn(item) → fulltext
        2. tokenizer.morphs(fulltext) → tokens
        3. " ".join(tokens) → search_text
        4. fts_metadata_fn(item) + search_text → fts_record
        """
        config = get_config(config_name)
        row = self._get_row(config_name, first_rows)

        # 1-3: search_text 생성
        fulltext = config.fulltext_fn(row)
        tokens = mecab_tokenizer.morphs(fulltext)
        search_text = " ".join(tokens)

        # 4: fts_metadata + search_text 조합 (db_writer.py 재현)
        fts_meta = config.fts_metadata_fn(row)
        fts_meta["search_text"] = search_text or None

        # fts_index 테이블의 필수 컬럼 7개 모두 존재
        fts_record_keys = {
            "source_id", "data_type", "title", "date",
            "source_name", "case_number", "search_text",
        }
        missing = fts_record_keys - fts_meta.keys()
        assert not missing, (
            f"FTS 레코드에 누락된 키: {missing}. "
            f"존재하는 키: {set(fts_meta.keys())}"
        )

        # search_text가 채워져 있는지 확인
        assert fts_meta["search_text"] is not None, (
            "search_text가 None입니다 (토크나이징 실패)"
        )
        assert len(fts_meta["search_text"]) > 0, (
            "search_text가 빈 문자열입니다"
        )

        # source_id가 fts_metadata와 일치하는지 확인
        assert fts_meta["source_id"] == str(
            row.get(config.id_field, "")
        ), (
            f"source_id 불일치: {fts_meta['source_id']!r} != "
            f"{str(row.get(config.id_field, ''))!r}"
        )
