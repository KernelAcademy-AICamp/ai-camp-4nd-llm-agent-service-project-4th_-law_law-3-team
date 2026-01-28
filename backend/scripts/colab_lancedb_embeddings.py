#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (Colab 독립 실행용)

Google Colab에서 실행 가능한 독립형 스크립트입니다.
법령/판례 데이터를 읽어 KURE-v1 임베딩을 생성하고 LanceDB에 저장합니다.

사용법 (Colab):
    1. 이 파일을 Colab에 업로드하거나 복사
    2. 셀에서 실행:
       !pip install lancedb sentence-transformers pyarrow
    3. 데이터 파일 업로드 (law_cleaned.json 등)
    4. 아래 설정 섹션 수정 후 실행
"""

# ============================================================================
# 패키지 설치 (Colab에서 첫 실행 시)
# ============================================================================
# !pip install lancedb sentence-transformers pyarrow -q

import gc
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass, field

import pyarrow as pa
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


# ============================================================================
# 디바이스 감지 및 최적화
# ============================================================================

@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device: str  # "cuda", "mps", "cpu"
    name: str
    vram_gb: float
    is_laptop: bool = False
    compute_capability: Optional[tuple] = None  # CUDA only

    def __str__(self) -> str:
        return f"{self.name} ({self.device}, {self.vram_gb:.1f}GB)"


@dataclass
class OptimalConfig:
    """환경별 최적 설정"""
    batch_size: int
    num_workers: int
    prefetch_factor: int = 2
    gc_interval: int = 10  # gc.collect() 호출 간격 (배치 수)
    use_fp16: bool = False  # 향후 확장용


def get_device() -> str:
    """사용 가능한 최적 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info() -> DeviceInfo:
    """디바이스 상세 정보 조회"""
    device = get_device()

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        name = props.name
        vram_gb = props.total_memory / (1024**3)
        compute_capability = (props.major, props.minor)

        # 노트북 GPU 감지 (이름에 Laptop, Mobile, Max-Q 등 포함)
        is_laptop = any(
            kw in name.lower()
            for kw in ["laptop", "mobile", "max-q", "notebook"]
        )

        return DeviceInfo(
            device=device,
            name=name,
            vram_gb=vram_gb,
            is_laptop=is_laptop,
            compute_capability=compute_capability,
        )

    elif device == "mps":
        # Apple Silicon - 통합 메모리 사용
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # psutil 없으면 기본값
            total_ram = 16.0

        # MPS는 시스템 RAM의 약 75%까지 사용 가능
        usable_memory = total_ram * 0.75

        return DeviceInfo(
            device=device,
            name="Apple Silicon (MPS)",
            vram_gb=usable_memory,
            is_laptop=True,  # 대부분 MacBook
        )

    else:
        # CPU 전용
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 8.0

        return DeviceInfo(
            device=device,
            name="CPU",
            vram_gb=total_ram,
        )


def get_optimal_config(device_info: DeviceInfo = None) -> OptimalConfig:
    """디바이스에 따른 최적 설정 반환"""
    if device_info is None:
        device_info = get_device_info()

    device = device_info.device
    vram = device_info.vram_gb

    if device == "cuda":
        if vram >= 14:  # 5060 Ti (16GB), 3080, 4080, etc.
            return OptimalConfig(
                batch_size=100,
                num_workers=4,
                gc_interval=20,
            )
        elif vram >= 8:  # 3070, 4060, etc.
            return OptimalConfig(
                batch_size=70,
                num_workers=2,
                gc_interval=15,
            )
        elif vram >= 6:  # 3060, 3060 Laptop
            return OptimalConfig(
                batch_size=50,
                num_workers=2,
                gc_interval=10,
            )
        else:  # 낮은 VRAM
            return OptimalConfig(
                batch_size=30,
                num_workers=1,
                gc_interval=5,
            )

    elif device == "mps":
        # Mac - MPS에서는 num_workers=0이 안정적
        if vram >= 12:  # M3 16GB (usable ~12GB)
            return OptimalConfig(
                batch_size=50,
                num_workers=0,  # MPS + multiprocessing 호환성 문제
                gc_interval=10,
            )
        else:  # M1/M2 8GB
            return OptimalConfig(
                batch_size=30,
                num_workers=0,
                gc_interval=5,
            )

    else:  # CPU
        return OptimalConfig(
            batch_size=20,
            num_workers=2,
            gc_interval=5,
        )


def print_device_info():
    """디바이스 정보 및 최적 설정 출력"""
    device_info = get_device_info()
    config = get_optimal_config(device_info)

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"  Device: {device_info.device.upper()}")
    print(f"  Name: {device_info.name}")
    print(f"  Memory: {device_info.vram_gb:.1f} GB")
    if device_info.is_laptop:
        print(f"  Type: Laptop/Mobile")
    if device_info.compute_capability:
        print(f"  Compute Capability: {device_info.compute_capability[0]}.{device_info.compute_capability[1]}")

    print("\nRecommended Settings:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_workers: {config.num_workers}")
    print(f"  gc_interval: {config.gc_interval} batches")
    print("=" * 60)

    return device_info, config

# ============================================================================
# 설정 (필요에 따라 수정)
# ============================================================================

CONFIG = {
    # LanceDB 저장 경로
    "LANCEDB_URI": "/content/lancedb_data",
    "LANCEDB_TABLE_NAME": "legal_chunks",

    # 임베딩 모델
    "EMBEDDING_MODEL": "nlpai-lab/KURE-v1",
    "VECTOR_DIM": 1024,

    # 배치 크기
    "BATCH_SIZE": 100,

    # 판례 청킹 설정
    "PRECEDENT_CHUNK_SIZE": 1250,
    "PRECEDENT_CHUNK_OVERLAP": 125,
    "PRECEDENT_MIN_CHUNK_SIZE": 100,

    # 법령 청킹 설정 (토큰 기반)
    "LAW_MAX_TOKENS": 800,
    "LAW_MIN_TOKENS": 100,
}


# ============================================================================
# LanceDB 스키마 정의 (v2)
# ============================================================================

VECTOR_DIM = CONFIG["VECTOR_DIM"]

LEGAL_CHUNKS_SCHEMA = pa.schema([
    # 공통 필드
    pa.field("id", pa.utf8()),
    pa.field("source_id", pa.utf8()),
    pa.field("data_type", pa.utf8()),  # "법령" | "판례"
    pa.field("title", pa.utf8()),
    pa.field("content", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("date", pa.utf8()),
    pa.field("source_name", pa.utf8()),
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),

    # 법령 전용 (판례는 NULL)
    pa.field("promulgation_date", pa.utf8()),
    pa.field("promulgation_no", pa.utf8()),
    pa.field("law_type", pa.utf8()),
    pa.field("article_no", pa.utf8()),

    # 판례 전용 (법령은 NULL)
    # NOTE: ruling, claim, reasoning은 LanceDB에 저장하지 않음 (메모리 효율화)
    # 검색 후 원본 JSON 또는 PostgreSQL에서 조회
    pa.field("case_number", pa.utf8()),
    pa.field("case_type", pa.utf8()),
    pa.field("judgment_type", pa.utf8()),
    pa.field("judgment_status", pa.utf8()),
    pa.field("reference_provisions", pa.utf8()),
    pa.field("reference_cases", pa.utf8()),
])


def create_law_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: List[float],
    enforcement_date: str,
    department: str,
    total_chunks: int = 1,
    promulgation_date: Optional[str] = None,
    promulgation_no: Optional[str] = None,
    law_type: Optional[str] = None,
    article_no: Optional[str] = None,
) -> Dict[str, Any]:
    """법령 청크 레코드 생성"""
    return {
        "id": f"{source_id}_{chunk_index}",
        "source_id": source_id,
        "data_type": "법령",
        "title": title,
        "content": content,
        "vector": vector,
        "date": enforcement_date,
        "source_name": department,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "promulgation_date": promulgation_date,
        "promulgation_no": promulgation_no,
        "law_type": law_type,
        "article_no": article_no,
        "case_number": None,
        "case_type": None,
        "judgment_type": None,
        "judgment_status": None,
        "reference_provisions": None,
        "reference_cases": None,
    }


def create_precedent_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: List[float],
    decision_date: str,
    court_name: str,
    total_chunks: int = 1,
    case_number: Optional[str] = None,
    case_type: Optional[str] = None,
    judgment_type: Optional[str] = None,
    judgment_status: Optional[str] = None,
    reference_provisions: Optional[str] = None,
    reference_cases: Optional[str] = None,
) -> Dict[str, Any]:
    """
    판례 청크 레코드 생성

    NOTE: ruling, claim, reasoning은 LanceDB에 저장하지 않음 (메모리 효율화)
    """
    return {
        "id": f"{source_id}_{chunk_index}",
        "source_id": source_id,
        "data_type": "판례",
        "title": title,
        "content": content,
        "vector": vector,
        "date": decision_date,
        "source_name": court_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "promulgation_date": None,
        "promulgation_no": None,
        "law_type": None,
        "article_no": None,
        "case_number": case_number,
        "case_type": case_type,
        "judgment_type": judgment_type,
        "judgment_status": judgment_status,
        "reference_provisions": reference_provisions,
        "reference_cases": reference_cases,
    }


# ============================================================================
# LanceDB Store (간소화 버전)
# ============================================================================

class LanceDBStore:
    """LanceDB 벡터 저장소"""

    def __init__(self, db_path: str = None, table_name: str = None):
        import lancedb

        db_path = db_path or CONFIG["LANCEDB_URI"]
        self.table_name = table_name or CONFIG["LANCEDB_TABLE_NAME"]

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._table = None

        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)

    def _ensure_table(self):
        if self._table is None:
            self._table = self.db.create_table(
                self.table_name,
                schema=LEGAL_CHUNKS_SCHEMA,
            )
        return self._table

    def add_law_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        enforcement_dates: List[str],
        departments: List[str],
        total_chunks_list: List[int],
        promulgation_dates: List[str],
        promulgation_nos: List[str],
        law_types: List[str],
        article_nos: List[str],
    ) -> None:
        """법령 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_law_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                enforcement_date=enforcement_dates[i],
                department=departments[i],
                total_chunks=total_chunks_list[i],
                promulgation_date=promulgation_dates[i] if promulgation_dates else None,
                promulgation_no=promulgation_nos[i] if promulgation_nos else None,
                law_type=law_types[i] if law_types else None,
                article_no=article_nos[i] if article_nos else None,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def add_precedent_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        decision_dates: List[str],
        court_names: List[str],
        total_chunks_list: List[int],
        case_numbers: List[str] = None,
        case_types: List[str] = None,
        reference_provisions_list: List[str] = None,
        reference_cases_list: List[str] = None,
    ) -> None:
        """
        판례 문서 배치 추가

        NOTE: ruling, claim, reasoning은 LanceDB에 저장하지 않음 (메모리 효율화)
        """
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_precedent_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                decision_date=decision_dates[i],
                court_name=court_names[i],
                total_chunks=total_chunks_list[i],
                case_number=case_numbers[i] if case_numbers else None,
                case_type=case_types[i] if case_types else None,
                reference_provisions=reference_provisions_list[i] if reference_provisions_list else None,
                reference_cases=reference_cases_list[i] if reference_cases_list else None,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def count(self) -> int:
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        if self._table is None:
            return 0
        df = self._table.search().where(f"data_type = '{data_type}'").limit(1000000).to_pandas()
        return len(df)

    def get_existing_source_ids(self, data_type: str) -> set:
        """이미 저장된 source_id 조회"""
        if self._table is None:
            return set()
        try:
            df = self._table.search().where(f"data_type = '{data_type}'").limit(1000000).to_pandas()
            return set(df["source_id"].unique())
        except Exception:
            return set()

    def delete_by_type(self, data_type: str) -> int:
        """특정 유형 데이터 삭제"""
        if self._table is None:
            return 0
        count = self.count_by_type(data_type)
        if count > 0:
            self._table.delete(f"data_type = '{data_type}'")
        return count

    def reset(self):
        """테이블 초기화"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None


# ============================================================================
# 임베딩 모델 (KURE-v1)
# ============================================================================

_local_model = None
_current_device = None


def get_embedding_model(device: str = None):
    """
    KURE-v1 모델 로드

    Args:
        device: 사용할 디바이스 ("cuda", "mps", "cpu", None=자동감지)
    """
    global _local_model, _current_device

    if device is None:
        device = get_device()

    if _local_model is None or _current_device != device:
        from sentence_transformers import SentenceTransformer

        model_name = CONFIG["EMBEDDING_MODEL"]
        print(f"[INFO] Loading model: {model_name}")
        print(f"[INFO] Device: {device.upper()}")

        _local_model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        _current_device = device

        dim = _local_model.get_sentence_embedding_dimension()
        print(f"[INFO] Model loaded. Dimension: {dim}")

        if dim != VECTOR_DIM:
            raise ValueError(f"Model dimension ({dim}) != Schema dimension ({VECTOR_DIM})")

    return _local_model


def create_embeddings(texts: List[str], device: str = None) -> List[List[float]]:
    """
    임베딩 생성

    Args:
        texts: 임베딩할 텍스트 리스트
        device: 사용할 디바이스 (None=자동감지)
    """
    model = get_embedding_model(device)
    processed = [t.strip()[:4000] if t else "(내용 없음)" for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def clear_model_cache():
    """모델 캐시 정리 (메모리 해제)"""
    global _local_model, _current_device

    if _local_model is not None:
        del _local_model
        _local_model = None
        _current_device = None

        # GPU 메모리 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        print("[INFO] Model cache cleared")


# ============================================================================
# DataLoader용 Dataset 클래스
# ============================================================================

class PrecedentIterableDataset(IterableDataset):
    """판례 데이터를 스트리밍으로 읽어 청크를 생성하는 Dataset"""

    def __init__(
        self,
        source_path: str,
        chunk_config: "PrecedentChunkConfig",
        existing_source_ids: set = None,
    ):
        self.source_path = source_path
        self.chunk_config = chunk_config
        self.existing_source_ids = existing_source_ids or set()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """판례를 읽어 청크 단위로 yield"""
        use_streaming = IJSON_AVAILABLE
        f = None

        if use_streaming:
            # 파일 시작 문자로 배열/객체 판별
            with open(self.source_path, "rb") as check_f:
                first_char = check_f.read(1).decode("utf-8").strip()
                # BOM이나 공백 건너뛰기
                while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
                    first_char = check_f.read(1).decode("utf-8")

            f = open(self.source_path, "rb")
            if first_char == "[":
                # 배열 형태: [{...}, {...}]
                items = ijson.items(f, "item")
            else:
                # 객체 형태: {"items": [...]} 또는 {"precedents": [...]}
                items = ijson.items(f, "items.item")
        else:
            with open(self.source_path, "r", encoding="utf-8") as json_f:
                data = json.load(json_f)
            if isinstance(data, list):
                items = data
            else:
                items = data.get("items", data.get("precedents", []))

        for idx, item in enumerate(items):
            source_id = str(item.get("판례정보일련번호", item.get("id", item.get("source_id", idx))))

            if source_id in self.existing_source_ids:
                continue

            # 임베딩 텍스트: 판시사항 + 판결요지
            parts = []
            case_name = item.get("사건명", item.get("case_name", ""))
            if case_name:
                parts.append(f"[{case_name}]")

            summary = item.get("판시사항", item.get("summary", ""))
            if summary:
                parts.append(summary)

            judgment_summary = item.get("판결요지", item.get("judgment_summary", ""))
            if judgment_summary:
                parts.append(judgment_summary)

            text = "\n".join(parts)
            if not text or len(text) < self.chunk_config.min_chunk_size:
                continue

            chunks = chunk_precedent_text(text, self.chunk_config)
            if not chunks:
                continue

            total_chunks = len(chunks)

            # 메타데이터 (ruling, claim, reasoning 제외)
            decision_date = item.get("선고일자", item.get("decision_date", ""))
            court_name = item.get("법원명", item.get("court_name", ""))
            case_number = item.get("사건번호", item.get("case_number", ""))
            case_type = item.get("사건종류명", item.get("case_type", ""))
            ref_provisions = item.get("참조조문", item.get("reference_provisions", ""))
            ref_cases = item.get("참조판례", item.get("reference_cases", ""))

            for chunk_idx, chunk_content in chunks:
                yield {
                    "source_id": source_id,
                    "chunk_index": chunk_idx,
                    "title": case_name,
                    "content": f"[판례] {chunk_content}",
                    "decision_date": decision_date,
                    "court_name": court_name,
                    "total_chunks": total_chunks,
                    "case_number": case_number,
                    "case_type": case_type,
                    "reference_provisions": ref_provisions,
                    "reference_cases": ref_cases,
                }

        if use_streaming:
            f.close()


class LawIterableDataset(IterableDataset):
    """법령 데이터를 읽어 청크를 생성하는 Dataset"""

    def __init__(
        self,
        source_path: str,
        chunk_config: "LawChunkConfig",
        existing_source_ids: set = None,
    ):
        self.source_path = source_path
        self.chunk_config = chunk_config
        self.existing_source_ids = existing_source_ids or set()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """법령을 읽어 청크 단위로 yield"""
        use_streaming = IJSON_AVAILABLE
        f = None

        if use_streaming:
            # 파일 시작 문자로 배열/객체 판별
            with open(self.source_path, "rb") as check_f:
                first_char = check_f.read(1).decode("utf-8").strip()
                while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
                    first_char = check_f.read(1).decode("utf-8")

            f = open(self.source_path, "rb")
            if first_char == "[":
                items = ijson.items(f, "item")
            else:
                items = ijson.items(f, "items.item")
        else:
            with open(self.source_path, "r", encoding="utf-8") as json_f:
                data = json.load(json_f)
            if isinstance(data, list):
                items = data
            else:
                items = data.get("items", [])

        for item in items:
            source_id = item.get("law_id", "")

            if source_id in self.existing_source_ids:
                continue

            content = item.get("content", "")
            if not content:
                continue

            chunks = chunk_law_content(content, self.chunk_config)
            if not chunks:
                continue

            total_chunks = len(chunks)
            law_name = item.get("law_name", "")
            enforcement_date = item.get("enforcement_date", "")
            ministry = item.get("ministry", "")
            promulgation_date = item.get("promulgation_date", "")
            promulgation_no = item.get("promulgation_no", "")
            law_type = item.get("law_type", "")

            for chunk_idx, chunk_content, article_no in chunks:
                if article_no:
                    prefixed_content = f"[법령] {article_no} {chunk_content}"
                else:
                    prefixed_content = f"[법령] {chunk_content}"

                yield {
                    "source_id": source_id,
                    "chunk_index": chunk_idx,
                    "title": law_name,
                    "content": prefixed_content,
                    "enforcement_date": enforcement_date,
                    "department": ministry,
                    "total_chunks": total_chunks,
                    "promulgation_date": promulgation_date,
                    "promulgation_no": promulgation_no,
                    "law_type": law_type,
                    "article_no": article_no or "",
                }

        if use_streaming and f is not None:
            f.close()


def collate_chunks(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """DataLoader용 collate 함수 - 청크 리스트를 배치 딕셔너리로 변환"""
    if not batch:
        return {}

    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


# ============================================================================
# 판례 청킹
# ============================================================================

@dataclass
class PrecedentChunkConfig:
    chunk_size: int = CONFIG["PRECEDENT_CHUNK_SIZE"]
    chunk_overlap: int = CONFIG["PRECEDENT_CHUNK_OVERLAP"]
    min_chunk_size: int = CONFIG["PRECEDENT_MIN_CHUNK_SIZE"]


def chunk_precedent_text(text: str, config: PrecedentChunkConfig) -> List[tuple]:
    """판례 텍스트를 청크로 분할"""
    if not text or len(text) < config.min_chunk_size:
        return [(0, text)] if text else []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + config.chunk_size, len(text))

        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                sep_pos = text.rfind(sep, start + config.min_chunk_size, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_content = text[start:end].strip()
        if chunk_content and len(chunk_content) >= config.min_chunk_size:
            chunks.append((chunk_index, chunk_content))
            chunk_index += 1

        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 법령 청킹 (조문 → 항 단위)
# ============================================================================

@dataclass
class LawChunkConfig:
    max_tokens: int = CONFIG["LAW_MAX_TOKENS"]
    min_tokens: int = CONFIG["LAW_MIN_TOKENS"]


PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def estimate_tokens(text: str) -> int:
    """토큰 수 추정"""
    if not text:
        return 0
    korean_chars = len(re.findall(r"[가-힣]", text))
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)


def split_by_paragraphs(article_text: str) -> List[str]:
    """조문을 항(①②③) 단위로 분리"""
    parts = PARAGRAPH_PATTERN.split(article_text)
    if len(parts) <= 1:
        return [article_text]

    result = []
    if parts[0].strip():
        result.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            paragraph = parts[i] + parts[i + 1]
            if paragraph.strip():
                result.append(paragraph.strip())
        elif parts[i].strip():
            result.append(parts[i].strip())

    return result


def chunk_law_content(content: str, config: LawChunkConfig) -> List[tuple]:
    """법령 내용을 청크로 분할"""
    if not content:
        return []

    articles = content.split("\n\n")
    chunks = []
    chunk_index = 0
    article_no_pattern = re.compile(r"^(제\d+조(?:의\d+)?)")

    for article in articles:
        article = article.strip()
        if not article:
            continue

        match = article_no_pattern.match(article)
        article_no = match.group(1) if match else None
        tokens = estimate_tokens(article)

        if tokens <= config.max_tokens:
            if tokens >= config.min_tokens:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
            elif chunks:
                prev_idx, prev_text, prev_article_no = chunks[-1]
                chunks[-1] = (prev_idx, prev_text + "\n\n" + article, prev_article_no)
            else:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
        else:
            paragraphs = split_by_paragraphs(article)
            current_chunk = ""
            current_article_no = article_no

            for para in paragraphs:
                if not current_chunk:
                    current_chunk = para
                elif estimate_tokens(current_chunk + "\n" + para) <= config.max_tokens:
                    current_chunk += "\n" + para
                else:
                    if estimate_tokens(current_chunk) >= config.min_tokens:
                        chunks.append((chunk_index, current_chunk, current_article_no))
                        chunk_index += 1
                    current_chunk = para

            if current_chunk:
                if estimate_tokens(current_chunk) >= config.min_tokens:
                    chunks.append((chunk_index, current_chunk, current_article_no))
                    chunk_index += 1
                elif chunks:
                    prev_idx, prev_text, prev_article_no = chunks[-1]
                    chunks[-1] = (prev_idx, prev_text + "\n" + current_chunk, prev_article_no)

    return chunks


# ============================================================================
# 법령 임베딩 생성
# ============================================================================

def create_law_embeddings(
    source_path: str,
    store: LanceDBStore = None,
    batch_size: int = None,
    reset: bool = False,
    num_workers: int = None,
    prefetch_factor: int = 2,
    auto_config: bool = True,
) -> dict:
    """
    법령 임베딩 생성 (DataLoader 사용, 디바이스 자동 감지)

    Args:
        source_path: JSON 파일 경로
        store: LanceDBStore 인스턴스
        batch_size: 배치 크기 (None=자동)
        reset: 기존 데이터 삭제 여부
        num_workers: DataLoader 워커 수 (None=자동)
        prefetch_factor: 프리페치할 배치 수
        auto_config: 디바이스에 따라 자동 설정 사용
    """
    # 디바이스 감지 및 최적 설정
    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
        num_workers = num_workers if num_workers is not None else optimal.num_workers
        gc_interval = optimal.gc_interval
    else:
        batch_size = batch_size or CONFIG["BATCH_SIZE"]
        num_workers = num_workers if num_workers is not None else 0
        gc_interval = 10

    chunk_config = LawChunkConfig()

    if store is None:
        store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    # 데이터 로드 (전체 개수 파악)
    print(f"[INFO] Loading law data from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    stats["total_docs"] = len(items)
    print(f"[INFO] Total laws: {len(items):,}")

    if not items:
        return stats

    if reset:
        deleted = store.delete_by_type("법령")
        print(f"[INFO] Deleted {deleted} existing law chunks")

    existing_source_ids = store.get_existing_source_ids("법령") if not reset else set()
    if existing_source_ids:
        print(f"[INFO] Found {len(existing_source_ids)} laws already embedded")
        stats["skipped"] = len(existing_source_ids)

    # 모델 로드 (디바이스 자동 감지)
    get_embedding_model(device_info.device)

    # DataLoader 생성
    dataset = LawIterableDataset(
        source_path=source_path,
        chunk_config=chunk_config,
        existing_source_ids=existing_source_ids,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_chunks,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Settings: batch_size={batch_size}, num_workers={num_workers}, gc_interval={gc_interval}")

    # 배치 처리
    processed_source_ids = set()
    batch_count = 0
    pbar = tqdm(dataloader, desc="법령 임베딩", unit="batch")

    for batch in pbar:
        if not batch:
            continue

        batch_count += 1

        try:
            # 임베딩 생성
            embeddings = create_embeddings(batch["content"], device_info.device)

            # LanceDB에 저장
            store.add_law_documents(
                source_ids=batch["source_id"],
                chunk_indices=batch["chunk_index"],
                embeddings=embeddings,
                titles=batch["title"],
                contents=batch["content"],
                enforcement_dates=batch["enforcement_date"],
                departments=batch["department"],
                total_chunks_list=batch["total_chunks"],
                promulgation_dates=batch["promulgation_date"],
                promulgation_nos=batch["promulgation_no"],
                law_types=batch["law_type"],
                article_nos=batch["article_no"],
            )

            # 통계 업데이트
            stats["total_chunks"] += len(batch["source_id"])
            for sid in batch["source_id"]:
                if sid not in processed_source_ids:
                    processed_source_ids.add(sid)
                    stats["processed_docs"] += 1

        except Exception as e:
            stats["errors"] += len(batch.get("source_id", []))
            tqdm.write(f"  [ERROR] Batch error: {e}")

        # 메모리 정리 (gc_interval 배치마다)
        if batch_count % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 진행률 업데이트
        pbar.set_postfix(docs=stats["processed_docs"], chunks=stats["total_chunks"])

    # 최종 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


# ============================================================================
# 판례 임베딩 생성
# ============================================================================

def create_precedent_embeddings(
    source_path: str,
    store: LanceDBStore = None,
    batch_size: int = None,
    reset: bool = False,
    num_workers: int = None,
    prefetch_factor: int = 2,
    auto_config: bool = True,
) -> dict:
    """
    판례 임베딩 생성 (DataLoader 사용, 디바이스 자동 감지)

    Args:
        source_path: JSON 파일 경로
        store: LanceDBStore 인스턴스
        batch_size: 배치 크기 (None=자동)
        reset: 기존 데이터 삭제 여부
        num_workers: DataLoader 워커 수 (None=자동)
        prefetch_factor: 프리페치할 배치 수
        auto_config: 디바이스에 따라 자동 설정 사용
    """
    # 디바이스 감지 및 최적 설정
    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
        num_workers = num_workers if num_workers is not None else optimal.num_workers
        gc_interval = optimal.gc_interval
    else:
        batch_size = batch_size or CONFIG["BATCH_SIZE"]
        num_workers = num_workers if num_workers is not None else 0
        gc_interval = 10

    chunk_config = PrecedentChunkConfig()

    if store is None:
        store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"[INFO] Loading precedent data from: {source_path}")

    if reset:
        deleted = store.delete_by_type("판례")
        print(f"[INFO] Deleted {deleted} existing precedent chunks")

    existing_source_ids = store.get_existing_source_ids("판례") if not reset else set()
    if existing_source_ids:
        print(f"[INFO] Found {len(existing_source_ids)} precedents already embedded")
        stats["skipped"] = len(existing_source_ids)

    # 전체 개수 파악 (진행률 표시용)
    print("[INFO] Counting total items...")
    if IJSON_AVAILABLE:
        total_items = 0
        with open(source_path, "rb") as f:
            for _ in ijson.items(f, "items.item"):
                total_items += 1
        stats["total_docs"] = total_items
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", data.get("precedents", []))
        stats["total_docs"] = len(items)

    print(f"[INFO] Total precedents: {stats['total_docs']:,}")

    if stats["total_docs"] == 0:
        return stats

    # 모델 로드 (디바이스 자동 감지)
    get_embedding_model(device_info.device)

    # DataLoader 생성
    dataset = PrecedentIterableDataset(
        source_path=source_path,
        chunk_config=chunk_config,
        existing_source_ids=existing_source_ids,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_chunks,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Settings: batch_size={batch_size}, num_workers={num_workers}, gc_interval={gc_interval}")

    # 배치 처리
    processed_source_ids = set()
    batch_count = 0
    pbar = tqdm(dataloader, desc="판례 임베딩", unit="batch")

    for batch in pbar:
        if not batch:
            continue

        batch_count += 1

        try:
            # 임베딩 생성
            embeddings = create_embeddings(batch["content"], device_info.device)

            # LanceDB에 저장 (ruling, claim, reasoning 제외)
            store.add_precedent_documents(
                source_ids=batch["source_id"],
                chunk_indices=batch["chunk_index"],
                embeddings=embeddings,
                titles=batch["title"],
                contents=batch["content"],
                decision_dates=batch["decision_date"],
                court_names=batch["court_name"],
                total_chunks_list=batch["total_chunks"],
                case_numbers=batch["case_number"],
                case_types=batch["case_type"],
                reference_provisions_list=batch["reference_provisions"],
                reference_cases_list=batch["reference_cases"],
            )

            # 통계 업데이트
            stats["total_chunks"] += len(batch["source_id"])
            for sid in batch["source_id"]:
                if sid not in processed_source_ids:
                    processed_source_ids.add(sid)
                    stats["processed_docs"] += 1

        except Exception as e:
            stats["errors"] += len(batch.get("source_id", []))
            tqdm.write(f"  [ERROR] Batch error: {e}")

        # 메모리 정리 (gc_interval 배치마다)
        if batch_count % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 진행률 업데이트
        pbar.set_postfix(docs=stats["processed_docs"], chunks=stats["total_chunks"])

    # 최종 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


# ============================================================================
# 통계 출력
# ============================================================================

def show_stats(store: LanceDBStore = None):
    """LanceDB 통계 출력"""
    if store is None:
        store = LanceDBStore()

    print("\n" + "=" * 60)
    print("LanceDB Statistics")
    print("=" * 60)

    total = store.count()
    print(f"Total chunks: {total:,}")

    if total > 0:
        law_count = store.count_by_type("법령")
        precedent_count = store.count_by_type("판례")

        print(f"\nBy data_type:")
        print(f"  - 법령: {law_count:,}")
        print(f"  - 판례: {precedent_count:,}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수 (Colab에서 호출)"""
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB 임베딩 생성 (Colab용)")
    parser.add_argument("--type", choices=["law", "precedent", "all"], required=True)
    parser.add_argument("--source", type=str, required=True, help="JSON 파일 경로")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (Colab Edition)")
    print("=" * 60)
    print(f"Model: {CONFIG['EMBEDDING_MODEL']}")
    print(f"Vector dimension: {VECTOR_DIM}")

    if args.stats:
        show_stats()
        return

    store = LanceDBStore()
    start_time = datetime.now()

    if args.type == "law":
        stats = create_law_embeddings(args.source, store, args.batch_size, args.reset)
    elif args.type == "precedent":
        stats = create_precedent_embeddings(args.source, store, args.batch_size, args.reset)
    else:
        # all
        print("\n[법령 임베딩]")
        law_stats = create_law_embeddings(args.source, store, args.batch_size, args.reset)
        print("\n[판례 임베딩]")
        # 판례는 별도 파일 필요
        stats = {"law": law_stats}

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nElapsed time: {elapsed}")

    show_stats(store)


# ============================================================================
# Colab에서 직접 실행용 함수들
# ============================================================================

def run_law_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: int = None,
    num_workers: int = None,
    auto_config: bool = True,
):
    """
    법령 임베딩 실행 (디바이스 자동 감지)

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 처음부터 시작
        batch_size: 배치 크기 (None=자동)
        num_workers: DataLoader 병렬 워커 수 (None=자동)
        auto_config: 디바이스에 따라 자동 설정 사용

    예시:
        # 자동 설정 (권장)
        run_law_embedding("law_cleaned.json", reset=True)

        # 수동 설정
        run_law_embedding("law_cleaned.json", batch_size=50, auto_config=False)
    """
    print("=" * 60)
    print("법령 임베딩 시작")
    print("=" * 60)

    # 디바이스 정보 출력
    device_info, optimal = print_device_info()

    if auto_config:
        print("\n[자동 설정 사용]")
    else:
        print(f"\n[수동 설정] batch_size={batch_size}, num_workers={num_workers}")

    print(f"  reset: {reset}")

    store = LanceDBStore()
    start = datetime.now()

    stats = create_law_embeddings(
        source_path=source_path,
        store=store,
        batch_size=batch_size,
        reset=reset,
        num_workers=num_workers,
        auto_config=auto_config,
    )

    elapsed = datetime.now() - start
    print(f"\n완료! 소요시간: {elapsed}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    show_stats(store)
    return stats


def run_precedent_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: int = None,
    num_workers: int = None,
    auto_config: bool = True,
):
    """
    판례 임베딩 실행 (디바이스 자동 감지)

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 처음부터 시작
        batch_size: 배치 크기 (None=자동)
        num_workers: DataLoader 병렬 워커 수 (None=자동)
        auto_config: 디바이스에 따라 자동 설정 사용

    예시:
        # 자동 설정 (권장)
        run_precedent_embedding("precedents.json", reset=True)

        # 수동 설정
        run_precedent_embedding("precedents.json", batch_size=30, auto_config=False)
    """
    print("=" * 60)
    print("판례 임베딩 시작")
    print("=" * 60)

    # 디바이스 정보 출력
    device_info, optimal = print_device_info()

    if auto_config:
        print("\n[자동 설정 사용]")
    else:
        print(f"\n[수동 설정] batch_size={batch_size}, num_workers={num_workers}")

    print(f"  reset: {reset}")

    store = LanceDBStore()
    start = datetime.now()

    stats = create_precedent_embeddings(
        source_path=source_path,
        store=store,
        batch_size=batch_size,
        reset=reset,
        num_workers=num_workers,
        auto_config=auto_config,
    )

    elapsed = datetime.now() - start
    print(f"\n완료! 소요시간: {elapsed}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    show_stats(store)
    return stats


# ============================================================================
# Colab 사용법
# ============================================================================
"""
사용하는 방법:

0. 지원 환경:
   - Google Colab (T4 GPU)
   - Mac M1/M2/M3 (MPS)
   - NVIDIA CUDA GPU (3060, 4060, 5060 Ti 등)
   - CPU only

1. 패키지 설치:
   !pip install lancedb sentence-transformers pyarrow ijson psutil -q

2. 디바이스 정보 확인:
   print_device_info()  # 현재 환경 및 권장 설정 출력

3. 데이터 업로드 후 함수 호출:

   # 자동 설정 사용 (권장) - 디바이스에 맞게 자동 최적화
   run_law_embedding("law_cleaned.json", reset=True)
   run_precedent_embedding("precedents.json", reset=True)

   # 수동 설정 (필요시)
   run_precedent_embedding("precedents.json", batch_size=30, auto_config=False)

   # 통계 확인
   show_stats()

4. 중단 후 재개:
   - reset=False (기본값)로 호출하면 이미 임베딩된 문서는 건너뜁니다.
   run_precedent_embedding("precedents.json")  # 이어서 계속

5. 메모리 정리:
   clear_model_cache()  # 모델 언로드 및 GPU 메모리 해제

환경별 자동 설정:
   - Mac M3 16GB:    batch_size=50,  num_workers=0
   - CUDA 3060 6GB:  batch_size=50,  num_workers=2
   - CUDA 5060Ti:    batch_size=100, num_workers=4
   - Colab T4:       batch_size=100, num_workers=0

CLI 사용법 (터미널):
   python colab_lancedb_embeddings.py --type law --source law_cleaned.json --reset
"""


def is_notebook():
    """Jupyter/Colab 환경인지 확인"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


if __name__ == "__main__":
    # Colab/Jupyter에서는 main() 자동 실행 안 함
    if not is_notebook():
        main()
    else:
        print("=" * 60)
        print("LanceDB Embedding Creator (Multi-Device Support)")
        print("=" * 60)

        # 디바이스 정보 출력
        device_info, optimal = print_device_info()

        print("\n사용 가능한 함수:")
        print("  - print_device_info()     # 디바이스 정보 확인")
        print("  - run_law_embedding()     # 법령 임베딩")
        print("  - run_precedent_embedding() # 판례 임베딩")
        print("  - show_stats()            # 통계 확인")
        print("  - clear_model_cache()     # 메모리 정리")
        print("\n예시:")
        print("  run_law_embedding('law_cleaned.json', reset=True)")
        print("  run_precedent_embedding('precedents.json', reset=True)")
