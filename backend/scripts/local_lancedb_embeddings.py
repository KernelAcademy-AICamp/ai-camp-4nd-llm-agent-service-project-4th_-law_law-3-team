#!/usr/bin/env python3
"""
로컬 LanceDB 임베딩 생성 스크립트

embedding_common/ 패키지를 활용한 멀티 하드웨어 임베딩 스크립트.
runpod_lancedb_embeddings.py와 동일한 스키마/ID 형식을 사용합니다.

하드웨어 자동 감지:
- 5060Ti Desktop: batch=128, 온도 모니터링 OFF
- 3060 Laptop: batch=50, 온도 모니터링 ON (85°C)
- Mac M3 16GB: batch=50, 온도 모니터링 OFF (MPS)
- CPU: batch=20, 온도 모니터링 OFF

사용법:
    # 전체 임베딩 (자동 감지)
    uv run --no-sync python scripts/local_lancedb_embeddings.py --type all

    # 하드웨어 프로필 지정
    uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --profile laptop

    # 리셋 후 재생성
    uv run --no-sync python scripts/local_lancedb_embeddings.py --type all --reset

    # 통계 확인
    uv run --no-sync python scripts/local_lancedb_embeddings.py --stats

    # 데이터 검증
    uv run --no-sync python scripts/local_lancedb_embeddings.py --verify
"""

import gc
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 백엔드 루트를 sys.path에 추가 (embedding_common import 전에 필요)
_backend_root = Path(__file__).parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.embedding_common.config import (  # noqa: E402
    DEFAULT_CONFIG,
    HardwareProfile,
    get_hardware_profile,
    get_optimal_config,
)
from scripts.embedding_common.device import (  # noqa: E402
    get_device_info,
    print_device_info,
)
from scripts.embedding_common.memory import (  # noqa: E402
    check_memory_pressure,
    print_memory_status,
)
from scripts.embedding_common.model import (  # noqa: E402
    clear_memory,
    create_embeddings,
    get_embedding_model,
)
from scripts.embedding_common.schema import (  # noqa: E402
    VECTOR_DIM,
    create_law_chunk,
    create_precedent_chunk,
)
from scripts.embedding_common.store import EmbeddingStore  # noqa: E402
from scripts.embedding_common.temperature import TemperatureMonitor  # noqa: E402

try:
    import ijson

    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("[WARN] ijson 미설치. 전체 JSON 로드 사용 (메모리 사용량 증가)")


# ============================================================================
# 체크포인트 관리
# ============================================================================

CHECKPOINT_DIR = Path("./embedding_checkpoints")


@dataclass
class Checkpoint:
    """임베딩 체크포인트"""

    data_type: str
    last_source_id: str = ""
    processed_count: int = 0
    chunk_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_type": self.data_type,
            "last_source_id": self.last_source_id,
            "processed_count": self.processed_count,
            "chunk_count": self.chunk_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            data_type=data["data_type"],
            last_source_id=data.get("last_source_id", ""),
            processed_count=data.get("processed_count", 0),
            chunk_count=data.get("chunk_count", 0),
            skipped_count=data.get("skipped_count", 0),
            error_count=data.get("error_count", 0),
            timestamp=data.get("timestamp", ""),
        )


def save_checkpoint(checkpoint: Checkpoint) -> None:
    """체크포인트 저장"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_{checkpoint.data_type}.json"
    checkpoint.timestamp = datetime.now().isoformat()
    path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2))


def load_checkpoint(data_type: str) -> Optional[Checkpoint]:
    """체크포인트 로드"""
    path = CHECKPOINT_DIR / f"checkpoint_{data_type}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Checkpoint.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def clear_checkpoint(data_type: str) -> None:
    """체크포인트 삭제"""
    path = CHECKPOINT_DIR / f"checkpoint_{data_type}.json"
    if path.exists():
        path.unlink()


# ============================================================================
# MeCab 토크나이저 로딩
# ============================================================================


def _load_mecab_tokenizer() -> Any:
    """MeCab 토크나이저 로드 (없으면 None)"""
    try:
        import importlib.util

        module_path = (
            Path(__file__).parent.parent
            / "app"
            / "tools"
            / "vectorstore"
            / "mecab_tokenizer.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mecab_tokenizer", str(module_path)
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tokenizer = mod.MeCabTokenizer()
            if tokenizer.is_available:
                print("[INFO] MeCab 토크나이저 초기화 완료 (FTS 사전 토크나이징)")
                return tokenizer
            print("[WARN] MeCab 사용 불가. content_tokenized = None")
        else:
            print("[WARN] MeCab 토크나이저 모듈 미발견. content_tokenized = None")
    except Exception as e:
        print(f"[WARN] MeCab 토크나이저 초기화 실패: {e}. content_tokenized = None")
    return None


# ============================================================================
# JSON 로딩 헬퍼
# ============================================================================


def _detect_json_format(file_path: str) -> str:
    """JSON 파일 형식 감지 (array 또는 object)"""
    with open(file_path, "rb") as f:
        first_char = f.read(1).decode("utf-8").strip()
        while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
            first_char = f.read(1).decode("utf-8")
    return "array" if first_char == "[" else "object"


def _load_json_streaming(
    file_path: str,
) -> Optional[tuple[Any, Any]]:
    """JSON 스트리밍 로드 (ijson)"""
    if not IJSON_AVAILABLE:
        return None
    json_format = _detect_json_format(file_path)
    f = open(file_path, "rb")  # noqa: SIM115
    if json_format == "array":
        return f, ijson.items(f, "item")
    return f, ijson.items(f, "items.item")


def _load_json_full(file_path: str) -> list[dict[str, Any]]:
    """JSON 전체 로드"""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("items", data.get("precedents", []))


# ============================================================================
# 법령 청킹 로직
# ============================================================================

PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")

LAW_MAX_TOKENS = int(DEFAULT_CONFIG["LAW_MAX_TOKENS"])
LAW_MIN_TOKENS = int(DEFAULT_CONFIG["LAW_MIN_TOKENS"])
PRECEDENT_CHUNK_SIZE = int(DEFAULT_CONFIG["PRECEDENT_CHUNK_SIZE"])
PRECEDENT_CHUNK_OVERLAP = int(DEFAULT_CONFIG["PRECEDENT_CHUNK_OVERLAP"])
PRECEDENT_MIN_CHUNK_SIZE = int(DEFAULT_CONFIG["PRECEDENT_MIN_CHUNK_SIZE"])


def _estimate_tokens(text: str) -> int:
    """토큰 수 추정"""
    if not text:
        return 0
    korean_chars = len(re.findall(r"[가-힣]", text))
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)


def _split_by_paragraphs(article_text: str) -> list[str]:
    """조문을 항 단위로 분리"""
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


def chunk_law_content(content: str) -> list[tuple[int, str, Optional[str]]]:
    """법령 내용을 청크로 분할. 반환: [(chunk_index, content, article_no), ...]"""
    if not content:
        return []
    articles = content.split("\n\n")
    chunks: list[tuple[int, str, Optional[str]]] = []
    chunk_index = 0
    article_no_pattern = re.compile(r"^(제\d+조(?:의\d+)?)")

    for article in articles:
        article = article.strip()
        if not article:
            continue
        match = article_no_pattern.match(article)
        article_no = match.group(1) if match else None
        tokens = _estimate_tokens(article)

        if tokens <= LAW_MAX_TOKENS:
            if tokens >= LAW_MIN_TOKENS:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
            elif chunks:
                prev_idx, prev_text, prev_art = chunks[-1]
                chunks[-1] = (prev_idx, prev_text + "\n\n" + article, prev_art)
            else:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
        else:
            paragraphs = _split_by_paragraphs(article)
            current_chunk = ""
            for para in paragraphs:
                if not current_chunk:
                    current_chunk = para
                elif (
                    _estimate_tokens(current_chunk + "\n" + para) <= LAW_MAX_TOKENS
                ):
                    current_chunk += "\n" + para
                else:
                    if _estimate_tokens(current_chunk) >= LAW_MIN_TOKENS:
                        chunks.append((chunk_index, current_chunk, article_no))
                        chunk_index += 1
                    current_chunk = para
            if current_chunk:
                if _estimate_tokens(current_chunk) >= LAW_MIN_TOKENS:
                    chunks.append((chunk_index, current_chunk, article_no))
                    chunk_index += 1
                elif chunks:
                    prev_idx, prev_text, prev_art = chunks[-1]
                    chunks[-1] = (
                        prev_idx,
                        prev_text + "\n" + current_chunk,
                        prev_art,
                    )
    return chunks


def chunk_precedent_text(
    text: str,
) -> list[tuple[int, str]]:
    """판례 텍스트를 청크로 분할. 반환: [(chunk_index, content), ...]"""
    if not text or len(text) < PRECEDENT_MIN_CHUNK_SIZE:
        return [(0, text)] if text else []

    if len(text) <= PRECEDENT_CHUNK_SIZE:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + PRECEDENT_CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                sep_pos = text.rfind(sep, start + PRECEDENT_MIN_CHUNK_SIZE, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_content = text[start:end].strip()
        if chunk_content and len(chunk_content) >= PRECEDENT_MIN_CHUNK_SIZE:
            chunks.append((chunk_index, chunk_content))
            chunk_index += 1

        new_start = end - PRECEDENT_CHUNK_OVERLAP
        if new_start <= start:
            break
        start = new_start
        if start >= len(text) - PRECEDENT_MIN_CHUNK_SIZE:
            break

    return chunks if chunks else [(0, text)]


# ============================================================================
# 임베딩 통계
# ============================================================================


@dataclass
class EmbeddingStats:
    """임베딩 처리 통계"""

    total_docs: int = 0
    processed_docs: int = 0
    total_chunks: int = 0
    skipped: int = 0
    errors: int = 0
    device: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "total_chunks": self.total_chunks,
            "skipped": self.skipped,
            "errors": self.errors,
            "device": self.device,
        }


# ============================================================================
# 로컬 임베딩 프로세서
# ============================================================================


class LocalEmbeddingProcessor(ABC):
    """
    로컬 임베딩 프로세서 (베이스 클래스)

    embedding_common/ 패키지를 사용하며, 체크포인트/재개 기능을 지원합니다.
    """

    def __init__(
        self,
        data_type: str,
        profile: Optional[HardwareProfile] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.data_type = data_type
        self.device_info = get_device_info()
        self.profile = profile or get_hardware_profile(self.device_info)
        self.optimal_config = get_optimal_config(self.device_info, self.profile)
        self.batch_size = batch_size or self.optimal_config.batch_size
        self.store = EmbeddingStore()
        self.stats = EmbeddingStats(device=str(self.device_info))
        self._mecab = _load_mecab_tokenizer()

        # 온도 모니터링
        self.temp_monitor: Optional[TemperatureMonitor] = None
        if self.optimal_config.temp_monitoring:
            self.temp_monitor = TemperatureMonitor(
                threshold=self.optimal_config.temp_threshold,
            )
            print(
                f"[THERMAL] 온도 모니터링 활성화 "
                f"(임계값: {self.optimal_config.temp_threshold}°C)"
            )

    @abstractmethod
    def extract_source_id(self, item: dict[str, Any], idx: int) -> str:
        """소스 ID 추출"""

    @abstractmethod
    def extract_text_for_embedding(self, item: dict[str, Any]) -> str:
        """임베딩 텍스트 추출"""

    @abstractmethod
    def chunk_text(self, text: str) -> list[tuple[int, str]]:
        """텍스트 청킹"""

    @abstractmethod
    def extract_metadata(self, item: dict[str, Any]) -> dict[str, str]:
        """메타데이터 추출"""

    @abstractmethod
    def build_chunk_record(
        self,
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        vector: list[float],
        metadata: dict[str, str],
        content_tokenized: Optional[str],
    ) -> dict[str, Any]:
        """청크 레코드 빌드"""

    def _tokenize(self, text: str) -> Optional[str]:
        """MeCab 사전 토크나이징"""
        if self._mecab:
            return self._mecab.tokenize(text)
        return None

    def run(
        self,
        source_path: str,
        reset: bool = False,
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        임베딩 실행

        Args:
            source_path: JSON 파일 경로
            reset: 기존 데이터 삭제 후 시작
            resume: 체크포인트에서 재개 (기본 True)
        """
        print("=" * 60)
        print(f"{self.data_type} 임베딩 시작 (로컬)")
        print("=" * 60)
        print(f"  Device: {self.device_info}")
        print(f"  Profile: {self.profile.value}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  온도 모니터링: {'ON' if self.temp_monitor else 'OFF'}")
        print(f"  Source: {source_path}")
        print(f"  Reset: {reset}")

        # 리셋 처리
        if reset:
            deleted = self.store.count_by_type(self.data_type)
            if self.store.table is not None and deleted > 0:
                self.store.table.delete(f"data_type = '{self.data_type}'")
            print(f"[INFO] 기존 {self.data_type} 데이터 삭제: {deleted}건")
            clear_checkpoint(self.data_type)

        # 체크포인트 로드
        checkpoint = load_checkpoint(self.data_type) if (resume and not reset) else None
        resume_after_id = ""
        if checkpoint:
            resume_after_id = checkpoint.last_source_id
            self.stats.processed_docs = checkpoint.processed_count
            self.stats.total_chunks = checkpoint.chunk_count
            self.stats.skipped = checkpoint.skipped_count
            self.stats.errors = checkpoint.error_count
            print(
                f"[RESUME] 체크포인트에서 재개: "
                f"processed={checkpoint.processed_count}, "
                f"chunks={checkpoint.chunk_count}, "
                f"last_id={resume_after_id}"
            )

        # 기존 source_id 조회 (증분 처리)
        existing_ids = (
            self.store.get_existing_source_ids(self.data_type) if not reset else set()
        )
        if existing_ids:
            print(f"[INFO] 기존 {self.data_type} 임베딩: {len(existing_ids)}건")

        # 데이터 로드
        file_handle, items = self._load_data(source_path)

        # 모델 로드
        print("[INFO] 임베딩 모델 로딩 중...")
        get_embedding_model(
            device=f"{self.device_info.device}:0"
            if self.device_info.device == "cuda"
            else self.device_info.device,
        )
        print("[INFO] 모델 로드 완료")

        # 배치 처리
        batch_records: list[dict[str, Any]] = []
        batch_contents: list[str] = []
        start_time = datetime.now()
        batch_count = 0
        found_resume_point = not bool(resume_after_id)
        current_batch_size = self.batch_size

        for idx, item in enumerate(tqdm(items, desc=f"{self.data_type} 처리", total=None)):
            source_id = self.extract_source_id(item, idx)

            # 체크포인트 재개 지점 찾기
            if not found_resume_point:
                if source_id == resume_after_id:
                    found_resume_point = True
                continue

            # 이미 처리된 항목 스킵
            if source_id in existing_ids:
                continue

            # 텍스트 추출
            text = self.extract_text_for_embedding(item)
            if not text:
                self.stats.skipped += 1
                continue

            # 청킹
            chunks = self.chunk_text(text)
            if not chunks:
                self.stats.skipped += 1
                continue

            total_chunks = len(chunks)
            self.stats.processed_docs += 1
            metadata = self.extract_metadata(item)

            # 배치에 청크 추가
            for chunk_idx, chunk_content in chunks:
                tokenized = self._tokenize(chunk_content)
                # 레코드는 벡터 없이 일단 저장 (배치 임베딩 후 벡터 추가)
                record = self.build_chunk_record(
                    source_id=source_id,
                    chunk_idx=chunk_idx,
                    chunk_content=chunk_content,
                    total_chunks=total_chunks,
                    vector=[],  # placeholder
                    metadata=metadata,
                    content_tokenized=tokenized,
                )
                batch_records.append(record)
                batch_contents.append(chunk_content)

            # 배치 크기 도달 시 처리
            if len(batch_contents) >= current_batch_size:
                saved = self._process_batch(batch_records, batch_contents)
                self.stats.total_chunks += saved
                batch_records = []
                batch_contents = []
                batch_count += 1

                # 메모리 정리
                clear_memory()

                # 온도 체크
                if self.temp_monitor:
                    current_batch_size, should_stop = self.temp_monitor.check_and_adjust(
                        current_batch_size
                    )
                    if should_stop:
                        print("[THERMAL] 과열 자동 중지! 체크포인트 저장 중...")
                        save_checkpoint(
                            Checkpoint(
                                data_type=self.data_type,
                                last_source_id=source_id,
                                processed_count=self.stats.processed_docs,
                                chunk_count=self.stats.total_chunks,
                                skipped_count=self.stats.skipped,
                                error_count=self.stats.errors,
                            )
                        )
                        break

                # 메모리 압력 체크
                if check_memory_pressure(threshold_percent=90.0):
                    print("[MEM] 메모리 압력 감지! GC 실행 중...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 주기적 체크포인트 (50배치마다)
                if batch_count % 50 == 0:
                    save_checkpoint(
                        Checkpoint(
                            data_type=self.data_type,
                            last_source_id=source_id,
                            processed_count=self.stats.processed_docs,
                            chunk_count=self.stats.total_chunks,
                            skipped_count=self.stats.skipped,
                            error_count=self.stats.errors,
                        )
                    )
                    print_memory_status()

        # 남은 배치 처리
        if batch_contents:
            saved = self._process_batch(batch_records, batch_contents)
            self.stats.total_chunks += saved

        # 파일 핸들 정리
        if file_handle:
            file_handle.close()

        # 최종 정리
        clear_memory()

        # FTS 인덱스 생성
        if self.store.table is not None:
            try:
                self.store.table.create_fts_index("content_tokenized", replace=True)
                print("[INFO] FTS 인덱스 생성 완료")
            except Exception as e:
                print(f"[WARN] FTS 인덱스 생성 실패: {e}")

        # 체크포인트 정리 (완료 시)
        clear_checkpoint(self.data_type)

        # 결과 출력
        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print(f"{self.data_type} 임베딩 완료")
        print("=" * 60)
        print(f"  처리 문서: {self.stats.processed_docs:,}")
        print(f"  생성 청크: {self.stats.total_chunks:,}")
        print(f"  스킵: {self.stats.skipped:,}")
        print(f"  에러: {self.stats.errors}")
        print(f"  소요 시간: {elapsed:.1f}초")
        if self.stats.total_chunks > 0 and elapsed > 0:
            print(f"  처리 속도: {self.stats.total_chunks / elapsed:.1f} chunks/sec")

        return self.stats.to_dict()

    def _load_data(self, source_path: str) -> tuple[Any, Any]:
        """데이터 로드 (스트리밍 우선)"""
        if IJSON_AVAILABLE:
            result = _load_json_streaming(source_path)
            if result:
                print("[INFO] 스트리밍 모드 (ijson)")
                return result
        print("[INFO] 전체 로드 모드")
        return None, _load_json_full(source_path)

    def _process_batch(
        self,
        records: list[dict[str, Any]],
        contents: list[str],
    ) -> int:
        """배치 임베딩 + 저장"""
        if not contents:
            return 0
        try:
            embeddings = create_embeddings(
                contents,
                batch_size=len(contents),
            )
            for i, record in enumerate(records):
                record["vector"] = embeddings[i]
            self.store.add_batch(records)
            return len(records)
        except (RuntimeError, ValueError, MemoryError) as e:
            print(f"[ERROR] 배치 처리 실패: {type(e).__name__}: {e}")
            self.stats.errors += len(contents)
            return 0


# ============================================================================
# 법령 임베딩 프로세서
# ============================================================================


class LawEmbeddingProcessor(LocalEmbeddingProcessor):
    """법령 임베딩 프로세서"""

    def __init__(
        self,
        profile: Optional[HardwareProfile] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__("법령", profile=profile, batch_size=batch_size)

    def extract_source_id(self, item: dict[str, Any], idx: int) -> str:
        return item.get("law_id", "")

    def extract_text_for_embedding(self, item: dict[str, Any]) -> str:
        return item.get("content", "")

    def chunk_text(self, text: str) -> list[tuple[int, str]]:
        chunks_with_art = chunk_law_content(text)
        # (chunk_index, content, article_no) → (chunk_index, content)
        return [(idx, content) for idx, content, _ in chunks_with_art]

    def extract_metadata(self, item: dict[str, Any]) -> dict[str, str]:
        return {
            "title": item.get("law_name", ""),
            "enforcement_date": item.get("enforcement_date", ""),
            "department": item.get("department", ""),
            "promulgation_date": item.get("promulgation_date", ""),
            "promulgation_no": item.get("promulgation_no", ""),
            "law_type": item.get("law_type", ""),
        }

    def build_chunk_record(
        self,
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        vector: list[float],
        metadata: dict[str, str],
        content_tokenized: Optional[str],
    ) -> dict[str, Any]:
        return create_law_chunk(
            source_id=source_id,
            chunk_index=chunk_idx,
            title=metadata["title"],
            content=chunk_content,
            vector=vector,
            enforcement_date=metadata["enforcement_date"],
            department=metadata["department"],
            total_chunks=total_chunks,
            promulgation_date=metadata.get("promulgation_date"),
            promulgation_no=metadata.get("promulgation_no"),
            law_type=metadata.get("law_type"),
            article_no=None,
            content_tokenized=content_tokenized,
        )


# ============================================================================
# 판례 임베딩 프로세서
# ============================================================================


class PrecedentEmbeddingProcessor(LocalEmbeddingProcessor):
    """판례 임베딩 프로세서"""

    def __init__(
        self,
        profile: Optional[HardwareProfile] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__("판례", profile=profile, batch_size=batch_size)

    def extract_source_id(self, item: dict[str, Any], idx: int) -> str:
        return str(item.get("판례정보일련번호", item.get("id", idx)))

    def extract_text_for_embedding(self, item: dict[str, Any]) -> str:
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
        return "\n".join(parts)

    def chunk_text(self, text: str) -> list[tuple[int, str]]:
        return chunk_precedent_text(text)

    def extract_metadata(self, item: dict[str, Any]) -> dict[str, str]:
        return {
            "case_name": item.get("사건명", item.get("case_name", "")),
            "case_number": item.get("사건번호", item.get("case_number", "")),
            "decision_date": item.get("선고일자", item.get("decision_date", "")),
            "court_name": item.get("법원명", item.get("court_name", "")),
            "case_type": item.get("사건종류명", item.get("case_type", "")),
            "judgment_type": item.get("판결유형", item.get("judgment_type", "")),
            "judgment_status": item.get("판결상태", item.get("judgment_status", "")),
            "reference_provisions": item.get(
                "참조조문", item.get("reference_provisions", "")
            ),
            "reference_cases": item.get(
                "참조판례", item.get("reference_cases", "")
            ),
        }

    def build_chunk_record(
        self,
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        vector: list[float],
        metadata: dict[str, str],
        content_tokenized: Optional[str],
    ) -> dict[str, Any]:
        return create_precedent_chunk(
            source_id=source_id,
            chunk_index=chunk_idx,
            title=metadata["case_name"],
            content=chunk_content,
            vector=vector,
            decision_date=metadata["decision_date"],
            court_name=metadata["court_name"],
            total_chunks=total_chunks,
            case_number=metadata.get("case_number"),
            case_type=metadata.get("case_type"),
            judgment_type=metadata.get("judgment_type"),
            judgment_status=metadata.get("judgment_status"),
            reference_provisions=metadata.get("reference_provisions"),
            reference_cases=metadata.get("reference_cases"),
            content_tokenized=content_tokenized,
        )


# ============================================================================
# 통계 및 검증
# ============================================================================


def show_stats() -> None:
    """LanceDB 통계 출력"""
    store = EmbeddingStore()
    print("\n" + "=" * 60)
    print("LanceDB Statistics")
    print("=" * 60)

    total = store.count()
    print(f"Total chunks: {total:,}")

    if total > 0:
        law_count = store.count_by_type("법령")
        prec_count = store.count_by_type("판례")
        print("\nBy data_type:")
        print(f"  - 법령: {law_count:,}")
        print(f"  - 판례: {prec_count:,}")


def verify_data() -> bool:
    """임베딩 데이터 검증"""
    store = EmbeddingStore()
    print("\n" + "=" * 60)
    print("데이터 검증")
    print("=" * 60)

    total = store.count()
    if total == 0:
        print("[ERROR] 데이터가 없습니다")
        return False

    law_count = store.count_by_type("법령")
    prec_count = store.count_by_type("판례")
    print(f"  총 청크: {total:,}")
    print(f"  법령: {law_count:,}")
    print(f"  판례: {prec_count:,}")

    # 벡터 차원 검증 (샘플)
    if store.table is not None:
        try:
            sample = store.table.head(5).to_pandas()
            if "vector" in sample.columns:
                vector_dims = sample["vector"].apply(len)
                print(f"  벡터 차원: {vector_dims.iloc[0]} (expected: {VECTOR_DIM})")
                if all(d == VECTOR_DIM for d in vector_dims):
                    print("  [OK] 벡터 차원 일치")
                else:
                    print("  [ERROR] 벡터 차원 불일치!")
                    return False

            # 필수 필드 존재 확인
            required_fields = [
                "id",
                "source_id",
                "data_type",
                "title",
                "content",
                "vector",
            ]
            missing = [f for f in required_fields if f not in sample.columns]
            if missing:
                print(f"  [ERROR] 누락 필드: {missing}")
                return False
            print("  [OK] 필수 필드 확인")
        except Exception as e:
            print(f"  [WARN] 샘플 검증 실패: {e}")

    print("\n검증 완료!")
    return True


# ============================================================================
# CLI 메인
# ============================================================================


def main() -> None:
    """CLI 메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="로컬 LanceDB 임베딩 생성 (멀티 하드웨어)",
    )
    parser.add_argument(
        "--type",
        choices=["law", "precedent", "all"],
        default="all",
        help="임베딩 대상 (기본: all)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="디바이스 (기본: auto)",
    )
    parser.add_argument(
        "--profile",
        choices=["desktop", "laptop", "mac", "cpu"],
        default=None,
        help="하드웨어 프로필 (기본: 자동 감지)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (기본: 프로필에 따라 자동)",
    )
    parser.add_argument(
        "--law-source",
        type=str,
        default="../data/law_cleaned.json",
        help="법령 JSON 경로",
    )
    parser.add_argument(
        "--precedent-source",
        type=str,
        default="../data/[cleaned]precedents_partial_done.json",
        help="판례 JSON 경로",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재생성",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="체크포인트 무시 (처음부터 시작)",
    )
    parser.add_argument("--stats", action="store_true", help="통계 출력")
    parser.add_argument("--verify", action="store_true", help="데이터 검증")

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (Local Edition)")
    print("=" * 60)
    print(f"Model: {DEFAULT_CONFIG['EMBEDDING_MODEL']}")
    print(f"Vector dimension: {VECTOR_DIM}")

    if args.stats:
        show_stats()
        return

    if args.verify:
        verify_data()
        return

    # 디바이스 정보 출력
    device_info, config = print_device_info()

    # 프로필 설정
    profile: Optional[HardwareProfile] = None
    if args.profile:
        profile = HardwareProfile(args.profile)

    if args.type in ("law", "all"):
        processor = LawEmbeddingProcessor(
            profile=profile,
            batch_size=args.batch_size,
        )
        processor.run(
            args.law_source,
            reset=args.reset,
            resume=not args.no_resume,
        )

    if args.type in ("precedent", "all"):
        processor = PrecedentEmbeddingProcessor(
            profile=profile,
            batch_size=args.batch_size,
        )
        processor.run(
            args.precedent_source,
            reset=args.reset,
            resume=not args.no_resume,
        )

    # 최종 통계
    show_stats()


if __name__ == "__main__":
    main()
