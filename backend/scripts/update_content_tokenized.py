"""
content_tokenized 컬럼 업데이트 스크립트

기존 LanceDB 테이블의 content_tokenized 컬럼만 재생성.
벡터 임베딩은 유지하고, MeCab + 법률용어사전으로 재토크나이징.

메모리 요구: ~2.5GB (Arrow 테이블 로드 + 토크나이징 버퍼)

Usage:
    cd backend

    # 현재 상태 확인
    uv run --no-sync python scripts/update_content_tokenized.py --stats

    # 드라이런 (10개 샘플 비교)
    uv run --no-sync python scripts/update_content_tokenized.py --dry-run

    # 실행 (법률용어사전 포함)
    uv run --no-sync python scripts/update_content_tokenized.py

    # 법률용어사전 없이 (MeCab 기본만)
    uv run --no-sync python scripts/update_content_tokenized.py --no-legal-dict

    # MeCab userdic 사용 (법률 복합명사 직접 인식)
    uv run --no-sync python scripts/update_content_tokenized.py --userdic
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

LANCEDB_PATH = BACKEND_DIR / "lancedb_data"
TABLE_NAME = "legal_chunks"
LEGAL_TERMS_JSON = PROJECT_ROOT / "data" / "lawterms_full.json"

PROGRESS_INTERVAL = 10_000  # 진행률 출력 간격
USERDIC_PATH = BACKEND_DIR / "data" / "mecab_userdic" / "legal_terms.dic"
DECOMP_MAP_PATH = BACKEND_DIR / "data" / "mecab_userdic" / "decomposition_map.json"


def _load_module(name: str, path: Path) -> object:
    """파일 경로에서 Python 모듈을 동적 로드"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"모듈 로드 실패: {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mecab_tokenizer(
    *,
    use_legal_dict: bool = True,
    use_userdic: bool = False,
) -> object:
    """MeCab 토크나이저 로드 (법률용어사전/userdic 포함)"""
    vectorstore_dir = BACKEND_DIR / "app" / "tools" / "vectorstore"

    tok_mod = _load_module("mecab_tokenizer", vectorstore_dir / "mecab_tokenizer.py")

    legal_dict = None
    if use_legal_dict and LEGAL_TERMS_JSON.exists():
        dict_mod = _load_module("legal_term_dict", vectorstore_dir / "legal_term_dict.py")
        ld = dict_mod.LegalTermDictionary()  # type: ignore[attr-defined]
        count = ld.load_from_json(str(LEGAL_TERMS_JSON))
        print(f"[INFO] 법률용어사전 로드: {count:,}개")

        # userdic 분해맵 로드
        if use_userdic and DECOMP_MAP_PATH.exists():
            decomp_count = ld.load_decomposition_map(DECOMP_MAP_PATH)
            print(f"[INFO] userdic 분해맵 로드: {decomp_count:,}개")

        legal_dict = ld
    elif use_legal_dict:
        print(f"[WARN] 법률용어 JSON 없음: {LEGAL_TERMS_JSON}")

    # userdic 경로 결정
    userdic_path = None
    if use_userdic and USERDIC_PATH.exists():
        userdic_path = str(USERDIC_PATH)

    tokenizer = tok_mod.MeCabTokenizer(  # type: ignore[attr-defined]
        legal_dict=legal_dict,
        userdic_path=userdic_path,
    )
    if tokenizer.is_available:
        mode_parts = []
        if legal_dict:
            mode_parts.append("법률용어사전")
        if userdic_path:
            mode_parts.append("userdic")
        mode_str = f"(+ {' + '.join(mode_parts)})" if mode_parts else "(MeCab 기본)"
        print(f"[INFO] MeCab 토크나이저 초기화 완료 {mode_str}")
    else:
        print("[WARN] MeCab 미설치, 공백 분리 fallback 사용")

    return tokenizer


def show_stats(table: lancedb.table.Table, total: int) -> None:
    """현재 content_tokenized 상태 출력"""
    print(f"테이블: {TABLE_NAME}")
    print(f"총 행 수: {total:,}")
    print()

    # content_tokenized 상태 확인 (벡터 제외하여 메모리 절약)
    cols = ["content_tokenized", "data_type"]
    arrow = table.to_arrow().select(cols)
    ct_col = arrow.column("content_tokenized")
    dt_col = arrow.column("data_type")

    null_count = ct_col.null_count
    non_null = total - null_count
    empty_count = sum(
        1 for v in ct_col if v.is_valid and v.as_py() == ""
    )
    has_value = non_null - empty_count

    print("content_tokenized 현황:")
    print(f"  값 있음: {has_value:,}")
    print(f"  Null:    {null_count:,}")
    print(f"  Empty:   {empty_count:,}")
    print()

    # 데이터 유형별 건수
    type_counts: dict[str, int] = {}
    for v in dt_col:
        dt = v.as_py()
        type_counts[dt] = type_counts.get(dt, 0) + 1

    print("데이터 유형별:")
    for dt, count in sorted(type_counts.items()):
        print(f"  {dt}: {count:,}")

    # 샘플 출력
    print()
    sample = table.search().limit(3).to_pandas()
    for _, row in sample.iterrows():
        ct = row.get("content_tokenized", "")
        ct_preview = str(ct)[:80] if ct else "(없음)"
        print(f"  [{row['data_type']}] {row['title'][:30]}")
        print(f"    → {ct_preview}...")


def dry_run(table: lancedb.table.Table, tokenizer: object) -> None:
    """10개 샘플로 변경 전후 비교"""
    # 판례 5개 + 법령 5개
    samples = []
    for dtype in ["판례", "법령"]:
        df = table.search().where(f"data_type = '{dtype}'").limit(5).to_pandas()
        samples.append(df)

    import pandas as pd

    sample = pd.concat(samples, ignore_index=True)

    diff_count = 0
    for _, row in sample.iterrows():
        old = row.get("content_tokenized", "")
        new = tokenizer.tokenize(row["content"])  # type: ignore[attr-defined]
        is_diff = str(old) != new

        if is_diff:
            diff_count += 1

        marker = "DIFF" if is_diff else " == "
        print(f"[{marker}] {row['data_type']} | {row['title'][:40]}")
        if is_diff:
            old_tokens = set(str(old).split()) if old else set()
            new_tokens = set(new.split())
            added = new_tokens - old_tokens
            removed = old_tokens - new_tokens
            if added:
                added_sample = sorted(added)[:10]
                suffix = f" ... 외 {len(added) - 10}개" if len(added) > 10 else ""
                print(f"     + 추가: {added_sample}{suffix}")
            if removed:
                removed_sample = sorted(removed)[:10]
                print(f"     - 제거: {removed_sample}")
        print()

    print(f"결과: {diff_count}/{len(sample)}개 변경")


def tokenize_content_column(
    content_col: pa.ChunkedArray,
    tokenizer: object,
    total: int,
) -> pa.Array:
    """content 컬럼을 토크나이징하여 새 Arrow 배열 반환"""
    tokenized: list[Optional[str]] = []
    processed = 0
    start_time = time.time()

    for chunk in content_col.chunks:
        for val in chunk:
            if val.is_valid:
                text = val.as_py()
                tokenized.append(
                    tokenizer.tokenize(text) if text else None  # type: ignore[attr-defined]
                )
            else:
                tokenized.append(None)

            processed += 1
            if processed % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (total - processed) / rate
                pct = processed / total * 100
                print(
                    f"  토크나이징 [{processed:>7,}/{total:,}]"
                    f" {pct:5.1f}% | {rate:,.0f} rows/s | ETA {eta:.0f}s"
                )

    return pa.array(tokenized, type=pa.utf8())


def update_table(
    db: lancedb.DBConnection,
    table: lancedb.table.Table,
    tokenizer: object,
    total: int,
) -> None:
    """content_tokenized 컬럼 업데이트 실행"""
    print()
    print("=" * 60)
    print(f"content_tokenized 업데이트 시작 ({total:,}행)")
    print("=" * 60)
    print()

    # 1. Arrow 테이블 로드
    print("[1/4] Arrow 테이블 로드 중... (약 2GB)")
    t0 = time.time()
    arrow = table.to_arrow()
    print(f"  완료 ({time.time() - t0:.1f}초, {arrow.num_rows:,}행)")
    print()

    # 2. 토크나이징
    print("[2/4] content 토크나이징 중...")
    t1 = time.time()
    content_col = arrow.column("content")
    new_tokenized = tokenize_content_column(content_col, tokenizer, total)
    print(f"  완료 ({time.time() - t1:.1f}초)")
    print()

    # 3. 컬럼 교체 및 테이블 재생성
    print("[3/4] 테이블 재생성 중...")
    t2 = time.time()
    col_idx = arrow.schema.get_field_index("content_tokenized")
    new_arrow = arrow.set_column(col_idx, "content_tokenized", new_tokenized)

    # 기존 테이블 삭제 후 새로 생성
    db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, new_arrow)
    print(f"  완료 ({time.time() - t2:.1f}초)")
    print()

    # 4. FTS 인덱스 재생성
    print("[4/4] FTS 인덱스 생성 중...")
    t3 = time.time()
    new_table = db.open_table(TABLE_NAME)
    new_table.create_fts_index("content_tokenized", replace=True)
    print(f"  완료 ({time.time() - t3:.1f}초)")

    # 결과 요약
    elapsed = time.time() - t0
    null_count = new_tokenized.null_count
    print()
    print("=" * 60)
    print(f"완료! {total:,}행 업데이트 (총 {elapsed:.1f}초)")
    print(f"  tokenized: {total - null_count:,}행")
    print(f"  null:      {null_count:,}행")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LanceDB content_tokenized 컬럼 업데이트",
    )
    parser.add_argument("--stats", action="store_true", help="현재 상태 확인")
    parser.add_argument("--dry-run", action="store_true", help="드라이런 (샘플 비교)")
    parser.add_argument(
        "--no-legal-dict", action="store_true", help="법률용어사전 없이 실행",
    )
    parser.add_argument(
        "--userdic", action="store_true", help="MeCab userdic 사용",
    )
    parser.add_argument(
        "--no-userdic", action="store_true", help="MeCab userdic 사용 안 함 (기본)",
    )
    args = parser.parse_args()

    # LanceDB 연결
    db = lancedb.connect(str(LANCEDB_PATH))
    try:
        table = db.open_table(TABLE_NAME)
    except Exception:
        print(f"[ERROR] 테이블 '{TABLE_NAME}'을 찾을 수 없습니다: {LANCEDB_PATH}")
        sys.exit(1)

    total = table.count_rows()

    if args.stats:
        show_stats(table, total)
        return

    # 토크나이저 로드
    use_userdic = args.userdic and not args.no_userdic
    tokenizer = load_mecab_tokenizer(
        use_legal_dict=not args.no_legal_dict,
        use_userdic=use_userdic,
    )

    if args.dry_run:
        dry_run(table, tokenizer)
        return

    # 실행 확인
    print(f"\n{total:,}행의 content_tokenized를 재생성합니다.")
    print("메모리 약 2.5GB 필요. 기존 FTS 인덱스도 재생성됩니다.")
    answer = input("계속하시겠습니까? [y/N] ").strip().lower()
    if answer != "y":
        print("취소되었습니다.")
        return

    update_table(db, table, tokenizer, total)


if __name__ == "__main__":
    main()
