#!/usr/bin/env python3
"""
LanceDB 검색 테스트 스크립트

LanceDB에 저장된 법령/판례 데이터가 정상적으로 검색되는지 확인합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lancedb


def test_lancedb_search():
    """LanceDB 검색 테스트"""
    print("=" * 60)
    print("LanceDB 검색 테스트")
    print("=" * 60)

    # 1. LanceDB 연결
    db_path = PROJECT_ROOT / "lancedb_data"

    if not db_path.exists():
        print(f"[ERROR] LanceDB 데이터 없음: {db_path}")
        return False

    print(f"\n[INFO] DB 경로: {db_path}")
    db = lancedb.connect(str(db_path))

    # 2. 테이블 확인
    tables = db.table_names()
    print(f"[INFO] 테이블 목록: {tables}")

    if "legal_chunks" not in tables:
        print("[ERROR] legal_chunks 테이블 없음")
        return False

    table = db.open_table("legal_chunks")
    total_count = len(table)
    print(f"[INFO] 총 레코드 수: {total_count:,}")

    if total_count == 0:
        print("[ERROR] 데이터가 없습니다")
        return False

    # 3. 데이터 유형별 통계
    print("\n" + "-" * 40)
    print("데이터 유형별 통계")
    print("-" * 40)

    try:
        law_df = table.search().where("data_type = '법령'").limit(1000000).to_pandas()
        law_count = len(law_df)
        print(f"  - 법령: {law_count:,}")
    except Exception as e:
        print(f"  - 법령: 조회 실패 ({e})")
        law_count = 0

    try:
        prec_df = table.search().where("data_type = '판례'").limit(1000000).to_pandas()
        prec_count = len(prec_df)
        print(f"  - 판례: {prec_count:,}")
    except Exception as e:
        print(f"  - 판례: 조회 실패 ({e})")
        prec_count = 0

    # 4. 샘플 데이터 확인
    print("\n" + "-" * 40)
    print("샘플 데이터 (처음 5개)")
    print("-" * 40)

    sample_df = table.search().limit(5).to_pandas()

    for idx, row in sample_df.iterrows():
        print(f"\n[{idx + 1}] ID: {row.get('id', 'N/A')}")
        print(f"    유형: {row.get('data_type', 'N/A')}")
        print(f"    제목: {row.get('title', 'N/A')[:50]}...")
        print(f"    출처: {row.get('source_name', 'N/A')}")
        print(f"    날짜: {row.get('date', 'N/A')}")
        content = row.get('content', '')
        if content:
            print(f"    내용: {content[:100]}...")

    # 5. 벡터 검색 테스트
    print("\n" + "-" * 40)
    print("벡터 검색 테스트")
    print("-" * 40)

    try:
        from sentence_transformers import SentenceTransformer

        print("[INFO] 임베딩 모델 로딩...")
        model = SentenceTransformer("nlpai-lab/KURE-v1", trust_remote_code=True)

        # 테스트 쿼리들
        test_queries = [
            "손해배상 청구",
            "임대차 계약 해지",
            "민법 불법행위",
        ]

        for query in test_queries:
            print(f"\n검색어: '{query}'")
            query_vector = model.encode(query).tolist()

            results = table.search(query_vector).metric("cosine").limit(3).to_pandas()

            print(f"  결과 ({len(results)}건):")
            for _, row in results.iterrows():
                similarity = 1 - row.get('_distance', 0)
                data_type = row.get('data_type', 'N/A')
                title = row.get('title', 'N/A')[:40]
                print(f"    [{similarity:.4f}] [{data_type}] {title}")

        print("\n[SUCCESS] 벡터 검색 정상 작동!")

    except ImportError:
        print("[WARN] sentence-transformers 미설치, 벡터 검색 스킵")
        print("       설치: pip install sentence-transformers")
    except Exception as e:
        print(f"[ERROR] 벡터 검색 실패: {e}")
        return False

    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"  총 레코드: {total_count:,}")
    print(f"  법령: {law_count:,}")
    print(f"  판례: {prec_count:,}")
    print("  상태: OK")

    return True


if __name__ == "__main__":
    success = test_lancedb_search()
    sys.exit(0 if success else 1)
