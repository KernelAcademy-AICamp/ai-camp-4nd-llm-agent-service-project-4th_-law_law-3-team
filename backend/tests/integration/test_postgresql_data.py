#!/usr/bin/env python3
"""
PostgreSQL 데이터 확인 스크립트

법령/판례 원본 데이터가 PostgreSQL에 정상적으로 저장되었는지 확인합니다.
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.database import async_session_factory
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument


async def test_postgresql_data():
    """PostgreSQL 데이터 확인"""
    print("=" * 60)
    print("PostgreSQL 데이터 확인")
    print("=" * 60)

    async with async_session_factory() as session:
        # 1. 연결 테스트
        try:
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"\n[INFO] PostgreSQL 연결 성공")
            print(f"[INFO] 버전: {version[:50]}...")
        except Exception as e:
            print(f"[ERROR] PostgreSQL 연결 실패: {e}")
            return False

        # 2. 테이블 존재 확인
        print("\n" + "-" * 40)
        print("테이블 확인")
        print("-" * 40)

        tables_query = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        result = await session.execute(tables_query)
        tables = [row[0] for row in result.fetchall()]
        print(f"[INFO] 테이블 목록: {tables}")

        has_law = "law_documents" in tables
        has_precedent = "precedent_documents" in tables

        if not has_law:
            print("[WARN] law_documents 테이블 없음")
        if not has_precedent:
            print("[WARN] precedent_documents 테이블 없음")

        # 3. 법령 데이터 확인
        print("\n" + "-" * 40)
        print("법령 데이터 (law_documents)")
        print("-" * 40)

        if has_law:
            # 총 개수
            count_result = await session.execute(
                select(func.count()).select_from(LawDocument)
            )
            law_count = count_result.scalar()
            print(f"  총 건수: {law_count:,}")

            if law_count > 0:
                # 법령 유형별 통계
                type_stats = await session.execute(
                    select(
                        LawDocument.law_type,
                        func.count(LawDocument.id)
                    ).group_by(LawDocument.law_type)
                )
                print("  유형별 통계:")
                for row in type_stats:
                    print(f"    - {row[0] or '미분류'}: {row[1]:,}")

                # 샘플 데이터
                sample = await session.execute(
                    select(LawDocument).limit(3)
                )
                print("\n  샘플 데이터:")
                for law in sample.scalars():
                    print(f"    [{law.law_id}] {law.law_name[:30]}...")
                    print(f"      유형: {law.law_type}, 부처: {law.ministry}")
                    if law.content:
                        print(f"      내용: {law.content[:80]}...")
        else:
            law_count = 0

        # 4. 판례 데이터 확인
        print("\n" + "-" * 40)
        print("판례 데이터 (precedent_documents)")
        print("-" * 40)

        if has_precedent:
            # 총 개수
            count_result = await session.execute(
                select(func.count()).select_from(PrecedentDocument)
            )
            precedent_count = count_result.scalar()
            print(f"  총 건수: {precedent_count:,}")

            if precedent_count > 0:
                # 법원별 통계
                court_stats = await session.execute(
                    select(
                        PrecedentDocument.court_name,
                        func.count(PrecedentDocument.id)
                    ).group_by(PrecedentDocument.court_name)
                    .order_by(func.count(PrecedentDocument.id).desc())
                    .limit(5)
                )
                print("  법원별 통계 (상위 5개):")
                for row in court_stats:
                    print(f"    - {row[0] or '미분류'}: {row[1]:,}")

                # 사건종류별 통계
                case_type_stats = await session.execute(
                    select(
                        PrecedentDocument.case_type,
                        func.count(PrecedentDocument.id)
                    ).group_by(PrecedentDocument.case_type)
                    .order_by(func.count(PrecedentDocument.id).desc())
                    .limit(5)
                )
                print("  사건종류별 통계 (상위 5개):")
                for row in case_type_stats:
                    print(f"    - {row[0] or '미분류'}: {row[1]:,}")

                # 샘플 데이터
                sample = await session.execute(
                    select(PrecedentDocument).limit(3)
                )
                print("\n  샘플 데이터:")
                for prec in sample.scalars():
                    print(f"    [{prec.serial_number}] {prec.case_name[:30] if prec.case_name else 'N/A'}...")
                    print(f"      사건번호: {prec.case_number}, 법원: {prec.court_name}")
                    print(f"      선고일: {prec.decision_date}")
                    if prec.summary:
                        print(f"      판시사항: {prec.summary[:60]}...")
                    if prec.ruling:
                        print(f"      주문: {prec.ruling[:60]}...")
        else:
            precedent_count = 0

        # 5. LanceDB와 비교
        print("\n" + "-" * 40)
        print("LanceDB vs PostgreSQL 비교")
        print("-" * 40)

        try:
            import lancedb
            db_path = PROJECT_ROOT / "lancedb_data"
            if db_path.exists():
                db = lancedb.connect(str(db_path))
                if "legal_chunks" in db.table_names():
                    table = db.open_table("legal_chunks")

                    lance_law = len(table.search().where("data_type = '법령'").limit(1000000).to_pandas())
                    lance_prec = len(table.search().where("data_type = '판례'").limit(1000000).to_pandas())

                    print(f"  LanceDB 법령 청크: {lance_law:,}")
                    print(f"  LanceDB 판례 청크: {lance_prec:,}")
                    print(f"  PostgreSQL 법령 원본: {law_count:,}")
                    print(f"  PostgreSQL 판례 원본: {precedent_count:,}")

                    if law_count > 0 and lance_law > 0:
                        ratio = lance_law / law_count
                        print(f"  법령 청크/원본 비율: {ratio:.1f}")

                    if precedent_count > 0 and lance_prec > 0:
                        ratio = lance_prec / precedent_count
                        print(f"  판례 청크/원본 비율: {ratio:.1f}")
        except Exception as e:
            print(f"  [WARN] LanceDB 비교 실패: {e}")

        # 6. 결과 요약
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        print(f"  법령 원본: {law_count:,}")
        print(f"  판례 원본: {precedent_count:,}")

        if law_count > 0 and precedent_count > 0:
            print("  상태: OK")
            return True
        elif law_count == 0 and precedent_count == 0:
            print("  상태: 데이터 없음")
            return False
        else:
            print("  상태: 부분 저장")
            return True


if __name__ == "__main__":
    success = asyncio.run(test_postgresql_data())
    sys.exit(0 if success else 1)
