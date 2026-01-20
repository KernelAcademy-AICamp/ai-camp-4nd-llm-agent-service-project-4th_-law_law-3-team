"""JSON 백업 파일을 PostgreSQL에 임포트하는 스크립트"""
import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from app.core.config import settings


def import_legal_documents(json_path: str, batch_size: int = 1000, force: bool = False):
    """JSON 파일에서 legal_documents 테이블로 데이터 임포트"""

    # JSON 파일 로드
    print(f"Loading JSON file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records to import: {len(data)}")

    # DB 연결
    engine = create_engine(str(settings.DATABASE_URL))

    with engine.connect() as conn:
        # 기존 데이터 확인
        result = conn.execute(text("SELECT COUNT(*) FROM legal_documents"))
        existing_count = result.scalar()
        print(f"Existing records in DB: {existing_count}")

        if existing_count > 0:
            if force:
                conn.execute(text("TRUNCATE legal_documents RESTART IDENTITY"))
                conn.commit()
                print("기존 데이터 삭제 완료")
            else:
                print("기존 데이터가 있습니다. --force 옵션으로 삭제 후 진행하세요.")
                return

        # 배치 단위로 INSERT
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            values = []
            for record in batch:
                # raw_data에 원본 데이터 저장
                raw_data = json.dumps(record, ensure_ascii=False)
                values.append({
                    "id": record.get("id"),
                    "doc_type": record.get("doc_type", "unknown"),
                    "serial_number": record.get("serial_number", ""),
                    "case_name": record.get("case_name"),
                    "case_number": record.get("case_number"),
                    "raw_data": raw_data,
                })

            # Bulk insert
            insert_sql = text("""
                INSERT INTO legal_documents (id, doc_type, serial_number, case_name, case_number, raw_data)
                VALUES (:id, :doc_type, :serial_number, :case_name, :case_number, CAST(:raw_data AS jsonb))
                ON CONFLICT (doc_type, serial_number) DO NOTHING
            """)

            conn.execute(insert_sql, values)
            conn.commit()

            total_inserted += len(batch)
            print(f"Progress: {total_inserted}/{len(data)} ({total_inserted * 100 // len(data)}%)")

        # ID 시퀀스 재설정
        conn.execute(text("""
            SELECT setval('legal_documents_id_seq', (SELECT MAX(id) FROM legal_documents))
        """))
        conn.commit()

        # 최종 확인
        result = conn.execute(text("SELECT COUNT(*) FROM legal_documents"))
        final_count = result.scalar()
        print(f"\nImport completed! Total records in DB: {final_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import JSON backup to PostgreSQL")
    parser.add_argument("--force", "-f", action="store_true", help="기존 데이터 삭제 후 진행")
    args = parser.parse_args()

    json_path = Path(__file__).parent.parent / "data" / "legal_documents_backup.json"
    import_legal_documents(str(json_path), force=args.force)
