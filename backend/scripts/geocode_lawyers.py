#!/usr/bin/env python3
"""
변호사 주소를 좌표로 변환하는 지오코딩 스크립트

사용법:
    # backend/.env에 KAKAO_REST_API_KEY 설정 후 실행
    python scripts/geocode_lawyers.py

    # 또는 인자로 전달
    python scripts/geocode_lawyers.py --api-key YOUR_API_KEY
"""

import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import httpx
from collections import defaultdict

# 프로젝트 경로
BACKEND_ROOT = Path(__file__).parent.parent  # backend/
PROJECT_ROOT = BACKEND_ROOT.parent  # law-3-team/
BACKEND_ENV = BACKEND_ROOT / ".env"

# backend/.env 파일에서 환경변수 로드
def load_env():
    if BACKEND_ENV.exists():
        with open(BACKEND_ENV, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:  # 기존 환경변수 우선
                        os.environ[key] = value

load_env()
INPUT_FILE = PROJECT_ROOT / "all_lawyers.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "lawyers_with_coords.json"
FAILED_FILE = PROJECT_ROOT / "data" / "geocode_failed.json"

# API 설정
KAKAO_GEOCODE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
RATE_LIMIT = 10  # 초당 요청 수
BATCH_SIZE = 100  # 배치 크기


async def geocode_address(
    client: httpx.AsyncClient,
    address: str,
    api_key: str
) -> Optional[dict]:
    """카카오 지오코딩 API로 주소를 좌표로 변환"""
    if not address:
        return None

    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}

    try:
        response = await client.get(
            KAKAO_GEOCODE_URL,
            headers=headers,
            params=params,
            timeout=10.0
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("documents"):
                doc = data["documents"][0]
                return {
                    "latitude": float(doc["y"]),
                    "longitude": float(doc["x"]),
                    "address_type": doc.get("address_type"),
                }
        elif response.status_code == 401:
            print(f"[ERROR] API 키가 유효하지 않습니다.")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] 지오코딩 실패 ({address}): {e}")

    return None


async def process_batch(
    client: httpx.AsyncClient,
    lawyers: list,
    api_key: str,
    address_cache: dict
) -> tuple[list, list]:
    """배치 단위로 지오코딩 처리"""
    success = []
    failed = []

    for lawyer in lawyers:
        address = lawyer.get("address")

        # 캐시 확인 (동일 주소 중복 방지)
        if address and address in address_cache:
            coords = address_cache[address]
            lawyer["latitude"] = coords["latitude"]
            lawyer["longitude"] = coords["longitude"]
            success.append(lawyer)
            continue

        # 지오코딩 실행
        coords = await geocode_address(client, address, api_key)

        if coords:
            lawyer["latitude"] = coords["latitude"]
            lawyer["longitude"] = coords["longitude"]
            address_cache[address] = coords
            success.append(lawyer)
        else:
            lawyer["latitude"] = None
            lawyer["longitude"] = None
            failed.append(lawyer)

        # Rate limiting
        await asyncio.sleep(1 / RATE_LIMIT)

    return success, failed


async def main(api_key: str):
    """메인 함수"""
    print(f"[INFO] 입력 파일: {INPUT_FILE}")
    print(f"[INFO] 출력 파일: {OUTPUT_FILE}")

    # 입력 파일 로드
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    lawyers = data.get("lawyers", [])
    total = len(lawyers)
    print(f"[INFO] 총 {total}명의 변호사 데이터를 처리합니다.")

    # 주소 캐시 (동일 주소 중복 API 호출 방지)
    address_cache = {}

    # 결과 저장용
    all_success = []
    all_failed = []

    async with httpx.AsyncClient() as client:
        for i in range(0, total, BATCH_SIZE):
            batch = lawyers[i:i + BATCH_SIZE]
            success, failed = await process_batch(client, batch, api_key, address_cache)

            all_success.extend(success)
            all_failed.extend(failed)

            processed = min(i + BATCH_SIZE, total)
            success_rate = len(all_success) / processed * 100
            print(f"[PROGRESS] {processed}/{total} 처리완료 (성공률: {success_rate:.1f}%)")

    # 결과 저장
    output_data = {
        "metadata": {
            **data.get("metadata", {}),
            "geocoded_at": __import__("datetime").datetime.now().isoformat(),
            "total_geocoded": len(all_success),
            "total_failed": len(all_failed),
        },
        "lawyers": all_success + all_failed  # 실패한 것도 포함 (좌표만 null)
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 실패 목록 별도 저장
    if all_failed:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"[WARN] {len(all_failed)}건 지오코딩 실패 → {FAILED_FILE}")

    print(f"[DONE] 완료! 결과: {OUTPUT_FILE}")
    print(f"       성공: {len(all_success)}건, 실패: {len(all_failed)}건")
    print(f"       캐시된 주소: {len(address_cache)}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="변호사 주소 지오코딩")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KAKAO_REST_API_KEY"),
        help="카카오 REST API 키 (또는 KAKAO_REST_API_KEY 환경변수)"
    )
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] 카카오 REST API 키가 필요합니다.")
        print("        --api-key 옵션 또는 KAKAO_REST_API_KEY 환경변수를 설정하세요.")
        print("\n        카카오 API 키 발급: https://developers.kakao.com")
        sys.exit(1)

    asyncio.run(main(args.api_key))
