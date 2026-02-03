#!/usr/bin/env python3
"""
변호사 주소를 좌표로 변환하는 지오코딩 스크립트

사용법:
    # backend/.env에 KAKAO_REST_API_KEY 설정 후 실행
    python scripts/geocode_lawyers.py

    # 또는 인자로 전달
    python scripts/geocode_lawyers.py --api-key YOUR_API_KEY

    # 실패 항목만 재시도
    python scripts/geocode_lawyers.py --retry-failed

    # 현재 데이터 상태 확인
    python scripts/geocode_lawyers.py --stats

    # 입출력 경로 지정
    python scripts/geocode_lawyers.py --input path/to/input.json --output path/to/output.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# 프로젝트 경로
BACKEND_ROOT = Path(__file__).parent.parent  # backend/
PROJECT_ROOT = BACKEND_ROOT.parent  # law-3-team/
BACKEND_ENV = BACKEND_ROOT / ".env"

# 기본 파일 경로
DEFAULT_INPUT = PROJECT_ROOT / "all_lawyers.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "lawyers_with_coords.json"
FAILED_FILE = PROJECT_ROOT / "data" / "geocode_failed.json"

# API 설정
KAKAO_GEOCODE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
RATE_LIMIT = 10  # 초당 요청 수
BATCH_SIZE = 100  # 배치 크기


def load_env() -> None:
    """backend/.env 파일에서 환경변수 로드"""
    if BACKEND_ENV.exists():
        with open(BACKEND_ENV, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:  # 기존 환경변수 우선
                        os.environ[key] = value


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
            print("[ERROR] API 키가 유효하지 않습니다.")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] 지오코딩 실패 ({address}): {e}")

    return None


async def process_batch(
    client: httpx.AsyncClient,
    lawyers: list,
    api_key: str,
    address_cache: dict
) -> Tuple[List[dict], List[dict]]:
    """배치 단위로 지오코딩 처리"""
    success: List[dict] = []
    failed: List[dict] = []

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


async def main(
    api_key: str,
    input_path: Path,
    output_path: Path,
) -> None:
    """
    전체 지오코딩 실행

    Args:
        api_key: 카카오 REST API 키
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로
    """
    print(f"[INFO] 입력 파일: {input_path}")
    print(f"[INFO] 출력 파일: {output_path}")

    # 입력 파일 로드
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lawyers = data.get("lawyers", [])
    total = len(lawyers)
    print(f"[INFO] 총 {total}명의 변호사 데이터를 처리합니다.")

    # 주소 캐시 (동일 주소 중복 API 호출 방지)
    address_cache: Dict[str, dict] = {}

    # 결과 저장용
    all_success: List[dict] = []
    all_failed: List[dict] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, total, BATCH_SIZE):
            batch = lawyers[i:i + BATCH_SIZE]
            success, failed = await process_batch(
                client, batch, api_key, address_cache
            )

            all_success.extend(success)
            all_failed.extend(failed)

            processed = min(i + BATCH_SIZE, total)
            success_rate = len(all_success) / processed * 100
            print(
                f"[PROGRESS] {processed}/{total} "
                f"처리완료 (성공률: {success_rate:.1f}%)"
            )

    # 결과 저장
    output_data = {
        "metadata": {
            **data.get("metadata", {}),
            "geocoded_at": datetime.now().isoformat(),
            "total_geocoded": len(all_success),
            "total_failed": len(all_failed),
        },
        "lawyers": all_success + all_failed  # 실패한 것도 포함 (좌표만 null)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 실패 목록 별도 저장
    if all_failed:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(all_failed, f, ensure_ascii=False, indent=2)
        print(f"[WARN] {len(all_failed)}건 지오코딩 실패 → {FAILED_FILE}")

    print(f"[DONE] 완료! 결과: {output_path}")
    print(f"       성공: {len(all_success)}건, 실패: {len(all_failed)}건")
    print(f"       캐시된 주소: {len(address_cache)}개")


async def retry_failed(api_key: str, output_path: Path) -> None:
    """
    기존 출력 파일에서 좌표 없는 항목만 재시도

    Args:
        api_key: 카카오 REST API 키
        output_path: 기존 출력 JSON 파일 경로 (읽기 + 덮어쓰기)
    """
    if not output_path.exists():
        print(f"[ERROR] 출력 파일이 없습니다: {output_path}")
        print("        먼저 전체 지오코딩을 실행하세요.")
        sys.exit(1)

    with open(output_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    lawyers = data.get("lawyers", [])

    # 좌표 없는 항목 필터링
    needs_retry = [law for law in lawyers if law.get("latitude") is None]
    already_done = [law for law in lawyers if law.get("latitude") is not None]

    if not needs_retry:
        print("[INFO] 재시도할 항목이 없습니다. 모든 변호사에 좌표가 있습니다.")
        return

    print(f"[INFO] 출력 파일: {output_path}")
    print(f"[INFO] 기존 좌표 있음: {len(already_done)}명")
    print(f"[INFO] 재시도 대상: {len(needs_retry)}명")

    address_cache: Dict[str, dict] = {}
    retry_success: List[dict] = []
    retry_failed_list: List[dict] = []

    async with httpx.AsyncClient() as client:
        total = len(needs_retry)
        for i in range(0, total, BATCH_SIZE):
            batch = needs_retry[i:i + BATCH_SIZE]
            success, failed = await process_batch(
                client, batch, api_key, address_cache
            )

            retry_success.extend(success)
            retry_failed_list.extend(failed)

            processed = min(i + BATCH_SIZE, total)
            success_rate = (
                len(retry_success) / processed * 100 if processed > 0 else 0
            )
            print(
                f"[PROGRESS] {processed}/{total} "
                f"재시도 처리완료 (성공률: {success_rate:.1f}%)"
            )

    # 결과 병합: 기존 성공 + 재시도 성공 + 재시도 실패
    merged_lawyers = already_done + retry_success + retry_failed_list

    output_data = {
        "metadata": {
            **data.get("metadata", {}),
            "geocoded_at": datetime.now().isoformat(),
            "total_geocoded": len(already_done) + len(retry_success),
            "total_failed": len(retry_failed_list),
            "last_retry_at": datetime.now().isoformat(),
        },
        "lawyers": merged_lawyers,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 실패 목록 갱신
    if retry_failed_list:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(retry_failed_list, f, ensure_ascii=False, indent=2)
        print(f"[WARN] {len(retry_failed_list)}건 여전히 실패 → {FAILED_FILE}")
    elif FAILED_FILE.exists():
        FAILED_FILE.unlink()
        print("[INFO] 모든 항목 성공, 실패 목록 파일 삭제")

    print("[DONE] 재시도 완료!")
    print(
        f"       신규 성공: {len(retry_success)}건, "
        f"여전히 실패: {len(retry_failed_list)}건"
    )
    print(
        f"       전체: 좌표 있음 {len(already_done) + len(retry_success)}명 / "
        f"총 {len(merged_lawyers)}명"
    )


def show_stats(file_path: Path) -> None:
    """
    데이터 파일의 현재 상태를 출력

    Args:
        file_path: 확인할 JSON 파일 경로
    """
    if not file_path.exists():
        print(f"[ERROR] 파일이 없습니다: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lawyers = data.get("lawyers", [])
    metadata = data.get("metadata", {})
    total = len(lawyers)

    has_coords = sum(1 for law in lawyers if law.get("latitude") is not None)
    no_coords = total - has_coords
    has_specialties = sum(
        1 for law in lawyers
        if law.get("specialties") and len(law["specialties"]) > 0
    )

    coord_pct = (has_coords / total * 100) if total > 0 else 0
    no_coord_pct = (no_coords / total * 100) if total > 0 else 0
    spec_pct = (has_specialties / total * 100) if total > 0 else 0

    print(f"[INFO] 데이터 파일: {file_path}")
    print(f"[INFO] 총 변호사: {total:,}명")
    print(f"[INFO] 좌표 있음: {has_coords:,}명 ({coord_pct:.1f}%)")
    print(f"[INFO] 좌표 없음: {no_coords:,}명 ({no_coord_pct:.1f}%)")
    print(f"[INFO] 전문분야 있음: {has_specialties:,}명 ({spec_pct:.1f}%)")

    geocoded_at = metadata.get("geocoded_at")
    if geocoded_at:
        print(f"[INFO] 지오코딩 일시: {geocoded_at}")

    last_retry = metadata.get("last_retry_at")
    if last_retry:
        print(f"[INFO] 마지막 재시도: {last_retry}")


if __name__ == "__main__":
    load_env()

    parser = argparse.ArgumentParser(
        description="변호사 주소 지오코딩 (카카오 API)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("KAKAO_REST_API_KEY"),
        help="카카오 REST API 키 (또는 KAKAO_REST_API_KEY 환경변수)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 파일 경로 (기본: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 파일 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="기존 출력 파일에서 좌표 없는 항목만 재시도",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="현재 데이터 상태만 출력 (지오코딩 실행 안 함)",
    )

    args = parser.parse_args()

    # --stats: 상태만 출력하고 종료 (API 키 불필요)
    if args.stats:
        show_stats(args.output)
        sys.exit(0)

    # API 키 확인 (지오코딩 실행 시 필수)
    if not args.api_key:
        print("[ERROR] 카카오 REST API 키가 필요합니다.")
        print(
            "        --api-key 옵션 또는 "
            "KAKAO_REST_API_KEY 환경변수를 설정하세요."
        )
        print("\n        카카오 API 키 발급: https://developers.kakao.com")
        sys.exit(1)

    if args.retry_failed:
        asyncio.run(retry_failed(args.api_key, args.output))
    else:
        asyncio.run(main(args.api_key, args.input, args.output))
