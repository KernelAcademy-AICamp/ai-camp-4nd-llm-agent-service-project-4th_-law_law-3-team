"""
인구 데이터 CSV 파일을 JSON으로 변환하는 스크립트.

사용법:
    cd backend
    python scripts/update_population.py

입력 파일:
    - data/population_202512.csv: 현재 인구 (주민등록인구)
    - data/population_pred.csv: 추계인구

데이터 출처:
    KOSIS (https://kosis.kr)
    -> e지방지표(주제별) -> 인구
    -> 주민등록인구(시도/시/군/구) 또는 추계인구(시/군/구)
    -> 조회 조건 '합계'로 다운로드 (CSV UTF-8)

출력 파일:
    - data/population.json
"""

import csv
import json
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"

# 시도명 정규화 매핑
PROVINCE_NORMALIZE_MAP: dict[str, str] = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

# 시도 목록 (CSV에서 시도 구분용)
PROVINCE_NAMES = set(PROVINCE_NORMALIZE_MAP.keys())


def normalize_province(name: str) -> str:
    """시도명을 표준 형식으로 정규화."""
    return PROVINCE_NORMALIZE_MAP.get(name, name)


def parse_current_population(csv_path: Path) -> dict[str, int]:
    """
    현재 인구 CSV 파싱.

    Args:
        csv_path: population_202512.csv 경로

    Returns:
        {"서울 종로구": 137048, "서울 중구": 117805, ...}
    """
    result: dict[str, int] = {}
    current_province = ""

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 헤더 2줄 스킵, 전국 스킵
    for row in rows[3:]:
        if len(row) < 2:
            continue

        name = row[0].strip()
        population_str = row[1].strip().replace(",", "")

        if not population_str.isdigit():
            continue

        population = int(population_str)

        # 시도명인 경우
        if name in PROVINCE_NAMES:
            current_province = normalize_province(name)
            continue

        # 시군구인 경우
        if current_province:
            # 세종시는 특별 처리 (시군구 없이 시 자체가 기초자치단체)
            if current_province == "세종":
                region_name = "세종 세종시"
            else:
                region_name = f"{current_province} {name}"
            result[region_name] = population

    return result


def parse_prediction_population(csv_path: Path) -> dict[str, dict[str, int]]:
    """
    추계인구 CSV 파싱.

    Args:
        csv_path: population_pred.csv 경로

    Returns:
        {"서울 종로구": {"2030": 136848, "2035": 132763, "2040": 129636}, ...}
    """
    result: dict[str, dict[str, int]] = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 헤더 2줄 스킵
    for row in rows[2:]:
        if len(row) < 6:
            continue

        province = row[0].strip()
        district = row[1].strip()

        # "소계"는 시도 전체 인구이므로 스킵
        if district == "소계":
            continue

        # 시도명 정규화
        normalized_province = normalize_province(province)

        # 세종시 특별 처리
        if normalized_province == "세종":
            region_name = "세종 세종시"
        else:
            region_name = f"{normalized_province} {district}"

        # 추계인구 파싱
        try:
            pop_2030 = int(row[3].strip().replace(",", ""))
            pop_2035 = int(row[4].strip().replace(",", ""))
            pop_2040 = int(row[5].strip().replace(",", ""))
        except (ValueError, IndexError):
            continue

        result[region_name] = {
            "2030": pop_2030,
            "2035": pop_2035,
            "2040": pop_2040,
        }

    return result


def merge_population_data(
    current: dict[str, int],
    prediction: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """
    현재 인구와 추계인구 데이터 병합.

    Args:
        current: 현재 인구 데이터
        prediction: 추계인구 데이터

    Returns:
        병합된 데이터 (JSON 저장용)
    """
    result: dict[str, dict[str, int]] = {}

    # 모든 지역 수집 (현재 + 추계)
    all_regions = set(current.keys()) | set(prediction.keys())

    for region in sorted(all_regions):
        data: dict[str, int] = {}

        # 현재 인구
        if region in current:
            data["current"] = current[region]

        # 추계인구
        if region in prediction:
            for year, pop in prediction[region].items():
                data[year] = pop

        result[region] = data

    return result


def save_population_json(
    data: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    """JSON 파일로 저장."""
    output = {
        "meta": {
            "current_year": 2025,
            "current_month": 12,
            "source": "KOSIS e지방지표 (https://kosis.kr)",
            "source_current": "주민등록인구(시도/시/군/구)",
            "source_prediction": "추계인구(시/군/구)",
            "prediction_years": [2030, 2035, 2040],
        },
        "data": data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main() -> None:
    """메인 함수."""
    current_csv = DATA_DIR / "population_202512.csv"
    pred_csv = DATA_DIR / "population_pred.csv"
    output_json = OUTPUT_DIR / "population.json"

    print(f"현재 인구 CSV: {current_csv}")
    print(f"추계인구 CSV: {pred_csv}")
    print(f"출력 JSON: {output_json}")

    # 파일 존재 확인
    if not current_csv.exists():
        print(f"Error: {current_csv} 파일이 없습니다.")
        return
    if not pred_csv.exists():
        print(f"Error: {pred_csv} 파일이 없습니다.")
        return

    # 파싱
    print("\n현재 인구 데이터 파싱 중...")
    current = parse_current_population(current_csv)
    print(f"  - {len(current)}개 지역 파싱 완료")

    print("추계인구 데이터 파싱 중...")
    prediction = parse_prediction_population(pred_csv)
    print(f"  - {len(prediction)}개 지역 파싱 완료")

    # 병합
    print("\n데이터 병합 중...")
    merged = merge_population_data(current, prediction)
    print(f"  - {len(merged)}개 지역 병합 완료")

    # 저장
    print(f"\nJSON 저장 중: {output_json}")
    save_population_json(merged, output_json)
    print("완료!")

    # 샘플 출력
    print("\n=== 샘플 데이터 (처음 5개) ===")
    for i, (region, data) in enumerate(list(merged.items())[:5]):
        print(f"  {region}: {data}")


if __name__ == "__main__":
    main()
