"""
거리 계산 유틸리티

Haversine 공식을 사용한 두 좌표 간 거리 계산
"""

import math
from typing import Tuple

# 지구 반경 (킬로미터)
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    두 좌표 간의 거리 계산 (Haversine 공식)

    Args:
        lat1: 첫 번째 위치의 위도
        lon1: 첫 번째 위치의 경도
        lat2: 두 번째 위치의 위도
        lon2: 두 번째 위치의 경도

    Returns:
        두 지점 간의 거리 (킬로미터)
    """
    # 라디안 변환
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine 공식
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


def calculate_distance_km(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """
    두 좌표점 간의 거리 계산

    Args:
        point1: (위도, 경도) 튜플
        point2: (위도, 경도) 튜플

    Returns:
        두 지점 간의 거리 (킬로미터)
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    return haversine_distance(lat1, lon1, lat2, lon2)
