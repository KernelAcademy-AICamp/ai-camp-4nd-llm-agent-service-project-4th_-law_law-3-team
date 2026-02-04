# Spatial Query Patterns Skill

PostGIS 없이 PostgreSQL에서 위치 기반 검색을 수행하는 패턴 가이드.

## 1. Bounding Box 계산

```python
def get_bounding_box(lat: float, lng: float, radius_km: float) -> tuple:
    """반경 기준 바운딩 박스 계산."""
    lat_delta = radius_km / 111.0  # 위도 1도 ≈ 111km
    lng_delta = radius_km / (111.0 * cos(radians(lat)))
    return (lat - lat_delta, lat + lat_delta, lng - lng_delta, lng + lng_delta)
```

### SQL 적용
```sql
WHERE latitude BETWEEN :min_lat AND :max_lat
  AND longitude BETWEEN :min_lng AND :max_lng
```

B-tree 복합 인덱스 `(latitude, longitude)`로 빠른 1차 필터링.

## 2. Haversine SQL 표현식

PostGIS 없이 SQL에서 두 좌표 간 거리(km) 계산:

```sql
6371 * acos(
    cos(radians(:lat)) * cos(radians(latitude)) *
    cos(radians(longitude) - radians(:lng)) +
    sin(radians(:lat)) * sin(radians(latitude))
)
```

### SQLAlchemy 표현식
```python
from sqlalchemy import func, literal_column

distance_expr = (
    literal_column(str(EARTH_RADIUS_KM))
    * func.acos(
        func.cos(func.radians(literal_column(str(lat))))
        * func.cos(func.radians(Model.latitude))
        * func.cos(func.radians(Model.longitude) - func.radians(literal_column(str(lng))))
        + func.sin(func.radians(literal_column(str(lat))))
        * func.sin(func.radians(Model.latitude))
    )
)
```

## 3. Grid-based 클러스터링

```sql
SELECT
    ROUND(latitude / :grid_size) * :grid_size AS grid_lat,
    ROUND(longitude / :grid_size) * :grid_size AS grid_lng,
    COUNT(*) AS count
FROM lawyers
WHERE latitude BETWEEN :min_lat AND :max_lat
  AND longitude BETWEEN :min_lng AND :max_lng
GROUP BY grid_lat, grid_lng
```

### 줌 레벨 → 그리드 크기 매핑
| 줌 레벨 | 그리드 크기 | 대략적 거리 |
|---------|-----------|-----------|
| 5 | 0.1 | ~10km |
| 8 | 0.03 | ~3km |
| 10 | 0.01 | ~1km |
| 12 | 0.003 | ~300m |

## 4. 좌표 인덱싱 전략

### 데이터 규모별 권장 전략

| 데이터 규모 | 전략 | 이유 |
|-----------|------|------|
| ~20K건 | B-tree (latitude, longitude) | 단순하고 충분히 빠름 (~5ms) |
| ~100K건 | B-tree + Bounding Box | 1차 필터로 대부분 제거 |
| 1M건+ | PostGIS + GiST | 공간 인덱스 필요 |

### 현재 프로젝트 (17K건)
```python
# B-tree 복합 인덱스로 충분
Index("idx_lawyers_coords", "latitude", "longitude")
```

## 5. 2단계 검색 패턴 (Bounding Box + Haversine)

```python
async def find_nearby(db, lat, lng, radius_m):
    radius_km = radius_m / 1000
    min_lat, max_lat, min_lng, max_lng = get_bounding_box(lat, lng, radius_km)

    distance = haversine_sql(lat, lng)

    query = (
        select(Model, distance.label("distance"))
        .where(
            Model.latitude.between(min_lat, max_lat),   # 1차: Bounding Box (인덱스)
            Model.longitude.between(min_lng, max_lng),
        )
        .where(distance <= radius_km)                    # 2차: 정확한 거리 (CPU)
        .order_by("distance")
    )
```

### 성능
- 1차 필터(B-tree): ~1ms
- 2차 필터(Haversine): ~3ms
- 전체: ~5ms (17K건 기준)

## 6. 반경 확장 검색 패턴

검색 결과가 없으면 반경을 단계적으로 확장:

```python
# 1단계: 기본 반경 (3km)
results = await find_nearby_db(db, lat, lng, radius_m=3000)

# 2단계: 확장 반경 (10km)
if not results:
    results = await find_nearby_db(db, lat, lng, radius_m=10000)

# 3단계: 전체 검색 (반경 무제한)
if not results:
    results = await search_db(db, category=category_id)
```
