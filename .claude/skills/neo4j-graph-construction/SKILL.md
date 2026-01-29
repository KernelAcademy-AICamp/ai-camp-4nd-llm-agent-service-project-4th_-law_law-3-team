---
name: neo4j-graph-construction
description: Neo4j GraphDB 구축, 데이터 모델링(법령 계급 포함), LangGraph 에이전트 연동 가이드
---

# Neo4j GraphDB Construction & Agent Integration

이 스킬은 법률 데이터(판례, 법령)를 위한 Neo4j GraphDB를 구축하고, LangGraph 기반 에이전트에서 이를 활용하는 방법을 안내합니다.
특히 **법령 계급도(Statute Hierarchy)** 를 포함한 데이터 모델링과 대용량 데이터 마이그레이션, 그리고 에이전트가 그래프를 탐색하는 패턴을 중점적으로 다룹니다.

## 1. 환경 구성 (Environment Setup)

### 1.1 Docker Compose 설정
`docker-compose.yml` 파일에 Neo4j 5.x 컨테이너를 추가합니다. `APOC` 플러그인은 그래프 알고리즘 및 유틸리티 사용을 위해 필수적입니다.

```yaml
services:
  neo4j:
    image: neo4j:5.15.0  # 또는 호환되는 5.x 최신 버전
    container_name: neo4j-law-graph
    ports:
      - "7474:7474" # Browser UI
      - "7687:7687" # Bolt Protocol
    environment:
      - NEO4J_AUTH=neo4j/password  # 실제 운영 시 변경 권장
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=1G
      - NEO4J_dbms_memory_heap_max__size=2G
    volumes:
      - ./neo4j_data:/data
      - ./neo4j_logs:/logs
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 1.2 Python 의존성 (Dependencies)
`pyproject.toml` 또는 `requirements.txt`에 다음 라이브러리를 추가합니다.

```toml
# pyproject.toml
dependencies = [
    "neo4j>=5.15.0",
    "langchain-neo4j>=0.1.0",
    "langchain-community",
    "pydantic>=2.0"
]
```

## 2. 데이터 모델링 (Legal Graph Schema)

법률 도메인의 특성을 반영한 그래프 스키마입니다. **법령 계급(Hierarchy)** 구조 표현에 주의하세요.

### 2.1 Nodes (노드)

| Label | Description | 주요 속성 (Properties) |
| :--- | :--- | :--- |
| **Statute** | 법령 및 조문 | `name` (법령명), `article` (조문번호), `type` (법률/시행령/규칙), `content` (본문) |
| **Case** | 판례 | `case_number` (사건번호), `summary` (판결요지), `date` (선고일), `verdict` (주문) |
| **Lawyer** | 변호사 | `name`, `specialty` (전문분야) |

### 2.2 Relationships (엣지)

| Type | 방향 | 개수 | 설명 |
|------|------|------|------|
| `HIERARCHY_OF` | Statute → Statute | 3,624 | 법령 계급 (시행령→법률) |
| `CITES` | Case → Statute | 72,414 | 판례가 법령 인용 |
| `CITES_CASE` | Case → Case | 87,654 | 판례가 판례 인용 |
| `RELATED_TO` | Statute → Statute | 93 | 관련 법령 |

#### `HIERARCHY_OF` (법령 계급 관계)
- **방향**: `(Statute:Lower)-[:HIERARCHY_OF]->(Statute:Upper)`
- **의미**: 하위 법령이 상위 법령에 속함 (또는 근거함).
- **예시**: `(도로교통법 시행령)-[:HIERARCHY_OF]->(도로교통법)`

#### `CITES` (판례→법령 인용)
- **방향**: `(Case)-[:CITES]->(Statute)`
- **의미**: 판례가 법령을 참조조문으로 인용
- **데이터 소스**: 판례의 `참조조문` 필드

#### `CITES_CASE` (판례→판례 인용)
- **방향**: `(Case)-[:CITES_CASE]->(Case)`
- **의미**: 판례가 다른 판례를 참조판례로 인용
- **데이터 소스**: 판례의 `참조판례` 필드

#### `RELATED_TO` (관련 법령)
- **방향**: `(Statute)-[:RELATED_TO]->(Statute)`
- **의미**: 관련 법령 연결
- **데이터 소스**: 계급도의 `관련법령` 필드

## 3. 데이터 마이그레이션 (Migration Strategy)

PostgreSQL(관계형 DB) 데이터를 GraphDB로 옮길 때의 전략입니다.

### 3.1 Bulk Insert Pattern
단건 `CREATE` 대신 `UNWIND`를 사용하여 배치를 처리해야 속도가 빠릅니다.

**Cypher 예시 (법령 계급 적재):**
```cypher
// 파라미터 $batch는 [{'lower': '..', 'upper': '..'}, ...] 형태의 리스트
UNWIND $batch AS row
MERGE (l:Statute {name: row.lower})
MERGE (u:Statute {name: row.upper})
MERGE (l)-[:HIERARCHY_OF]->(u)
```

### 3.2 Constraints Creation
데이터 적재 전 제약조건을 먼저 생성하여 정합성을 보장합니다.

```python
def create_constraints(driver):
    queries = [
        "CREATE CONSTRAINT FOR (s:Statute) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT FOR (c:Case) REQUIRE c.case_number IS UNIQUE",
        "CREATE INDEX FOR (s:Statute) ON (s.type)"
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)
```

## 4. LangGraph Agent Integration

### 4.1 Schema for Graph Agent
GraphDB를 조회하는 에이전트는 스키마 정보를 정확히 알아야 합니다.

```python
graph_schema = """
Node properties:
- Statute: name, article, content
- Case: case_number, summary

Relationships:
- (:Statute)-[:HIERARCHY_OF]->(:Statute)
- (:Case)-[:CITES]->(:Statute)
"""
```

### 4.2 Hierarchy Traversal (계층 탐색 패턴)
특정 법령의 **최상위 근거법**을 찾거나, **하위 시행령**을 모두 찾는 질의 패턴입니다.

**하위법 -> 상위법 재귀 탐색:**
```cypher
MATCH path = (child:Statute {name: "교통안전법 시행규칙"})-[:HIERARCHY_OF*]->(root)
WHERE NOT (root)-[:HIERARCHY_OF]->()
RETURN path
```

**상위법 -> 하위법 전체 탐색:**
```cypher
MATCH path = (root:Statute {name: "형법"})<-[:HIERARCHY_OF*]-(child)
RETURN child.name, length(path) as depth
ORDER BY depth
```

## 5. 스크립트 사용법

### 5.1 그래프 구축
```bash
cd backend

# Neo4j 컨테이너 실행
docker compose up -d neo4j

# 그래프 데이터 구축 (전체)
uv run python scripts/build_graph.py

# 구축 결과:
# - Statute 노드: 5,572개
# - Case 노드: 65,107개
# - HIERARCHY_OF: 3,624개
# - CITES: 72,414개
# - CITES_CASE: 87,654개
# - RELATED_TO: 93개
```

### 5.2 검증
```bash
# CLI 검증
uv run python scripts/verify_graph.py

# Gradio UI 검증
uv run python scripts/verify_gradio.py
# → http://localhost:7860

# 테스트 실행 (27개)
uv run python tests/integration/test_neo4j_graph.py
```

### 5.3 Neo4j Browser
- URL: http://localhost:7474
- 로그인: neo4j / password (또는 .env 설정값)

## 6. 활용 시나리오

### 6.1 RAG 컨텍스트 보강
검색된 법령/판례의 관련 정보를 그래프에서 조회하여 컨텍스트 추가

```cypher
-- 법령 검색 시: 상위법, 하위법, 관련법령 컨텍스트
MATCH (s:Statute {name: $statute_name})
OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper)
OPTIONAL MATCH (lower)-[:HIERARCHY_OF]->(s)
OPTIONAL MATCH (s)-[:RELATED_TO]->(related)
RETURN upper, lower, related

-- 판례 검색 시: 인용 법령, 인용 판례 컨텍스트
MATCH (c:Case {case_number: $case_number})
OPTIONAL MATCH (c)-[:CITES]->(statute)
OPTIONAL MATCH (c)-[:CITES_CASE]->(cited_case)
RETURN statute, cited_case
```

### 6.2 판례 추천
```cypher
-- 같은 법령을 인용한 유사 판례
MATCH (c:Case {case_number: $case_number})-[:CITES]->(s:Statute)<-[:CITES]-(similar:Case)
WHERE c <> similar
RETURN similar, count(s) as common_statutes
ORDER BY common_statutes DESC
LIMIT 10

-- 특정 판례를 인용한 후속 판례
MATCH (citing:Case)-[:CITES_CASE]->(c:Case {case_number: $case_number})
RETURN citing
```

### 6.3 법령 탐색 UI
```cypher
-- 법령 계급 트리 (하위→상위)
MATCH path = (s:Statute {name: $name})-[:HIERARCHY_OF*]->(root)
RETURN [n in nodes(path) | n.name] as hierarchy_chain

-- 법령 계급 트리 (상위→하위)
MATCH path = (s:Statute {name: $name})<-[:HIERARCHY_OF*]-(child)
RETURN child.name, length(path) as depth
ORDER BY depth
```
