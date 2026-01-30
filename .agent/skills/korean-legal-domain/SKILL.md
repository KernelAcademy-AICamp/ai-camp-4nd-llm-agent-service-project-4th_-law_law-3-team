# Korean Legal Domain Skill

## Overview
한국 법령/판례 RAG 시스템 개발을 위한 도메인 지식과 데이터 구조 가이드입니다.

---

## 1. 법령 데이터 구조 (법제처 OpenAPI 기준)

### 법령 XML 계층 구조
```
법령 (법령키)
├── 기본정보
│   ├── 법령ID (고유 식별자)
│   ├── 법종구분 (법률/대통령령/총리령/부령)
│   ├── 법령명_한글
│   ├── 법령명_한자
│   ├── 법령명약칭
│   ├── 공포일자
│   ├── 공포번호
│   ├── 시행일자
│   ├── 소관부처 (소관부처코드)
│   ├── 제개정구분 (제정/개정/폐지)
│   └── 연락부서 정보
│
├── 조문
│   └── 조문단위 (조문키)
│       ├── 조문번호 (예: 750)
│       ├── 조문가지번호 (예: 의2 → 750조의2)
│       ├── 조문제목
│       ├── 조문내용
│       ├── 조문시행일자
│       ├── 조문제개정유형
│       ├── 조문참고자료
│       └── 항
│           ├── 항번호
│           ├── 항내용
│           └── 호
│               ├── 호번호
│               ├── 호내용
│               └── 목
│                   ├── 목번호
│                   └── 목내용
│
├── 부칙
│   └── 부칙단위 (부칙키)
│       ├── 부칙공포일자
│       ├── 부칙공포번호
│       └── 부칙내용
│
├── 별표
│   └── 별표단위 (별표키)
│       ├── 별표번호
│       ├── 별표제목
│       ├── 별표시행일자
│       └── 별표내용/파일링크
│
├── 개정문
│   └── 개정문내용
│
└── 제개정이유
    └── 제개정이유내용
```

### 법종 구분 코드
| 코드 | 법종 | 설명 |
|------|------|------|
| 법률 | 법률 | 국회에서 제정 |
| 대통령령 | 시행령 | 대통령이 발하는 명령 |
| 총리령 | 시행규칙 | 국무총리가 발하는 명령 |
| 부령 | 시행규칙 | 각부 장관이 발하는 명령 |
| 조약 | 조약 | 국제법 규범 |
| 자치법규 | 조례/규칙 | 지방자치단체 |

### 조문 인용 표기 패턴
```
[법령명] 제[조번호]조 (제목)
[법령명] 제[조번호]조의[가지번호]
[법령명] 제[조번호]조 제[항번호]항
[법령명] 제[조번호]조 제[항번호]항 제[호번호]호
[법령명] 제[조번호]조 제[항번호]항 제[호번호]호 [목번호]목

예시:
- 민법 제750조 (불법행위의 내용)
- 형법 제250조 제1항
- 상법 제382조의3 제2항 제1호
- 근로기준법 제2조 제1항 제1호 가목
```

---

## 2. 판례 데이터 구조

### 판례 기본 필드
```yaml
판례:
  판례일련번호: 고유 식별자
  사건번호: "2023다12345"
  사건명: "손해배상(기)"
  선고일자: "20231215"
  법원명: "대법원"
  사건종류명: "민사"
  사건종류코드: "400102"
  판결유형: "판결"
  선고: "선고"
  판시사항: 핵심 법리
  판결요지: 판결 요약
  참조조문: 인용된 법령
  참조판례: 인용된 다른 판례
  판례상세링크: URL
```

### 사건번호 해석
```
2023다12345
│   │  └─ 일련번호
│   └─── 사건 부호 (다: 민사상고)
└────── 접수년도

주요 사건 부호:
- 다: 민사 상고
- 도: 형사 상고
- 두: 민사 재항고
- 모: 형사 재항고
- 가합: 민사 합의부
- 고합: 형사 합의부
- 헌가: 위헌법률심판
- 헌바: 헌법소원(위헌심사형)
- 헌마: 헌법소원(권리구제형)
```

---

## 3. 참조 관계 유형

### 3.1 법령 → 법령 참조
```
유형:
├── 위임 참조: "~에서 정하는 바에 따라" (상위법 → 하위법)
├── 준용 참조: "~를 준용한다" (A법 규정을 B법에 적용)
├── 정의 참조: "제2조에서 정의한 바와 같이"
├── 적용배제: "~의 규정은 적용하지 아니한다"
└── 벌칙 참조: "~를 위반한 자는 처벌한다"
```

#### 참조 패턴 정규식
```python
patterns = {
    "direct_ref": r"제(\d+)조(의\d+)?(\s*제(\d+)항)?(\s*제(\d+)호)?",
    "same_law_ref": r"(이\s*법|동법|본법)\s*제(\d+)조",
    "other_law_ref": r"「([^」]+)」\s*제(\d+)조",
    "delegation": r"(대통령령|총리령|부령)으로\s*정하",
    "application": r"(준용|적용)한다",
}
```

### 3.2 판례 → 법령 참조
```
패턴:
├── 해석 판례: 특정 조문의 의미 해석
├── 위헌 판례: 해당 조문 위헌 결정
├── 합헌 판례: 해당 조문 합헌 결정
└── 적용 판례: 조문을 사실관계에 적용
```

### 3.3 판례 → 판례 참조
```
패턴:
├── 선례 인용: 동일 법리 적용
├── 판례 변경: 기존 판례 변경
├── 판례 보충: 기존 법리 보충/확장
└── 반대 해석: 기존 판례와 구분
```

---

## 4. Graph DB 스키마 설계

### 노드 (Nodes)
```cypher
// 법령 노드
(:Law {
  law_id: STRING,           // 법령ID
  name: STRING,             // 법령명
  name_abbr: STRING,        // 약칭
  law_type: STRING,         // 법종구분
  enforcement_date: DATE,   // 시행일자
  ministry: STRING,         // 소관부처
  status: STRING            // 현행/폐지
})

// 조문 노드
(:Article {
  article_id: STRING,       // 법령ID_조문키
  law_id: STRING,           // 소속 법령
  article_num: STRING,      // 조문번호 (예: "750", "382의3")
  title: STRING,            // 조문제목
  content: TEXT,            // 조문내용
  enforcement_date: DATE,   // 시행일자
  revision_type: STRING     // 제개정유형
})

// 항 노드 (필요시)
(:Paragraph {
  paragraph_id: STRING,
  article_id: STRING,
  para_num: INTEGER,
  content: TEXT
})

// 판례 노드
(:Case {
  case_id: STRING,          // 판례일련번호
  case_number: STRING,      // 사건번호
  case_name: STRING,        // 사건명
  ruling_date: DATE,        // 선고일자
  court: STRING,            // 법원명
  case_type: STRING,        // 사건종류
  summary: TEXT,            // 판결요지
  holding: TEXT             // 판시사항
})
```

### 관계 (Relationships)
```cypher
// 법령 구조 관계
(:Law)-[:CONTAINS]->(:Article)
(:Article)-[:CONTAINS]->(:Paragraph)

// 법령 계층 관계
(:Law)-[:DELEGATES_TO {type: "시행령"}]->(:Law)
(:Law)-[:AMENDS {date: DATE}]->(:Law)
(:Law)-[:REPEALS {date: DATE}]->(:Law)

// 조문 간 참조
(:Article)-[:REFERENCES {
  ref_type: "위임|준용|정의|적용배제",
  context: STRING
}]->(:Article)

// 판례-법령 관계
(:Case)-[:INTERPRETS {
  interpretation_type: "해석|위헌|합헌|적용"
}]->(:Article)

// 판례 간 관계
(:Case)-[:CITES {
  cite_type: "선례인용|변경|보충|구분"
}]->(:Case)
```

### 예시 Cypher 쿼리

```cypher
// 1. 특정 조문을 인용한 모든 판례 찾기
MATCH (c:Case)-[:INTERPRETS]->(a:Article)
WHERE a.law_id = "민법" AND a.article_num = "750"
RETURN c.case_number, c.ruling_date, c.summary
ORDER BY c.ruling_date DESC

// 2. 특정 법률의 위임 관계 체인
MATCH path = (parent:Law)-[:DELEGATES_TO*1..3]->(child:Law)
WHERE parent.name = "민법"
RETURN path

// 3. 판례 인용 네트워크 (2단계)
MATCH path = (c1:Case)-[:CITES*1..2]->(c2:Case)
WHERE c1.case_number = "2023다12345"
RETURN path

// 4. 특정 조문과 관련된 모든 참조 관계
MATCH (a:Article {law_id: "민법", article_num: "750"})
OPTIONAL MATCH (a)-[r1:REFERENCES]->(ref:Article)
OPTIONAL MATCH (c:Case)-[r2:INTERPRETS]->(a)
RETURN a, collect(distinct ref), collect(distinct c)
```

---

## 5. Graph DB 옵션 비교

### 예산 제약 하 추천 순위

| 순위 | DB | 월 예상 비용 | 장점 | 단점 |
|------|-----|-------------|------|------|
| 1 | **FalkorDB** | $0 (self-hosted) | 빠른 성능, Cypher 지원, GraphRAG 최적화 | Redis 의존성 |
| 2 | **PostgreSQL + Apache AGE** | $0 (기존 인프라 활용) | SQL + Cypher, 익숙한 PostgreSQL | 순수 Graph DB보다 느림 |
| 3 | **Neo4j Community** | $0 (self-hosted) | 성숙한 생태계, 풍부한 문서 | 클러스터링 미지원 |

### FalkorDB 설치 (OCI 환경)
```bash
# Redis 설치
sudo apt-get update
sudo apt-get install redis-server

# FalkorDB 모듈 설치
git clone https://github.com/FalkorDB/FalkorDB.git
cd FalkorDB
make

# Redis에 FalkorDB 모듈 로드
redis-server --loadmodule ./bin/linux-x64-release/falkordb.so
```

### PostgreSQL + AGE 설치
```bash
# PostgreSQL 확장 설치
CREATE EXTENSION age;

# 그래프 생성
SELECT create_graph('legal_graph');

# Cypher 쿼리 실행
SELECT * FROM cypher('legal_graph', $$
  MATCH (a:Article)-[:REFERENCES]->(b:Article)
  RETURN a.article_num, b.article_num
$$) as (from_article agtype, to_article agtype);
```

---

## 6. 청킹 전략 가이드

### 법령 특화 청킹 옵션

#### Option A: 조문 단위 청킹 (권장)
```python
def chunk_by_article(law_xml):
    """각 조문을 하나의 청크로"""
    chunks = []
    for article in law_xml.findall(".//조문단위"):
        chunk = {
            "id": f"{law_id}_{article.find('조문키').text}",
            "content": extract_article_text(article),
            "metadata": {
                "law_name": law_name,
                "article_num": article.find("조문번호").text,
                "article_title": article.find("조문제목").text,
                "enforcement_date": article.find("조문시행일자").text
            }
        }
        chunks.append(chunk)
    return chunks
```

장점: 법적 단위 보존, 참조 관계 추적 용이
단점: 긴 조문은 토큰 제한 초과 가능

#### Option B: 항 단위 청킹
```python
def chunk_by_paragraph(law_xml):
    """각 항을 하나의 청크로, 조문 컨텍스트 포함"""
    # 조문 제목 + 항 내용으로 청크 구성
```

장점: 더 세밀한 검색, 토큰 제한 준수
단점: 컨텍스트 손실 가능

#### Option C: 하이브리드 청킹
```python
def hybrid_chunk(law_xml, max_tokens=500):
    """조문 단위 기본, 긴 조문만 항 단위로 분할"""
    chunks = []
    for article in law_xml.findall(".//조문단위"):
        article_text = extract_article_text(article)
        if count_tokens(article_text) <= max_tokens:
            chunks.append(create_article_chunk(article))
        else:
            chunks.extend(split_by_paragraph(article))
    return chunks
```

### 판례 청킹 전략
```python
def chunk_case(case_data):
    """판례는 섹션별로 분리"""
    sections = [
        ("판시사항", case_data.get("판시사항")),
        ("판결요지", case_data.get("판결요지")),
        ("이유", case_data.get("판결이유")),
    ]
    chunks = []
    for section_name, content in sections:
        if content:
            chunks.append({
                "id": f"{case_id}_{section_name}",
                "content": f"[{section_name}]\n{content}",
                "metadata": {
                    "case_number": case_data["사건번호"],
                    "section": section_name,
                    "court": case_data["법원명"],
                    "ruling_date": case_data["선고일자"]
                }
            })
    return chunks
```

---

## 7. 참조 관계 추출

### 자동 추출 파이프라인
```python
import re

def extract_law_references(text):
    """법령 참조 추출"""
    patterns = [
        # 「법령명」 제N조
        (r'「([^」]+)」\s*제(\d+)조(의\d+)?', 'external'),
        # 이 법 제N조, 동법 제N조
        (r'(이\s*법|동법|본법)\s*제(\d+)조(의\d+)?', 'internal'),
        # 제N조 (같은 법령 내)
        (r'(?<!「[^」]{0,50})제(\d+)조(의\d+)?(?!\s*[^」]*」)', 'internal'),
    ]
    
    references = []
    for pattern, ref_type in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            references.append({
                "text": match.group(0),
                "type": ref_type,
                "position": match.span()
            })
    return references

def extract_case_references(text):
    """판례 참조 추출"""
    # 대법원 2023. 1. 15. 선고 2022다12345 판결
    pattern = r'(대법원|헌법재판소|[가-힣]+법원)\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*선고\s*(\d{4}[가-힣]+\d+)\s*판결'
    
    references = []
    for match in re.finditer(pattern, text):
        references.append({
            "court": match.group(1),
            "case_number": match.group(5),
            "ruling_date": f"{match.group(2)}-{match.group(3)}-{match.group(4)}"
        })
    return references
```

---

## 8. Claude 활용 팁

### 법령 구조 분석 요청
```
"[법령 도메인] 이 법률의 조문 구조를 분석해줘. 
특히 다른 법령을 참조하는 부분과 위임 조항을 찾아줘."
```

### 참조 관계 추출 요청
```
"[법령 도메인] 민법 제750조 텍스트에서 참조 관계를 추출해줘.
어떤 다른 조문을 인용하고 있고, 어떤 판례가 이 조문을 해석했는지."
```

### Graph DB 쿼리 작성 요청
```
"[법령 도메인] '불법행위 손해배상'과 관련된 법령과 판례의 
참조 네트워크를 탐색하는 Cypher 쿼리를 작성해줘."
```

### 청킹 전략 검토 요청
```
"[법령 도메인] 이 법률은 조문이 굉장히 긴데, 
검색 성능을 위한 최적의 청킹 전략을 제안해줘."
```
