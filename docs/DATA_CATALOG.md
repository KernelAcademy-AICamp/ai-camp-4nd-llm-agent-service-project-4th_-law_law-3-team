# 법률 데이터 카탈로그

`data/law_data/` 디렉토리에 저장된 법률 데이터 목록입니다.

## 데이터 현황 요약

| 분류 | 파일 수 | 총 레코드 | 총 용량 |
|------|---------|----------|---------|
| 법령/규칙 | 3개 | 33,571건 | ~583MB |
| 판례/결정례 | 3개 | 100,159건 | ~626MB |
| 위원회 결정문 | 10개 | 19,828건 | ~335MB |
| 기타 | 2개 | 40,758건 | ~76MB |
| **합계** | **18개** | **~194,316건** | **~1.6GB** |

---

## 1. 법령 및 규칙

### law.json
- **설명**: 법령 (민법, 형법, 상법 등 대한민국 법률 조문)
- **레코드 수**: 5,841건
- **파일 크기**: 214.3MB
- **주요 필드**: `law_id`, `law_name`, `content`, `promulgation_date`, `ministry`, `law_type`
- **출처**: 국가법령정보센터

### administrative_rules_full.json
- **설명**: 행정규칙 (훈령, 예규, 고시 등)
- **레코드 수**: 21,889건
- **파일 크기**: 312.2MB
- **출처**: 국가법령정보센터

### treaty-full.json
- **설명**: 조약 (대한민국이 체결한 국제조약)
- **레코드 수**: 3,589건
- **파일 크기**: 56.8MB
- **주요 필드**: `조약일련번호`, `조약명_한글`, `조약명_영문`, `발효일자`, `체결대상국가`
- **출처**: 국가법령정보센터

---

## 2. 판례 및 결정례

### precedents_full.json (+ 분할 파일 1~5)
- **설명**: 판례 (대법원, 고등법원, 지방법원 판결문)
- **레코드 수**: 29,120건
- **파일 크기**: ~1GB (분할 포함)
- **분할 파일**: `precedents_full-1.json` ~ `precedents_full-5.json`
- **출처**: 대법원 종합법률정보

### constitutional_full.json
- **설명**: 헌법재판소 결정례
- **레코드 수**: 36,781건
- **파일 크기**: 22.7MB
- **출처**: 헌법재판소

### administration_full.json
- **설명**: 행정심판례 (중앙행정심판위원회)
- **레코드 수**: 34,258건
- **파일 크기**: 423.9MB
- **출처**: 중앙행정심판위원회

---

## 3. 위원회 결정문

### 공정거래 관련

| 파일명 | 위원회 | 레코드 수 | 파일 크기 |
|--------|--------|----------|----------|
| ftc-full.json | 공정거래위원회 | 8,029건 | 164.6MB |
| sfc-full.json | 증권선물위원회 | 636건 | 1.2MB |
| fsc-full.json | 금융위원회 | 663건 | 2.0MB |

### 권익/인권 관련

| 파일명 | 위원회 | 레코드 수 | 파일 크기 |
|--------|--------|----------|----------|
| nhrck-full.json | 국가인권위원회 | 3,732건 | 114.0MB |
| acr-full.json | 국민권익위원회 | 635건 | 9.3MB |
| ppc-full.json | 개인정보보호위원회 | 3,889건 | 20.6MB |

### 노동/산재 관련

| 파일명 | 위원회 | 레코드 수 | 파일 크기 |
|--------|--------|----------|----------|
| eiac-full.json | 고용보험심사위원회 | 118건 | 2.1MB |
| iaciac-full.json | 산업재해보상보험재심사위원회 | 934건 | 10.8MB |

### 기타 위원회

| 파일명 | 위원회 | 레코드 수 | 파일 크기 |
|--------|--------|----------|----------|
| kcc-full.json | 방송통신위원회 | 811건 | 4.3MB |
| ecc-full.json | 중앙환경분쟁조정위원회 | 358건 | 5.7MB |
| oclt-full.json | 중앙토지수용위원회 | 23건 | 4.4KB |

---

## 4. 기타 데이터

### legislation_full.json
- **설명**: 법령해석례 (법제처 법령해석 사례)
- **레코드 수**: 8,597건
- **파일 크기**: 76.0MB
- **출처**: 법제처

### lawterms_full.json
- **설명**: 법률용어사전
- **레코드 수**: 37,169건
- **파일 크기**: 19.2MB
- **출처**: 법제처

---

## 파일명 약어 정리

| 약어 | 전체 명칭 |
|------|----------|
| ftc | Fair Trade Commission (공정거래위원회) |
| sfc | Securities and Futures Commission (증권선물위원회) |
| fsc | Financial Services Commission (금융위원회) |
| nhrck | National Human Rights Commission of Korea (국가인권위원회) |
| acr | Anti-Corruption & Civil Rights (국민권익위원회) |
| ppc | Personal Information Protection Commission (개인정보보호위원회) |
| eiac | Employment Insurance Appeals Commission (고용보험심사위원회) |
| iaciac | Industrial Accident Compensation Insurance Appeals Commission (산업재해보상보험재심사위원회) |
| kcc | Korea Communications Commission (방송통신위원회) |
| ecc | Environmental Conflict Conciliation Committee (중앙환경분쟁조정위원회) |
| oclt | Central Land Expropriation Committee (중앙토지수용위원회) |

---

## 데이터 활용

### RAG 검색 대상
- 판례, 헌재결정례, 행정심판례, 법령해석례
- 위원회 결정문 (법적 판단 사례)

### 법률 조문 참조
- `law.json`: 실제 법률 조문 검색 및 인용
- `administrative_rules_full.json`: 세부 행정규칙 확인

### 법률 용어 설명
- `lawterms_full.json`: 법률 용어 정의 및 해설

---

*최종 업데이트: 2025-01-20*
