"""
Document Parser - 다양한 형식의 문서 파싱
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False


class JSONParser:
    """
    JSON/JSONL 파서
    
    Example:
        parser = JSONParser()
        
        # JSON 파일
        documents = parser.parse_file("data.json")
        
        # JSONL 스트리밍
        for doc in parser.parse_jsonl_streaming("data.jsonl"):
            process(doc)
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """JSON 파일 파싱"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        return [data]
    
    def parse_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일 전체 파싱"""
        documents = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))
        return documents
    
    def parse_jsonl_streaming(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """JSONL 스트리밍 파싱"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def save_jsonl(self, documents: List[Dict[str, Any]], file_path: str):
        """JSONL 파일 저장"""
        with open(file_path, 'w', encoding=self.encoding) as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')


class XMLParser:
    """
    XML 파서 (법제처 XML 형식 지원)
    
    Args:
        text_xpath: 텍스트 추출 XPath
        id_xpath: ID 추출 XPath
        encoding: 파일 인코딩
    
    Example:
        parser = XMLParser(
            text_xpath=".//조문내용",
            id_xpath=".//법령ID"
        )
        documents = parser.parse_file("legal_data.xml")
    """
    
    def __init__(
        self,
        text_xpath: str = ".//content",
        id_xpath: str = ".//id",
        encoding: str = 'utf-8'
    ):
        self.text_xpath = text_xpath
        self.id_xpath = id_xpath
        self.encoding = encoding
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """XML 파일 파싱"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        documents = []
        
        for item in root:
            doc = self._parse_element(item)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _parse_element(self, element) -> Dict[str, Any]:
        """XML 요소 파싱"""
        doc = {}
        
        id_elem = element.find(self.id_xpath)
        if id_elem is not None and id_elem.text:
            doc['id'] = id_elem.text.strip()
        
        text_elem = element.find(self.text_xpath)
        if text_elem is not None and text_elem.text:
            doc['text'] = text_elem.text.strip()
        
        for child in element:
            if child.tag not in doc and child.text:
                doc[child.tag] = child.text.strip()
        
        return doc
    
    def parse_string(self, xml_string: str) -> List[Dict[str, Any]]:
        """XML 문자열 파싱"""
        root = ET.fromstring(xml_string)
        
        documents = []
        for item in root:
            doc = self._parse_element(item)
            if doc:
                documents.append(doc)
        
        return documents


class HTMLParser:
    """
    HTML 파서
    
    Args:
        content_selector: 콘텐츠 CSS 선택자
        remove_selectors: 제거할 요소 선택자 리스트
    
    Example:
        parser = HTMLParser(
            content_selector="article.content",
            remove_selectors=["script", "style", "nav", "footer"]
        )
        documents = parser.parse_file("page.html")
    """
    
    def __init__(
        self,
        content_selector: Optional[str] = None,
        remove_selectors: Optional[List[str]] = None,
        encoding: str = 'utf-8'
    ):
        if not HAS_BS4:
            raise ImportError("beautifulsoup4가 필요합니다: pip install beautifulsoup4")
        
        self.content_selector = content_selector
        self.remove_selectors = remove_selectors or ["script", "style", "nav", "footer", "header"]
        self.encoding = encoding
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """HTML 파일 파싱"""
        with open(file_path, 'r', encoding=self.encoding) as f:
            html_content = f.read()
        
        return self.parse_string(html_content, source=file_path)
    
    def parse_string(self, html_string: str, source: str = "") -> List[Dict[str, Any]]:
        """HTML 문자열 파싱"""
        soup = BeautifulSoup(html_string, 'html.parser')
        
        for selector in self.remove_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        if self.content_selector:
            content_elements = soup.select(self.content_selector)
            if content_elements:
                text = ' '.join(elem.get_text(strip=True) for elem in content_elements)
            else:
                text = soup.get_text(strip=True)
        else:
            text = soup.get_text(strip=True)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        return [{
            "id": source or hash(html_string),
            "title": title,
            "text": text,
            "source": source
        }]


class LegalDocumentParser:
    """
    법제처 API 응답 파서
    
    법제처 Open API의 JSON 응답을 파싱
    
    Example:
        parser = LegalDocumentParser()
        
        documents = parser.parse_law_list(api_response)
        full_docs = parser.parse_law_content(content_response)
    """
    
    def __init__(self):
        self.json_parser = JSONParser()
    
    def parse_law_list(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """법령 목록 API 응답 파싱"""
        documents = []
        
        items = response.get('법령목록', response.get('LawList', []))
        if isinstance(items, dict):
            items = [items]
        
        for item in items:
            doc = {
                'id': item.get('법령ID', item.get('LawId', '')),
                'name': item.get('법령명', item.get('LawName', '')),
                'type': item.get('법령종류', item.get('LawType', '')),
                'effective_date': item.get('시행일자', item.get('EffectiveDate', '')),
                'department': item.get('소관부처', item.get('Department', '')),
            }
            documents.append(doc)
        
        return documents
    
    def parse_law_content(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """법령 본문 API 응답 파싱"""
        law_info = response.get('법령', response.get('Law', {}))
        
        articles = []
        article_list = law_info.get('조문', law_info.get('Articles', []))
        if isinstance(article_list, dict):
            article_list = [article_list]
        
        for article in article_list:
            articles.append({
                'number': article.get('조문번호', article.get('ArticleNumber', '')),
                'title': article.get('조문제목', article.get('ArticleTitle', '')),
                'content': article.get('조문내용', article.get('ArticleContent', '')),
            })
        
        return {
            'id': law_info.get('법령ID', law_info.get('LawId', '')),
            'name': law_info.get('법령명', law_info.get('LawName', '')),
            'type': law_info.get('법령종류', law_info.get('LawType', '')),
            'effective_date': law_info.get('시행일자', law_info.get('EffectiveDate', '')),
            'articles': articles,
            'full_text': self._combine_articles(articles)
        }
    
    def _combine_articles(self, articles: List[Dict[str, Any]]) -> str:
        """조문들을 하나의 텍스트로 결합"""
        parts = []
        for article in articles:
            number = article.get('number', '')
            title = article.get('title', '')
            content = article.get('content', '')
            
            if number:
                parts.append(f"{number}")
            if title:
                parts.append(f"({title})")
            if content:
                parts.append(content)
            parts.append("")
        
        return '\n'.join(parts)


class UniversalParser:
    """
    범용 파서 - 파일 확장자에 따라 적절한 파서 선택
    
    Example:
        parser = UniversalParser()
        
        documents = parser.parse("data.json")
        documents = parser.parse("data.xml")
        documents = parser.parse("page.html")
    """
    
    def __init__(self):
        self.json_parser = JSONParser()
        self.xml_parser = XMLParser()
        self.html_parser = HTMLParser() if HAS_BS4 else None
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """파일 형식에 따라 자동 파싱"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            return self.json_parser.parse_file(file_path)
        elif suffix == '.jsonl':
            return self.json_parser.parse_jsonl(file_path)
        elif suffix == '.xml':
            return self.xml_parser.parse_file(file_path)
        elif suffix in ['.html', '.htm']:
            if self.html_parser is None:
                raise ImportError("beautifulsoup4가 필요합니다")
            return self.html_parser.parse_file(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
    
    def parse_streaming(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """스트리밍 파싱 (JSONL만 지원)"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.jsonl':
            yield from self.json_parser.parse_jsonl_streaming(file_path)
        else:
            for doc in self.parse(file_path):
                yield doc


if __name__ == "__main__":
    print("Document Parser 모듈 로드 완료")
    print(f"BeautifulSoup: {HAS_BS4}")
    print(f"XML: {HAS_XML}")
    
    json_parser = JSONParser()
    
    test_data = [
        {"id": "1", "text": "테스트 문서 1"},
        {"id": "2", "text": "테스트 문서 2"}
    ]
    json_parser.save_jsonl(test_data, "/tmp/test.jsonl")
    
    loaded = json_parser.parse_jsonl("/tmp/test.jsonl")
    print(f"Loaded {len(loaded)} documents")
