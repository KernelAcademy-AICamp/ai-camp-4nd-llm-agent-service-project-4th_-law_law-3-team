"""
Chunking Utilities - 문서 청킹 전략
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator

# Optional imports
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class Chunk:
    """청크 데이터 클래스"""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            import hashlib
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:16]


class BaseChunker(ABC):
    """청커 베이스 클래스"""
    
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """텍스트를 청크로 분리"""
        pass
    
    def chunk_batch(self, texts: List[str]) -> List[List[Chunk]]:
        """배치 청킹"""
        return [self.chunk(text) for text in texts]


class TokenChunker(BaseChunker):
    """
    토큰 기반 청킹
    
    Args:
        model_name: HuggingFace 모델 이름
        max_tokens: 최대 토큰 수
        overlap_tokens: 오버랩 토큰 수
    
    Example:
        chunker = TokenChunker("nlpai-lab/KURE-v1", max_tokens=480, overlap_tokens=50)
        chunks = chunker.chunk(long_text)
    """
    
    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        max_tokens: int = 480,
        overlap_tokens: int = 50
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers가 필요합니다: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str) -> List[Chunk]:
        """토큰 기반으로 텍스트 분리"""
        if not text.strip():
            return []
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        if len(tokens) <= self.max_tokens:
            return [Chunk(text=text, start_idx=0, end_idx=len(text))]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            
            # 문자 위치 계산
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            
            chunk_text = text[char_start:char_end]
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=char_start,
                end_idx=char_end,
                metadata={"token_count": end - start}
            ))
            
            # 다음 시작점 (오버랩 적용)
            start = end - self.overlap_tokens
            
            if start >= len(tokens) - self.overlap_tokens:
                break
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class SentenceChunker(BaseChunker):
    """
    문장 기반 청킹
    
    Args:
        max_sentences: 청크당 최대 문장 수
        overlap_sentences: 오버랩 문장 수
    """
    
    def __init__(
        self,
        max_sentences: int = 5,
        overlap_sentences: int = 1
    ):
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences
        
        # 한국어 문장 분리 패턴
        self.sentence_pattern = re.compile(r'(?<=[.!?。])\s+')
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[Chunk]:
        """문장 기반으로 텍스트 분리"""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.max_sentences:
            return [Chunk(text=text, start_idx=0, end_idx=len(text))]
        
        chunks = []
        start = 0
        current_pos = 0
        
        while start < len(sentences):
            end = min(start + self.max_sentences, len(sentences))
            
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            
            # 원본 텍스트에서 위치 찾기
            chunk_start = text.find(chunk_sentences[0], current_pos)
            chunk_end = text.find(chunk_sentences[-1], chunk_start) + len(chunk_sentences[-1])
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=chunk_start,
                end_idx=chunk_end,
                metadata={"sentence_count": len(chunk_sentences)}
            ))
            
            current_pos = chunk_start
            start = end - self.overlap_sentences
            
            if start >= len(sentences) - self.overlap_sentences:
                break
        
        return chunks


class ParagraphChunker(BaseChunker):
    """
    문단 기반 청킹
    
    Args:
        max_paragraphs: 청크당 최대 문단 수
        overlap_paragraphs: 오버랩 문단 수
        min_paragraph_length: 최소 문단 길이 (글자)
    """
    
    def __init__(
        self,
        max_paragraphs: int = 3,
        overlap_paragraphs: int = 1,
        min_paragraph_length: int = 10
    ):
        self.max_paragraphs = max_paragraphs
        self.overlap_paragraphs = overlap_paragraphs
        self.min_paragraph_length = min_paragraph_length
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """문단 분리"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [
            p.strip() for p in paragraphs
            if p.strip() and len(p.strip()) >= self.min_paragraph_length
        ]
    
    def chunk(self, text: str) -> List[Chunk]:
        """문단 기반으로 텍스트 분리"""
        paragraphs = self._split_paragraphs(text)
        
        if len(paragraphs) <= self.max_paragraphs:
            return [Chunk(text=text, start_idx=0, end_idx=len(text))]
        
        chunks = []
        start = 0
        
        while start < len(paragraphs):
            end = min(start + self.max_paragraphs, len(paragraphs))
            
            chunk_paragraphs = paragraphs[start:end]
            chunk_text = '\n\n'.join(chunk_paragraphs)
            
            # 위치 계산 (간단화)
            chunk_start = text.find(chunk_paragraphs[0])
            chunk_end = text.find(chunk_paragraphs[-1]) + len(chunk_paragraphs[-1])
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=max(0, chunk_start),
                end_idx=min(len(text), chunk_end),
                metadata={"paragraph_count": len(chunk_paragraphs)}
            ))
            
            start = end - self.overlap_paragraphs
            
            if start >= len(paragraphs) - self.overlap_paragraphs:
                break
        
        return chunks


class LegalChunker(BaseChunker):
    """
    법률 문서 특화 청킹
    
    조문(Article) 단위로 자연스럽게 분리
    
    Args:
        model_name: 토큰 수 계산용 모델
        max_tokens: 최대 토큰 수
        overlap_tokens: 오버랩 토큰 수
        preserve_article_structure: 조문 구조 유지
    
    Example:
        chunker = LegalChunker(max_tokens=480)
        chunks = chunker.chunk(legal_text)
    """
    
    # 법률 조문 패턴
    ARTICLE_PATTERN = re.compile(
        r'(제\s*\d+조(?:의\d+)?)\s*[\(（]([^)）]+)[\)）]?',
        re.MULTILINE
    )
    
    CLAUSE_PATTERN = re.compile(
        r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]',
        re.MULTILINE
    )
    
    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        max_tokens: int = 480,
        overlap_tokens: int = 50,
        preserve_article_structure: bool = True
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_article_structure = preserve_article_structure
        
        if HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback: 대략적 추정 (한국어는 글자당 약 1.5 토큰)
        return int(len(text) * 1.5)
    
    def _split_articles(self, text: str) -> List[Dict[str, Any]]:
        """조문 단위로 분리"""
        articles = []
        
        matches = list(self.ARTICLE_PATTERN.finditer(text))
        
        if not matches:
            # 조문 패턴이 없으면 전체를 하나로
            return [{"title": "", "content": text, "start": 0, "end": len(text)}]
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            articles.append({
                "title": match.group(1),
                "subtitle": match.group(2) if match.group(2) else "",
                "content": text[start:end].strip(),
                "start": start,
                "end": end
            })
        
        return articles
    
    def chunk(self, text: str) -> List[Chunk]:
        """법률 문서 청킹"""
        if not text.strip():
            return []
        
        articles = self._split_articles(text)
        chunks = []
        
        for article in articles:
            article_text = article["content"]
            article_tokens = self._count_tokens(article_text)
            
            if article_tokens <= self.max_tokens:
                # 조문이 최대 토큰 이하면 그대로 사용
                chunks.append(Chunk(
                    text=article_text,
                    start_idx=article["start"],
                    end_idx=article["end"],
                    metadata={
                        "article_title": article.get("title", ""),
                        "article_subtitle": article.get("subtitle", ""),
                        "token_count": article_tokens
                    }
                ))
            else:
                # 조문이 너무 길면 항(clause) 단위로 분리
                sub_chunks = self._split_long_article(article)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_long_article(self, article: Dict[str, Any]) -> List[Chunk]:
        """긴 조문을 항 단위로 분리"""
        text = article["content"]
        clauses = self.CLAUSE_PATTERN.split(text)
        
        chunks = []
        current_text = ""
        current_start = article["start"]
        
        for i, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause:
                continue
            
            test_text = current_text + " " + clause if current_text else clause
            
            if self._count_tokens(test_text) <= self.max_tokens:
                current_text = test_text
            else:
                # 현재까지 모은 텍스트로 청크 생성
                if current_text:
                    chunks.append(Chunk(
                        text=current_text,
                        start_idx=current_start,
                        end_idx=current_start + len(current_text),
                        metadata={
                            "article_title": article.get("title", ""),
                            "token_count": self._count_tokens(current_text)
                        }
                    ))
                
                current_text = clause
                current_start = article["start"] + text.find(clause)
        
        # 마지막 청크
        if current_text:
            chunks.append(Chunk(
                text=current_text,
                start_idx=current_start,
                end_idx=article["end"],
                metadata={
                    "article_title": article.get("title", ""),
                    "token_count": self._count_tokens(current_text)
                }
            ))
        
        return chunks


class RecursiveChunker(BaseChunker):
    """
    재귀적 청킹 - 여러 분리자를 순차적으로 시도
    
    Args:
        separators: 분리자 리스트 (우선순위 순)
        max_chars: 최대 문자 수
        overlap_chars: 오버랩 문자 수
    """
    
    def __init__(
        self,
        separators: List[str] = None,
        max_chars: int = 1000,
        overlap_chars: int = 100
    ):
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
    
    def chunk(self, text: str) -> List[Chunk]:
        """재귀적으로 텍스트 분리"""
        return self._recursive_chunk(text, self.separators, 0)
    
    def _recursive_chunk(
        self,
        text: str,
        separators: List[str],
        start_idx: int
    ) -> List[Chunk]:
        """재귀 청킹 구현"""
        if len(text) <= self.max_chars:
            return [Chunk(text=text, start_idx=start_idx, end_idx=start_idx + len(text))]
        
        if not separators:
            # 더 이상 분리자가 없으면 강제 분리
            chunks = []
            for i in range(0, len(text), self.max_chars - self.overlap_chars):
                end = min(i + self.max_chars, len(text))
                chunks.append(Chunk(
                    text=text[i:end],
                    start_idx=start_idx + i,
                    end_idx=start_idx + end
                ))
            return chunks
        
        separator = separators[0]
        splits = text.split(separator) if separator else list(text)
        
        chunks = []
        current_text = ""
        current_start = start_idx
        
        for split in splits:
            test_text = current_text + separator + split if current_text else split
            
            if len(test_text) <= self.max_chars:
                current_text = test_text
            else:
                if current_text:
                    chunks.append(Chunk(
                        text=current_text,
                        start_idx=current_start,
                        end_idx=current_start + len(current_text)
                    ))
                
                # 현재 split이 여전히 크면 재귀
                if len(split) > self.max_chars:
                    sub_chunks = self._recursive_chunk(split, separators[1:], current_start)
                    chunks.extend(sub_chunks)
                    current_text = ""
                else:
                    current_text = split
                
                current_start = start_idx + text.find(split, current_start - start_idx)
        
        if current_text:
            chunks.append(Chunk(
                text=current_text,
                start_idx=current_start,
                end_idx=current_start + len(current_text)
            ))
        
        return chunks


if __name__ == "__main__":
    # 테스트
    sample_text = """
    제1조(목적) 이 법은 민사에 관한 기본법으로서 개인의 사적 생활관계를 규율함을 목적으로 한다.
    
    제2조(신의성실) ①권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.
    ②권리는 남용하지 못한다.
    
    제3조(권리능력의 존속기간) 사람은 생존한 동안 권리와 의무의 주체가 된다.
    """
    
    print("=== Token Chunker ===")
    if HAS_TRANSFORMERS:
        chunker = TokenChunker(max_tokens=50, overlap_tokens=10)
        for i, chunk in enumerate(chunker.chunk(sample_text)):
            print(f"Chunk {i}: {chunk.text[:50]}...")
    
    print("\n=== Legal Chunker ===")
    legal_chunker = LegalChunker(max_tokens=100)
    for i, chunk in enumerate(legal_chunker.chunk(sample_text)):
        print(f"Chunk {i}: {chunk.metadata.get('article_title', 'N/A')} - {chunk.text[:50]}...")
