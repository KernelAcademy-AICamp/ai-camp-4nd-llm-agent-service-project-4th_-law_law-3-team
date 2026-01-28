"""
Tokenizer Utilities - 토크나이저 및 한국어 처리 유틸리티
"""

from typing import List, Optional

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from konlpy.tag import Mecab, Okt, Komoran, Hannanum, Kkma
    HAS_KONLPY = True
except ImportError:
    HAS_KONLPY = False


def get_tokenizer(model_name: str = "nlpai-lab/KURE-v1"):
    """
    HuggingFace 토크나이저 로드
    
    Example:
        tokenizer = get_tokenizer("nlpai-lab/KURE-v1")
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers가 필요합니다: pip install transformers")
    
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer) -> int:
    """
    토큰 수 계산
    
    Example:
        tokenizer = get_tokenizer()
        num_tokens = count_tokens("법률 텍스트", tokenizer)
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    add_ellipsis: bool = False
) -> str:
    """
    토큰 수에 맞게 텍스트 자르기
    
    Args:
        text: 원본 텍스트
        tokenizer: 토크나이저
        max_tokens: 최대 토큰 수
        add_ellipsis: 말줄임표 추가 여부
    
    Returns:
        잘린 텍스트
    """
    encoding = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    
    if len(encoding['input_ids']) < max_tokens:
        return text
    
    offsets = encoding['offset_mapping']
    end_char = offsets[-1][1]
    truncated = text[:end_char]
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def batch_count_tokens(texts: List[str], tokenizer) -> List[int]:
    """배치 토큰 수 계산"""
    return [count_tokens(text, tokenizer) for text in texts]


class KoreanTokenizer:
    """
    KoNLPy 기반 한국어 토크나이저
    
    Args:
        backend: 'mecab', 'okt', 'komoran', 'hannanum', 'kkma'
    
    Example:
        # Mecab (빠름, 설치 복잡)
        tokenizer = KoreanTokenizer(backend="mecab")
        
        # Okt (설치 쉬움, 느림)
        tokenizer = KoreanTokenizer(backend="okt")
        
        tokens = tokenizer.tokenize("법률 텍스트")
        nouns = tokenizer.nouns("법률 텍스트")
    """
    
    BACKENDS = {
        'mecab': Mecab if HAS_KONLPY else None,
        'okt': Okt if HAS_KONLPY else None,
        'komoran': Komoran if HAS_KONLPY else None,
        'hannanum': Hannanum if HAS_KONLPY else None,
        'kkma': Kkma if HAS_KONLPY else None,
    }
    
    def __init__(self, backend: str = "okt"):
        if not HAS_KONLPY:
            raise ImportError("konlpy가 필요합니다: pip install konlpy")
        
        backend = backend.lower()
        if backend not in self.BACKENDS:
            raise ValueError(f"지원하지 않는 백엔드: {backend}")
        
        backend_class = self.BACKENDS[backend]
        if backend_class is None:
            raise ImportError(f"{backend} 백엔드를 로드할 수 없습니다")
        
        self.backend_name = backend
        self.tagger = backend_class()
    
    def tokenize(self, text: str) -> List[str]:
        """형태소 분석"""
        return self.tagger.morphs(text)
    
    def nouns(self, text: str) -> List[str]:
        """명사 추출"""
        return self.tagger.nouns(text)
    
    def pos(self, text: str) -> List[tuple]:
        """품사 태깅"""
        return self.tagger.pos(text)
    
    def normalize(self, text: str) -> str:
        """텍스트 정규화 (형태소 재결합)"""
        morphs = self.tokenize(text)
        return ' '.join(morphs)


class LegalTermTokenizer:
    """
    법률 용어 특화 토크나이저
    
    법률 특수 용어를 보존하면서 토큰화
    
    Example:
        tokenizer = LegalTermTokenizer()
        tokens = tokenizer.tokenize("민법 제750조에 따른 손해배상책임")
    """
    
    LEGAL_TERMS = [
        "손해배상", "불법행위", "채무불이행", "계약해지", "계약해제",
        "소멸시효", "제척기간", "선의취득", "점유권", "소유권",
        "저당권", "질권", "유치권", "전세권", "지상권",
        "대위변제", "구상권", "연대채무", "보증채무", "담보",
        "가등기", "본등기", "말소등기", "이전등기", "설정등기",
        "가압류", "가처분", "강제집행", "채권양도", "채무인수",
    ]
    
    def __init__(self, base_tokenizer: Optional[KoreanTokenizer] = None):
        if base_tokenizer:
            self.base_tokenizer = base_tokenizer
        elif HAS_KONLPY:
            self.base_tokenizer = KoreanTokenizer(backend="okt")
        else:
            self.base_tokenizer = None
        
        self.legal_term_set = set(self.LEGAL_TERMS)
    
    def tokenize(self, text: str) -> List[str]:
        """법률 용어 보존 토큰화"""
        protected_text, placeholders = self._protect_legal_terms(text)
        
        if self.base_tokenizer:
            tokens = self.base_tokenizer.tokenize(protected_text)
        else:
            tokens = protected_text.split()
        
        restored_tokens = self._restore_legal_terms(tokens, placeholders)
        
        return restored_tokens
    
    def _protect_legal_terms(self, text: str) -> tuple:
        """법률 용어를 플레이스홀더로 대체"""
        import re
        
        placeholders = {}
        protected_text = text
        
        for i, term in enumerate(sorted(self.LEGAL_TERMS, key=len, reverse=True)):
            placeholder = f"__LEGAL_{i}__"
            if term in protected_text:
                placeholders[placeholder] = term
                protected_text = protected_text.replace(term, placeholder)
        
        return protected_text, placeholders
    
    def _restore_legal_terms(self, tokens: List[str], placeholders: dict) -> List[str]:
        """플레이스홀더를 원래 용어로 복원"""
        restored = []
        for token in tokens:
            restored_token = token
            for placeholder, term in placeholders.items():
                if placeholder in restored_token:
                    restored_token = restored_token.replace(placeholder, term)
            restored.append(restored_token)
        return restored
    
    def extract_legal_terms(self, text: str) -> List[str]:
        """텍스트에서 법률 용어 추출"""
        found_terms = []
        for term in self.LEGAL_TERMS:
            if term in text:
                found_terms.append(term)
        return found_terms


class TokenizerWrapper:
    """
    HuggingFace + KoNLPy 통합 래퍼
    
    임베딩용 토크나이저와 분석용 토크나이저 통합
    
    Example:
        wrapper = TokenizerWrapper(
            embedding_model="nlpai-lab/KURE-v1",
            analysis_backend="okt"
        )
        
        # 임베딩용 토큰 수
        num_tokens = wrapper.count_embedding_tokens(text)
        
        # 형태소 분석
        morphs = wrapper.analyze(text)
    """
    
    def __init__(
        self,
        embedding_model: str = "nlpai-lab/KURE-v1",
        analysis_backend: str = "okt"
    ):
        self.embedding_tokenizer = None
        self.analysis_tokenizer = None
        
        if HAS_TRANSFORMERS:
            self.embedding_tokenizer = get_tokenizer(embedding_model)
        
        if HAS_KONLPY:
            self.analysis_tokenizer = KoreanTokenizer(backend=analysis_backend)
    
    def count_embedding_tokens(self, text: str) -> int:
        """임베딩 모델 토큰 수"""
        if self.embedding_tokenizer is None:
            raise RuntimeError("transformers가 설치되지 않았습니다")
        return count_tokens(text, self.embedding_tokenizer)
    
    def truncate_for_embedding(self, text: str, max_tokens: int) -> str:
        """임베딩용 텍스트 자르기"""
        if self.embedding_tokenizer is None:
            raise RuntimeError("transformers가 설치되지 않았습니다")
        return truncate_to_tokens(text, self.embedding_tokenizer, max_tokens)
    
    def analyze(self, text: str) -> List[str]:
        """형태소 분석"""
        if self.analysis_tokenizer is None:
            raise RuntimeError("konlpy가 설치되지 않았습니다")
        return self.analysis_tokenizer.tokenize(text)
    
    def extract_nouns(self, text: str) -> List[str]:
        """명사 추출"""
        if self.analysis_tokenizer is None:
            raise RuntimeError("konlpy가 설치되지 않았습니다")
        return self.analysis_tokenizer.nouns(text)


if __name__ == "__main__":
    print("Tokenizer Utilities 모듈 로드 완료")
    print(f"Transformers: {HAS_TRANSFORMERS}")
    print(f"KoNLPy: {HAS_KONLPY}")
    
    if HAS_TRANSFORMERS:
        tokenizer = get_tokenizer()
        test_text = "민법 제750조에 따른 손해배상책임"
        num_tokens = count_tokens(test_text, tokenizer)
        print(f"Token count: {num_tokens}")
    
    if HAS_KONLPY:
        korean_tokenizer = KoreanTokenizer(backend="okt")
        tokens = korean_tokenizer.tokenize("손해배상청구권의 소멸시효")
        print(f"Morphs: {tokens}")
        nouns = korean_tokenizer.nouns("손해배상청구권의 소멸시효")
        print(f"Nouns: {nouns}")
