"""
Embedding Generator - 대규모 임베딩 생성
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Iterator, Callable, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Optional imports
try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    embeddings: np.ndarray
    ids: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None


class TextDataset(Dataset):
    """텍스트 데이터셋"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


class EmbeddingGenerator:
    """
    임베딩 생성기
    
    Args:
        model_name: HuggingFace 모델 이름
        device: 디바이스 ("auto", "cuda", "mps", "cpu")
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        use_amp: Mixed Precision 사용
        normalize: L2 정규화 적용
    
    Example:
        generator = EmbeddingGenerator("nlpai-lab/KURE-v1")
        embeddings = generator.encode_batch(texts, show_progress=True)
    """
    
    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        use_amp: bool = True,
        normalize: bool = True
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers가 필요합니다: pip install transformers")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        
        # 디바이스 설정
        self.device = self._get_device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.dtype = self._get_dtype()
        
        # 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 임베딩 차원
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"Model loaded: {model_name}")
        print(f"Device: {self.device}, dtype: {self.dtype}")
        print(f"Embedding dim: {self.embedding_dim}")
    
    def _get_device(self, device: str) -> torch.device:
        """디바이스 자동 선택"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def _get_dtype(self) -> torch.dtype:
        """최적 dtype 선택"""
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    
    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    @torch.inference_mode()
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트 임베딩 생성
        
        Args:
            text: 단일 텍스트 또는 텍스트 리스트
        
        Returns:
            임베딩 numpy 배열
        """
        if isinstance(text, str):
            text = [text]
        
        # 토큰화
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 추론
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.use_amp
        ):
            outputs = self.model(**encoded)
        
        # Pooling
        embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
        
        # 정규화
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.cpu().numpy()
    
    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1000
    ) -> np.ndarray:
        """
        대규모 배치 임베딩 생성
        
        Args:
            texts: 텍스트 리스트
            show_progress: 진행률 표시
            checkpoint_path: 체크포인트 저장 경로
            checkpoint_interval: 체크포인트 저장 간격
        
        Returns:
            임베딩 numpy 배열 (N x embedding_dim)
        """
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if self.device.type == "mps" else 2,
            pin_memory=self.device.type == "cuda"
        )
        
        all_embeddings = []
        processed = 0
        
        # 체크포인트 로드
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path)
            all_embeddings = list(checkpoint['embeddings'])
            processed = checkpoint['processed']
            print(f"Resumed from checkpoint: {processed} items processed")
        
        iterator = tqdm(dataloader, desc="Encoding", disable=not show_progress)
        
        for batch_idx, batch in enumerate(iterator):
            # 이미 처리된 배치 스킵
            current_start = batch_idx * self.batch_size
            if current_start < processed:
                continue
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp
            ):
                outputs = self.model(**batch)
            
            embeddings = self._mean_pooling(outputs, batch['attention_mask'])
            
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            processed = current_start + len(batch['input_ids'])
            
            # 체크포인트 저장
            if checkpoint_path and processed % checkpoint_interval == 0:
                torch.save({
                    'embeddings': np.vstack(all_embeddings),
                    'processed': processed
                }, checkpoint_path)
        
        return np.vstack(all_embeddings)


class BatchEmbeddingProcessor:
    """
    대규모 배치 임베딩 프로세서
    
    76만+ 문서 처리에 최적화
    
    Args:
        model_name: 모델 이름
        output_dir: 출력 디렉토리
        batch_size: 배치 크기
        checkpoint_interval: 체크포인트 간격
        cache: EmbeddingCache 인스턴스 (선택)
    
    Example:
        processor = BatchEmbeddingProcessor(
            model_name="nlpai-lab/KURE-v1",
            output_dir="embeddings/",
            checkpoint_interval=10000
        )
        processor.process_file("documents.jsonl", id_field='doc_id', text_field='content')
    """
    
    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        output_dir: str = "./embeddings",
        batch_size: int = 32,
        checkpoint_interval: int = 10000,
        cache=None
    ):
        self.generator = EmbeddingGenerator(
            model_name=model_name,
            batch_size=batch_size
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.cache = cache
        
        self.progress_path = self.output_dir / "progress.json"
    
    def _load_progress(self) -> Dict[str, Any]:
        """진행 상황 로드"""
        if self.progress_path.exists():
            with open(self.progress_path, 'r') as f:
                return json.load(f)
        return {"processed_ids": [], "last_batch": 0}
    
    def _save_progress(self, progress: Dict[str, Any]):
        """진행 상황 저장"""
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f)
    
    def process_streaming(
        self,
        data_iterator: Iterator[Dict[str, Any]],
        id_field: str = 'id',
        text_field: str = 'text',
        save_format: str = 'npy'
    ):
        """
        스트리밍 데이터 처리
        
        Args:
            data_iterator: 문서 딕셔너리 이터레이터
            id_field: ID 필드명
            text_field: 텍스트 필드명
            save_format: 저장 형식 ('npy', 'pt')
        """
        progress = self._load_progress()
        processed_ids = set(progress["processed_ids"])
        
        batch_texts = []
        batch_ids = []
        batch_metadata = []
        total_processed = len(processed_ids)
        
        print(f"Starting from {total_processed} already processed documents")
        
        for doc in tqdm(data_iterator, desc="Processing"):
            doc_id = str(doc[id_field])
            
            # 이미 처리된 문서 스킵
            if doc_id in processed_ids:
                continue
            
            batch_texts.append(doc[text_field])
            batch_ids.append(doc_id)
            batch_metadata.append({k: v for k, v in doc.items() if k != text_field})
            
            # 배치가 찼으면 처리
            if len(batch_texts) >= self.generator.batch_size:
                embeddings = self.generator.encode(batch_texts)
                
                # 저장
                self._save_batch(batch_ids, embeddings, batch_metadata, total_processed, save_format)
                
                # 진행 상황 업데이트
                processed_ids.update(batch_ids)
                total_processed += len(batch_ids)
                
                if total_processed % self.checkpoint_interval == 0:
                    progress["processed_ids"] = list(processed_ids)
                    progress["last_batch"] = total_processed
                    self._save_progress(progress)
                    print(f"Checkpoint saved: {total_processed} documents processed")
                
                batch_texts = []
                batch_ids = []
                batch_metadata = []
        
        # 마지막 배치 처리
        if batch_texts:
            embeddings = self.generator.encode(batch_texts)
            self._save_batch(batch_ids, embeddings, batch_metadata, total_processed, save_format)
            processed_ids.update(batch_ids)
        
        # 최종 저장
        progress["processed_ids"] = list(processed_ids)
        progress["completed"] = True
        self._save_progress(progress)
        
        print(f"Completed: {len(processed_ids)} documents processed")
    
    def _save_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
        batch_num: int,
        save_format: str
    ):
        """배치 저장"""
        batch_path = self.output_dir / f"batch_{batch_num:08d}"
        
        if save_format == 'npy':
            np.save(f"{batch_path}_embeddings.npy", embeddings)
        else:
            torch.save(torch.from_numpy(embeddings), f"{batch_path}_embeddings.pt")
        
        with open(f"{batch_path}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump({"ids": ids, "metadata": metadata}, f, ensure_ascii=False)
    
    def process_file(
        self,
        input_path: str,
        id_field: str = 'id',
        text_field: str = 'text'
    ):
        """
        JSONL 파일 처리
        """
        def file_iterator():
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line)
        
        self.process_streaming(file_iterator(), id_field, text_field)
    
    def encode_single(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        return self.generator.encode(text)[0]
    
    def encode_chunks(self, chunks: List) -> np.ndarray:
        """청크 리스트 임베딩"""
        texts = [chunk.text for chunk in chunks]
        return self.generator.encode_batch(texts)


class ResumableProcessor:
    """
    중단 가능한 프로세서
    
    Example:
        processor = ResumableProcessor("progress.json", "nlpai-lab/KURE-v1")
        processor.process_with_resume(documents, 'doc_id', "embeddings.npy")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "nlpai-lab/KURE-v1"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.generator = EmbeddingGenerator(model_name=model_name)
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {"processed_ids": set(), "embeddings_file": None}
    
    def _save_checkpoint(self, data: Dict[str, Any]):
        # set을 list로 변환
        save_data = {
            "processed_ids": list(data["processed_ids"]),
            "embeddings_file": data.get("embeddings_file")
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(save_data, f)
    
    def process_with_resume(
        self,
        documents: List[Dict[str, Any]],
        id_field: str,
        output_path: str
    ):
        """중단된 지점에서 재개하여 처리"""
        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint["processed_ids"])
        
        # 처리할 문서 필터링
        remaining_docs = [
            doc for doc in documents
            if str(doc[id_field]) not in processed_ids
        ]
        
        print(f"Total: {len(documents)}, Already processed: {len(processed_ids)}, "
              f"Remaining: {len(remaining_docs)}")
        
        if not remaining_docs:
            print("All documents already processed!")
            return
        
        # 기존 임베딩 로드
        if Path(output_path).exists():
            existing_embeddings = np.load(output_path)
        else:
            existing_embeddings = np.zeros((0, self.generator.embedding_dim))
        
        # 새 문서 처리
        texts = [doc.get('text', doc.get('content', '')) for doc in remaining_docs]
        new_embeddings = self.generator.encode_batch(texts)
        
        # 합치기
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        
        # 저장
        np.save(output_path, all_embeddings)
        
        # 체크포인트 업데이트
        processed_ids.update(str(doc[id_field]) for doc in remaining_docs)
        self._save_checkpoint({
            "processed_ids": processed_ids,
            "embeddings_file": output_path
        })
        
        print(f"Saved {len(all_embeddings)} embeddings to {output_path}")


class EmbeddingQualityChecker:
    """
    임베딩 품질 검증
    
    Example:
        checker = EmbeddingQualityChecker(generator)
        report = checker.evaluate(similar_pairs, dissimilar_pairs)
    """
    
    def __init__(self, generator: EmbeddingGenerator):
        self.generator = generator
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def evaluate(
        self,
        similar_pairs: List[tuple],
        dissimilar_pairs: List[tuple]
    ) -> Dict[str, float]:
        """
        임베딩 품질 평가
        
        Args:
            similar_pairs: 유사한 텍스트 쌍 리스트 [(text1, text2), ...]
            dissimilar_pairs: 유사하지 않은 텍스트 쌍 리스트
        
        Returns:
            평가 결과 딕셔너리
        """
        similar_scores = []
        dissimilar_scores = []
        
        print("Evaluating similar pairs...")
        for text1, text2 in tqdm(similar_pairs):
            emb1 = self.generator.encode(text1)[0]
            emb2 = self.generator.encode(text2)[0]
            similar_scores.append(self.cosine_similarity(emb1, emb2))
        
        print("Evaluating dissimilar pairs...")
        for text1, text2 in tqdm(dissimilar_pairs):
            emb1 = self.generator.encode(text1)[0]
            emb2 = self.generator.encode(text2)[0]
            dissimilar_scores.append(self.cosine_similarity(emb1, emb2))
        
        similar_avg = np.mean(similar_scores)
        dissimilar_avg = np.mean(dissimilar_scores)
        
        return {
            "similar_avg": similar_avg,
            "similar_std": np.std(similar_scores),
            "similar_min": np.min(similar_scores),
            "similar_max": np.max(similar_scores),
            "dissimilar_avg": dissimilar_avg,
            "dissimilar_std": np.std(dissimilar_scores),
            "dissimilar_min": np.min(dissimilar_scores),
            "dissimilar_max": np.max(dissimilar_scores),
            "separation": similar_avg - dissimilar_avg,
        }


if __name__ == "__main__":
    print("Embedding Generator 모듈 로드 완료")
    
    if HAS_TRANSFORMERS:
        # 간단한 테스트
        generator = EmbeddingGenerator(batch_size=8)
        
        test_texts = [
            "민법 제1조는 민사에 관한 기본법입니다.",
            "형법은 범죄와 형벌에 관한 법률입니다.",
        ]
        
        embeddings = generator.encode(test_texts)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Cosine similarity: {np.dot(embeddings[0], embeddings[1]):.4f}")
