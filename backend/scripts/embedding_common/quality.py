"""
임베딩 품질 검증 유틸리티

유사/비유사 문서 쌍의 코사인 유사도를 측정하여
임베딩 품질을 평가합니다.
"""

from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

from scripts.embedding_common.device import get_device
from scripts.embedding_common.model import create_embeddings


class EmbeddingQualityChecker:
    """
    임베딩 품질 검증 유틸리티

    유사/비유사 문서 쌍의 코사인 유사도를 측정하여
    임베딩 품질을 평가.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or get_device()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 코사인 유사도 계산"""
        embeddings = create_embeddings([text1, text2])
        emb1 = torch.tensor(embeddings[0])
        emb2 = torch.tensor(embeddings[1])

        similarity: float = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

        return similarity

    def evaluate(
        self,
        similar_pairs: list[tuple[str, str]],
        dissimilar_pairs: list[tuple[str, str]],
    ) -> dict[str, float | str]:
        """
        유사/비유사 쌍으로 임베딩 품질 평가

        Args:
            similar_pairs: 유사해야 하는 텍스트 쌍 리스트 [(text1, text2), ...]
            dissimilar_pairs: 비유사해야 하는 텍스트 쌍 리스트

        Returns:
            평가 결과 딕셔너리
        """
        similar_scores: list[float] = []
        dissimilar_scores: list[float] = []

        print("[INFO] Evaluating similar pairs...")
        for text1, text2 in tqdm(similar_pairs, desc="Similar"):
            score = self.compute_similarity(text1, text2)
            similar_scores.append(score)

        print("[INFO] Evaluating dissimilar pairs...")
        for text1, text2 in tqdm(dissimilar_pairs, desc="Dissimilar"):
            score = self.compute_similarity(text1, text2)
            dissimilar_scores.append(score)

        similar_avg = (
            sum(similar_scores) / len(similar_scores) if similar_scores else 0.0
        )
        dissimilar_avg = (
            sum(dissimilar_scores) / len(dissimilar_scores)
            if dissimilar_scores
            else 0.0
        )
        separation = similar_avg - dissimilar_avg

        if separation > 0.2:
            quality = "good"
        elif separation > 0.1:
            quality = "fair"
        else:
            quality = "poor"

        report: dict[str, float | str] = {
            "similar_avg": similar_avg,
            "similar_min": min(similar_scores) if similar_scores else 0.0,
            "similar_max": max(similar_scores) if similar_scores else 0.0,
            "dissimilar_avg": dissimilar_avg,
            "dissimilar_min": min(dissimilar_scores) if dissimilar_scores else 0.0,
            "dissimilar_max": max(dissimilar_scores) if dissimilar_scores else 0.0,
            "separation": separation,
            "quality": quality,
        }

        print("\n" + "=" * 50)
        print("Embedding Quality Report")
        print("=" * 50)
        print(f"  Similar pairs avg:    {similar_avg:.4f}")
        print(f"  Dissimilar pairs avg: {dissimilar_avg:.4f}")
        print(f"  Separation:           {separation:.4f}")
        print(f"  Quality:              {quality.upper()}")
        print("=" * 50)

        return report

    def quick_test(self) -> dict[str, float | str]:
        """법률 도메인 기본 테스트"""
        similar_pairs: list[tuple[str, str]] = [
            ("손해배상 청구권", "손해배상 청구"),
            ("민법 제750조 불법행위", "민법상 불법행위 책임"),
            ("임대차 계약 해지", "임대차계약의 해지"),
            ("형사소송법 제309조", "형사소송법 309조"),
        ]

        dissimilar_pairs: list[tuple[str, str]] = [
            ("민법 제750조 불법행위", "형법 제250조 살인죄"),
            ("손해배상 청구권", "회사 설립 절차"),
            ("임대차 계약", "특허권 침해"),
            ("형사소송법", "민사집행법"),
        ]

        return self.evaluate(similar_pairs, dissimilar_pairs)
