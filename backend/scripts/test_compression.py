"""LLMLingua-2 컨텍스트 압축 테스트 스크립트.

실제 판례/법령 원문으로 압축 품질과 성능을 검증한다.
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag.compression import (
    COLUMN_COMPRESSION_POLICIES,
    ContextCompressor,
    ColumnCompressionPolicy,
    FORCE_TOKENS,
)


# ---------------------------------------------------------------------------
# 테스트 데이터 (실제 법률 문서 샘플)
# ---------------------------------------------------------------------------

SAMPLE_RULING = """1. 피고는 원고에게 금 50,000,000원 및 이에 대하여 2023. 5. 15.부터 이 사건 판결 선고일까지는 연 5%의, 그 다음 날부터 다 갚는 날까지는 연 12%의 각 비율로 계산한 돈을 지급하라.
2. 소송비용은 피고가 부담한다.
3. 제1항은 가집행할 수 있다."""

SAMPLE_REASONING = """불법행위로 인한 손해배상책임의 성립 여부에 관하여 살펴본다.

민법 제750조는 "고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다"고 규정하고 있다. 여기서 불법행위가 성립하기 위해서는 가해자의 고의 또는 과실에 의한 위법행위가 있어야 하고, 그로 인하여 피해자에게 손해가 발생하여야 하며, 가해행위와 손해 사이에 상당인과관계가 있어야 한다(대법원 2020. 6. 25. 선고 2019다271051 판결 참조).

이 사건에서 갑 제1호증 내지 갑 제15호증의 각 기재 및 증인 김○○의 증언에 변론 전체의 취지를 종합하면, 다음과 같은 사실을 인정할 수 있다.

① 피고는 2023. 3. 10.경 원고 소유의 건물에 대한 철거공사를 시행하면서 안전조치를 취하지 아니하였다.
② 이로 인하여 공사 현장 인근에 있던 원고의 차량이 파손되었고, 원고는 차량 수리비 30,000,000원 및 대차비용 5,000,000원의 손해를 입었다.
③ 또한 원고는 위 사고로 인하여 경추부 염좌 등의 상해를 입고 2주간 치료를 받았으며, 치료비 10,000,000원 및 위자료 5,000,000원의 손해를 입었다.

위 인정사실에 의하면, 피고가 안전조치를 취하지 아니한 채 철거공사를 시행한 것은 불법행위에 해당하고, 이로 인하여 원고에게 합계 50,000,000원의 손해가 발생하였다고 봄이 상당하므로, 피고는 원고에게 위 손해를 배상할 의무가 있다.

피고는, 원고가 공사 현장 근처에 차량을 주차한 것 자체가 과실에 해당한다고 주장하나, 원고가 차량을 주차한 장소는 적법한 주차구역이었고, 피고의 공사로 인한 위험이 예견 가능한 범위를 초과하였으므로, 피고의 위 주장은 이유 없다.

그렇다면 원고의 청구는 이유 있으므로 이를 인용하기로 하여 주문과 같이 판결한다."""

SAMPLE_LAW_CONTENT = """제750조(불법행위의 내용) 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.

제751조(재산 이외의 손해의 배상) ①타인의 신체, 자유 또는 명예를 해하거나 기타 정신상고통을 가한 자는 재산 이외의 손해에 대하여도 배상할 책임이 있다.
②법원은 전항의 손해배상을 정기금채무로 지급할 것을 명할 수 있고 그 이행을 확보하기 위하여 상당한 담보의 제공을 명할 수 있다.

제752조(생명침해로 인한 위자료) 타인의 생명을 해한 자는 피해자의 직계존속, 직계비속 및 배우자에 대하여는 재산상의 손해없는 경우에도 손해배상의 책임이 있다.

제753조(미성년자의 책임능력) 미성년자가 타인에게 손해를 가한 경우에 그 행위의 책임을 변식할 지능이 없는 때에는 배상의 책임이 없다.

제754조(심신상실자의 책임능력) 심신상실 중에 타인에게 손해를 가한 자는 배상의 책임이 없다. 그러나 고의 또는 과실로 인하여 심신상실을 초래한 때에는 그러하지 아니하다.

제755조(감독자의 책임) ①다른 자에게 손해를 가한 자가 제753조 또는 제754조에 의하여 책임이 없는 경우에는 그를 감독할 법정의무있는 자가 그 손해를 배상할 책임이 있다. 다만, 감독의무를 게을리하지 아니한 때에는 그러하지 아니하다.
②감독의무자에 갈음하여 무능력자를 감독하는 자도 전항의 책임이 있다.

제756조(사용자의 배상책임) ①타인을 사용하여 어느 사무에 종사하게 한 자는 피용자가 그 사무집행에 관하여 제삼자에게 가한 손해를 배상할 책임이 있다. 그러나 사용자가 피용자의 선임 및 그 사무감독에 상당한 주의를 한 때 또는 상당한 주의를 하여도 손해가 있을 경우에는 그러하지 아니하다.
②사용자에 갈음하여 그 사무를 감독하는 자도 전항의 책임이 있다.
③전2항의 경우에 사용자 또는 감독자는 피용자에 대하여 구상권을 행사할 수 있다."""


def run_test() -> None:
    """압축 테스트 실행."""
    print("=" * 60)
    print("LLMLingua-2 컨텍스트 압축 테스트")
    print("=" * 60)

    # 1. 모델 로딩
    print("\n[1] 모델 로딩...")
    compressor = ContextCompressor()
    load_start = time.monotonic()
    compressor._get_compressor()  # 명시적 로딩
    load_time = time.monotonic() - load_start
    print(f"    모델 로딩 시간: {load_time:.1f}s")

    # 2. 판례 테스트 (ruling + reasoning)
    print("\n[2] 판례 압축 테스트")
    print("-" * 40)
    precedent_fields = {
        "ruling": SAMPLE_RULING,
        "reasoning": SAMPLE_REASONING,
    }
    print(f"    원문: ruling {len(SAMPLE_RULING)}자, reasoning {len(SAMPLE_REASONING)}자")
    print(f"    총 원문: {sum(len(v) for v in precedent_fields.values())}자")

    start = time.monotonic()
    compressed = compressor.compress_document_fields(precedent_fields)
    elapsed = (time.monotonic() - start) * 1000
    print(f"    압축 후: ruling {len(compressed['ruling'])}자, reasoning {len(compressed['reasoning'])}자")
    print(f"    총 압축: {sum(len(v) for v in compressed.values())}자")
    print(f"    압축률: {sum(len(v) for v in compressed.values()) / sum(len(v) for v in precedent_fields.values()):.1%}")
    print(f"    소요 시간: {elapsed:.0f}ms")

    # ruling 보존 확인
    if compressed["ruling"] == SAMPLE_RULING:
        print("    ✓ ruling 원문 보존 확인")
    else:
        print("    ✗ ruling이 변경됨!")

    # 핵심 토큰 보존 확인
    key_terms = ["민법 제750조", "불법행위", "손해배상", "원고", "피고", "대법원"]
    preserved = [t for t in key_terms if t in compressed["reasoning"]]
    missing = [t for t in key_terms if t not in compressed["reasoning"]]
    print(f"    핵심 용어 보존: {len(preserved)}/{len(key_terms)} ({', '.join(preserved)})")
    if missing:
        print(f"    누락된 용어: {', '.join(missing)}")

    print("\n    [압축된 reasoning 미리보기 (첫 300자)]:")
    print(f"    {compressed['reasoning'][:300]}...")

    # 3. 법령 테스트 (content)
    print("\n[3] 법령 압축 테스트")
    print("-" * 40)
    law_fields = {"content": SAMPLE_LAW_CONTENT}
    print(f"    원문: {len(SAMPLE_LAW_CONTENT)}자")

    start = time.monotonic()
    compressed_law = compressor.compress_document_fields(law_fields)
    elapsed = (time.monotonic() - start) * 1000
    print(f"    압축 후: {len(compressed_law['content'])}자")
    print(f"    압축률: {len(compressed_law['content']) / len(SAMPLE_LAW_CONTENT):.1%}")
    print(f"    소요 시간: {elapsed:.0f}ms")

    # 법조문 번호 보존 확인
    article_refs = ["제750조", "제751조", "제752조", "제755조", "제756조"]
    preserved_refs = [r for r in article_refs if r in compressed_law["content"]]
    print(f"    법조문 번호 보존: {len(preserved_refs)}/{len(article_refs)}")

    print("\n    [압축된 법령 미리보기 (첫 300자)]:")
    print(f"    {compressed_law['content'][:300]}...")

    # 4. 문서 리스트 일괄 압축 테스트
    print("\n[4] 문서 리스트 일괄 압축 테스트")
    print("-" * 40)
    documents = [
        {
            "content_fields": {"ruling": SAMPLE_RULING, "reasoning": SAMPLE_REASONING},
            "content": f"{SAMPLE_RULING}\n\n{SAMPLE_REASONING}",
            "metadata": {"doc_id": "test_1", "data_type": "판례"},
        },
        {
            "content_fields": {"content": SAMPLE_LAW_CONTENT},
            "content": SAMPLE_LAW_CONTENT,
            "metadata": {"doc_id": "test_2", "data_type": "법령"},
        },
    ]

    metrics = compressor.compress_documents(documents)
    print(f"    전체: {metrics['before_chars']}자 → {metrics['after_chars']}자")
    print(f"    압축률: {metrics['compression_ratio']:.1%}")
    print(f"    소요 시간: {metrics['time_ms']:.0f}ms")

    # content 재조인 확인
    for doc in documents:
        rejoined = "\n\n".join(doc["content_fields"].values())
        assert doc["content"] == rejoined, "content 재조인 불일치!"
    print("    ✓ content 재조인 정합성 확인")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
