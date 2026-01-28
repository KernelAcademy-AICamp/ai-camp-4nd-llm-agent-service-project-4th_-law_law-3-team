#!/usr/bin/env python3
"""
판례 임베딩 테스트 스크립트

수정된 runpod_lancedb_embeddings.py의 동작을 검증합니다.
메모리 최적화 적용: 단계별 테스트 + GC 관리
"""

import gc
import json
import sys
import tempfile
from pathlib import Path

# 테스트용 샘플 판례 데이터 (3건)
SAMPLE_PRECEDENTS = [
    {
        "판례정보일련번호": "TEST001",
        "사건명": "손해배상청구사건",
        "사건번호": "84나3990",
        "선고일자": "19860115",
        "법원명": "서울고법",
        "사건종류명": "민사",
        "판시사항": "수련의에게 마취를 담당케 하여 의료사고가 발생한 경우",
        "판결요지": "수술당일 환자측으로부터 집도의와 마취담당의를 특정한 신청을 받고 승낙하고서도 수련의에게 담당하게 하여 의료사고가 발생하였다면 병원측의 과실로 추정한다.",
        "참조조문": "민법 제750조, 제756조",
        "참조판례": "",
    },
    {
        "판례정보일련번호": "TEST002",
        "사건명": "부당이득금반환청구사건",
        "사건번호": "2020다12345",
        "선고일자": "20210315",
        "법원명": "대법원",
        "사건종류명": "민사",
        "판시사항": "계약 해제 후 원상회복의무의 범위",
        "판결요지": "계약이 해제되면 각 당사자는 원상회복의무가 있고, 금전 반환 시 이자를 가산하여야 한다.",
        "참조조문": "민법 제548조",
        "참조판례": "대법원 2019다54321 판결",
    },
    {
        "판례정보일련번호": "TEST003",
        "사건명": "강도상해",
        "사건번호": "2021도9876",
        "선고일자": "20220520",
        "법원명": "대법원",
        "사건종류명": "형사",
        "판시사항": "강도죄에 있어서 폭행·협박의 정도",
        "판결요지": "강도죄의 폭행·협박은 상대방의 반항을 억압할 정도의 것이어야 한다.",
        "참조조문": "형법 제333조",
        "참조판례": "",
    },
]


def test_chunking_only():
    """청킹 로직만 테스트 (모델 로드 없음 - 메모리 안전)"""
    print("=" * 60)
    print("[1/3] 청킹 로직 테스트 (모델 로드 없음)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from runpod_lancedb_embeddings import (
            chunk_precedent_text,
            PrecedentChunkConfig,
        )
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    chunk_config = PrecedentChunkConfig()
    total_chunks = 0

    for i, prec in enumerate(SAMPLE_PRECEDENTS):
        parts = []
        case_name = prec.get("사건명", "")
        if case_name:
            parts.append(f"[{case_name}]")

        summary = prec.get("판시사항", "")
        if summary:
            parts.append(summary)

        judgment = prec.get("판결요지", "")
        if judgment:
            parts.append(judgment)

        text = "\n".join(parts)
        chunks = chunk_precedent_text(text, chunk_config)
        total_chunks += len(chunks)
        print(f"  Precedent {i+1}: {len(chunks)} chunk(s), text_len={len(text)}")

    print(f"[OK] Total chunks: {total_chunks}")
    gc.collect()
    return True


def test_device_detection():
    """디바이스 감지 및 최적 설정 테스트"""
    print("\n" + "=" * 60)
    print("[2/3] 디바이스 감지 테스트")
    print("=" * 60)

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from runpod_lancedb_embeddings import (
            get_device_info,
            get_optimal_config,
        )
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    print(f"  Device: {device_info.device}")
    print(f"  Name: {device_info.name}")
    print(f"  Memory: {device_info.vram_gb:.1f}GB")
    print(f"  Optimal batch_size: {optimal.batch_size}")
    print(f"  Optimal num_workers: {optimal.num_workers}")
    print(f"  GC interval: {optimal.gc_interval}")

    print("[OK] Device detection passed")
    gc.collect()
    return True, device_info, optimal


def test_embedding_with_cleanup(device_info, optimal):
    """임베딩 테스트 (메모리 관리 적용)"""
    print("\n" + "=" * 60)
    print("[3/3] 임베딩 생성 테스트 (메모리 관리 적용)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        from runpod_lancedb_embeddings import (
            get_embedding_model,
            create_embeddings,
            clear_model_cache,
            print_memory_status,
        )
        import torch
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    # 메모리 상태 확인
    print_memory_status()

    # 모델 로드 (CPU 환경에서는 배치 크기 최소화)
    print(f"\n  Loading model on {device_info.device}...")
    try:
        model = get_embedding_model(device_info.device)
        print(f"  [OK] Model loaded! Dimension: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"  [ERROR] Model load failed: {e}")
        return False

    # GC 실행
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 최소한의 테스트 텍스트
    test_texts = ["[판례] 손해배상청구 테스트"]

    print(f"\n  Creating embedding for {len(test_texts)} text(s)...")
    try:
        embeddings = create_embeddings(test_texts, device_info.device)
        print(f"  [OK] Embedding created!")
        print(f"       - Dimension: {len(embeddings[0])}")
        print(f"       - First 3 values: {embeddings[0][:3]}")

        # 임베딩 메모리 즉시 해제
        del embeddings
        gc.collect()

    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 모델 캐시 정리
        print("\n  Clearing model cache...")
        clear_model_cache()
        print_memory_status()

    print("[OK] Embedding test passed")
    return True


def test_streaming_skip():
    """스트리밍 카운트 스킵 로직 테스트 (코드 검증)"""
    print("\n" + "=" * 60)
    print("[0/3] 코드 패턴 검증 (스트리밍 최적화)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    target_file = script_dir / "runpod_lancedb_embeddings.py"

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    checks = [
        ("total_count = None", "개수 세기 스킵"),
        ("Skipping count", "스킵 로그"),
        ("gc.collect()", "GC 호출"),
        ("get_optimal_config", "디바이스별 최적화"),
    ]

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  [OK] {description}")
        else:
            print(f"  [FAIL] {description}: '{pattern}' 없음")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("판례 임베딩 테스트 (메모리 최적화 버전)")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print()

    results = {}

    # 0. 코드 패턴 검증
    results["code_check"] = test_streaming_skip()

    # 1. 청킹 테스트 (메모리 안전)
    results["chunking"] = test_chunking_only()

    # 2. 디바이스 감지 테스트
    device_result = test_device_detection()
    if isinstance(device_result, tuple):
        results["device"] = device_result[0]
        device_info = device_result[1]
        optimal = device_result[2]
    else:
        results["device"] = device_result
        device_info = None
        optimal = None

    # 3. 임베딩 테스트 (선택적 - 메모리 충분할 때만)
    if device_info and device_info.vram_gb >= 6:
        results["embedding"] = test_embedding_with_cleanup(device_info, optimal)
    else:
        print("\n" + "=" * 60)
        print("[3/3] 임베딩 테스트 SKIP (메모리 부족)")
        print("=" * 60)
        if device_info:
            print(f"  Available: {device_info.vram_gb:.1f}GB, Required: 6GB+")
        print("  RunPod에서 실행하세요.")
        results["embedding"] = None

    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name}: {status}")

    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"\n실패한 테스트: {failed}")
        sys.exit(1)
    else:
        print("\n모든 테스트 통과 (또는 스킵)!")
        sys.exit(0)
