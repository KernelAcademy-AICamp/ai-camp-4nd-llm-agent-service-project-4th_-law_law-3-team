#!/usr/bin/env python3
"""
데이터 로드 전 환경 검증 스크립트

필수 조건(Python 버전, 환경변수, MeCab, Docker, 디스크 등)을
자동으로 검증하고, 실패 항목에 대한 해결 방법을 출력한다.

Usage:
    uv run python scripts/check_environment.py             # 전체 검증
    uv run python scripts/check_environment.py --step db    # PostgreSQL 관련만
    uv run python scripts/check_environment.py --step vector # LanceDB/임베딩 관련만
    uv run python scripts/check_environment.py --step neo4j  # Neo4j 관련만
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_backend_root = Path(__file__).parent.parent
_project_root = _backend_root.parent

# 색상 코드
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _pass(msg: str) -> bool:
    print(f"  {GREEN}✅{RESET} {msg}")
    return True


def _fail(msg: str, fix: str) -> bool:
    print(f"  {RED}❌{RESET} {msg}")
    print(f"     → {fix}")
    return False


def _warn(msg: str, fix: str) -> None:
    print(f"  {YELLOW}⚠️{RESET}  {msg}")
    print(f"     → {fix}")


def check_python_version() -> bool:
    """Python 3.11+ 확인"""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return _pass(f"Python {version_str}")
    return _fail(
        f"Python {version_str} (3.11+ 필요)",
        "Python 3.11 이상을 설치하세요",
    )


def check_env_file() -> bool:
    """backend/.env 파일 존재 확인"""
    env_path = _backend_root / ".env"
    if env_path.is_file():
        return _pass("backend/.env 존재")
    return _fail(
        "backend/.env 파일 없음",
        "cp backend/.env.example backend/.env 후 값 수정",
    )


def check_env_vars(step: str) -> bool:
    """필수 환경변수 확인"""
    # .env 파일에서 로드 시도 (dotenv 없이 간단히)
    env_path = _backend_root / ".env"
    env_values: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env_values[key.strip()] = val.strip()

    # 환경변수도 합치기 (환경변수가 .env보다 우선)
    merged = {**env_values, **os.environ}

    required: list[tuple[str, str]] = []
    if step in ("db", "all"):
        required.append(("DATABASE_URL", "backend/.env에 DATABASE_URL 설정 필요"))
    if step in ("neo4j", "all"):
        required.append(("NEO4J_PASSWORD", "backend/.env에 NEO4J_PASSWORD 설정 필요"))

    ok = True
    for var, fix in required:
        val = merged.get(var, "")
        if val and val not in ("change_me_to_match_docker_compose", "password"):
            _pass(f"{var} 설정됨")
        elif val:
            _warn(f"{var} = 기본값 사용 중", f"프로덕션 환경에서는 {fix}")
            # 경고이므로 ok 유지
        else:
            ok = _fail(f"{var} 미설정", fix) and ok
    return ok


def check_data_files() -> tuple[bool, int]:
    """핵심 JSON 데이터 파일 존재 확인"""
    data_dir = _project_root / "data"
    core_files = [
        "law_v3.json",
        "precedents_v2.json",
    ]

    ok = True
    warnings = 0
    if not data_dir.is_dir():
        return _fail("data/ 폴더 없음", "프로젝트 루트에 data/ 폴더 생성 및 JSON 배치"), 0

    for fname in core_files:
        fpath = data_dir / fname
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            _pass(f"data/{fname} ({size_mb:.0f}MB)")
        else:
            _warn(f"data/{fname} 없음", "팀에서 데이터 파일을 받아 data/ 폴더에 배치")
            warnings += 1

    return ok, warnings


def check_mecab() -> bool:
    """MeCab 시스템 패키지 확인"""
    mecab_path = shutil.which("mecab")
    if mecab_path:
        try:
            result = subprocess.run(
                ["mecab", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip()
            return _pass(f"MeCab 설치됨 ({version})")
        except (subprocess.TimeoutExpired, OSError):
            return _pass("MeCab 설치됨 (버전 확인 실패)")
    return _fail(
        "MeCab 미설치",
        "sudo apt install mecab libmecab-dev mecab-ko-dic (Ubuntu) "
        "또는 brew install mecab mecab-ko-dic (macOS)",
    )


def check_mecab_python() -> bool:
    """MeCab Python 바인딩 확인"""
    try:
        import MeCab  # noqa: F401

        return _pass("MeCab Python 바인딩 (mecab-python3)")
    except ImportError:
        return _fail(
            "mecab-python3 미설치",
            "uv add mecab-python3 또는 uv pip install mecab-python3",
        )


def check_pytorch() -> tuple[bool, bool]:
    """PyTorch 설치 확인 (경고 수준)"""
    try:
        import torch

        device = "CPU"
        if torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        _pass(f"PyTorch {torch.__version__} ({device})")
        return True, False
    except ImportError:
        _warn(
            "PyTorch 미설치",
            "uv pip install torch (임베딩 생성 시 필요, 서버 실행에는 불필요)",
        )
        return True, True  # 경고이므로 패스, 경고 플래그


def check_docker_container(name: str, label: str) -> bool:
    """Docker 컨테이너 실행 상태 확인"""
    docker_cmd = _find_docker_cmd()
    if not docker_cmd:
        return _fail(
            f"{label} 확인 불가 (docker 미설치)",
            "Docker Desktop 설치 또는 WSL2에서 docker.exe 확인",
        )

    try:
        result = subprocess.run(
            [docker_cmd, "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running = result.stdout.strip().splitlines()
        if name in running:
            return _pass(f"{label} 컨테이너 실행 중 ({name})")
        return _fail(
            f"{label} 컨테이너 미실행 ({name})",
            f"docker compose up -d {label.lower().replace(' ', '')}",
        )
    except (subprocess.TimeoutExpired, OSError):
        return _fail(
            f"{label} 확인 실패",
            "Docker 데몬이 실행 중인지 확인하세요",
        )


def check_embedding_model() -> bool:
    """임베딩 모델 캐시 확인"""
    model_dir = _backend_root / "data" / "models"
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return _pass("임베딩 모델 캐시 (data/models/)")
    return _fail(
        "임베딩 모델 미다운로드",
        "uv run python scripts/download_models.py",
    )


def check_alembic_migration() -> bool:
    """Alembic 마이그레이션 상태 확인"""
    try:
        result = subprocess.run(
            ["uv", "run", "alembic", "current"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(_backend_root),
        )
        output = result.stdout.strip()
        if "head" in output:
            return _pass("Alembic 마이그레이션 최신 (head)")
        if output:
            _warn(
                f"Alembic 마이그레이션 head 아님: {output[:80]}",
                "uv run alembic upgrade head",
            )
            return True  # 경고 수준
        return _fail(
            "Alembic 마이그레이션 상태 확인 불가",
            "uv run alembic upgrade head",
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return _fail(
            "Alembic 실행 실패 (DB 연결 또는 uv 문제)",
            "DB 컨테이너 실행 확인 후 uv run alembic upgrade head",
        )


def check_disk_space() -> bool:
    """디스크 공간 확인 (50GB 이상 권장)"""
    try:
        total, used, free = shutil.disk_usage(str(_project_root))
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        if free_gb >= 50:
            return _pass(f"디스크 여유: {free_gb:.1f}GB / {total_gb:.0f}GB")
        _warn(
            f"디스크 여유: {free_gb:.1f}GB (50GB 이상 권장)",
            "불필요한 파일 정리 또는 디스크 확장",
        )
        return True  # 경고 수준
    except OSError:
        _warn("디스크 공간 확인 실패", "수동으로 df -h 확인")
        return True


def _find_docker_cmd() -> str | None:
    """Docker 명령어 감지 (WSL2 호환)"""
    if shutil.which("docker"):
        return "docker"
    if shutil.which("docker.exe"):
        return "docker.exe"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터 로드 전 환경 검증")
    parser.add_argument(
        "--step",
        choices=["db", "vector", "neo4j", "all"],
        default="all",
        help="검증 범위 (기본: all)",
    )
    args = parser.parse_args()
    step: str = args.step

    print()
    print("=" * 50)
    print("  환경 검증")
    print("=" * 50)
    print()

    passed = 0
    failed = 0
    warnings = 0

    # --- 공통 검증 ---
    print("[공통]")
    if check_python_version():
        passed += 1
    else:
        failed += 1

    if check_env_file():
        passed += 1
    else:
        failed += 1

    if check_env_vars(step):
        passed += 1
    else:
        failed += 1

    if check_disk_space():
        passed += 1
    else:
        failed += 1

    # --- DB 관련 ---
    if step in ("db", "all"):
        print()
        print("[PostgreSQL]")
        if check_docker_container("law-platform-db", "PostgreSQL"):
            passed += 1
        else:
            failed += 1

        if check_alembic_migration():
            passed += 1
        else:
            failed += 1

        ok, w = check_data_files()
        if ok:
            passed += 1
        else:
            failed += 1
        warnings += w

    # --- MeCab ---
    if step in ("db", "all"):
        print()
        print("[MeCab]")
        if check_mecab():
            passed += 1
        else:
            failed += 1
        if check_mecab_python():
            passed += 1
        else:
            failed += 1

    # --- Vector/임베딩 ---
    if step in ("vector", "all"):
        print()
        print("[벡터 DB / 임베딩]")
        if check_embedding_model():
            passed += 1
        else:
            failed += 1

        pt_ok, pt_warn = check_pytorch()
        if pt_ok:
            passed += 1
        else:
            failed += 1
        if pt_warn:
            warnings += 1

    # --- Neo4j ---
    if step in ("neo4j", "all"):
        print()
        print("[Neo4j]")
        if check_docker_container("neo4j-law-graph", "Neo4j"):
            passed += 1
        else:
            failed += 1

    # --- 요약 ---
    total = passed + failed
    print()
    print("=" * 50)
    print(f"  총: {passed}/{total} 통과", end="")
    if failed > 0:
        print(f", {RED}{failed} 실패{RESET}", end="")
    if warnings > 0:
        print(f", {YELLOW}{warnings} 경고{RESET}", end="")
    print()
    print("=" * 50)
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
