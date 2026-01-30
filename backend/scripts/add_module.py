#!/usr/bin/env python3
"""
새 모듈 추가 스크립트

사용법:
    python scripts/add_module.py <module_name> "<module_description>"

예시:
    python scripts/add_module.py document_generator "법률 문서 자동 생성"
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_MODULES = PROJECT_ROOT / "backend" / "app" / "modules"
FRONTEND_APP = PROJECT_ROOT / "frontend" / "src" / "app"
FRONTEND_FEATURES = PROJECT_ROOT / "frontend" / "src" / "features"
FRONTEND_MODULES_TS = PROJECT_ROOT / "frontend" / "src" / "lib" / "modules.ts"


def to_kebab_case(name: str) -> str:
    """snake_case -> kebab-case"""
    return name.replace("_", "-")


def to_pascal_case(name: str) -> str:
    """snake_case -> PascalCase"""
    return "".join(word.capitalize() for word in name.split("_"))


def to_camel_case(name: str) -> str:
    """snake_case -> camelCase"""
    pascal = to_pascal_case(name)
    return pascal[0].lower() + pascal[1:]


def create_backend_module(module_name: str, description: str):
    """백엔드 모듈 생성"""
    module_path = BACKEND_MODULES / module_name

    if module_path.exists():
        print(f"[Backend] 모듈이 이미 존재합니다: {module_name}")
        return False

    # 폴더 구조 생성
    dirs = ["", "router", "service", "schema", "model"]
    for d in dirs:
        (module_path / d).mkdir(parents=True, exist_ok=True)
        (module_path / d / "__init__.py").touch()

    # router/__init__.py 내용 작성
    router_content = f'''"""
{description} 모듈
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_{module_name}_root():
    """모듈 루트 엔드포인트"""
    return {{"message": "{description}"}}
'''

    (module_path / "router" / "__init__.py").write_text(router_content)
    print(f"[Backend] 모듈 생성 완료: {module_name}")
    return True


def create_frontend_module(module_name: str, description: str):
    """프론트엔드 모듈 생성"""
    kebab_name = to_kebab_case(module_name)
    camel_name = to_camel_case(module_name)

    # 1. app 페이지 생성
    app_path = FRONTEND_APP / kebab_name
    if not app_path.exists():
        app_path.mkdir(parents=True, exist_ok=True)
        page_content = f''''use client'

export default function {to_pascal_case(module_name)}Page() {{
  return (
    <div className="min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-6">{description}</h1>
      <p className="text-gray-600">TODO: 기능 구현</p>
    </div>
  )
}}
'''
        (app_path / "page.tsx").write_text(page_content)
        print(f"[Frontend] 페이지 생성: app/{kebab_name}/page.tsx")

    # 2. features 서비스 생성
    features_path = FRONTEND_FEATURES / kebab_name / "services"
    if not features_path.exists():
        features_path.mkdir(parents=True, exist_ok=True)

        # components, hooks, types 폴더도 생성
        for folder in ["components", "hooks", "types"]:
            (FRONTEND_FEATURES / kebab_name / folder).mkdir(exist_ok=True)

        service_content = f'''import {{ api, endpoints }} from '@/lib/api'

export const {camel_name}Service = {{
  // TODO: API 메서드 구현
  getData: async () => {{
    const response = await api.get(`${{endpoints.{camel_name}}}/`)
    return response.data
  }},
}}
'''
        (features_path / "index.ts").write_text(service_content)
        print(f"[Frontend] 서비스 생성: features/{kebab_name}/services/index.ts")

    # 3. modules.ts에 추가
    update_modules_ts(module_name, kebab_name, description)

    # 4. api.ts endpoints에 추가
    update_api_ts(kebab_name, camel_name)

    return True


def update_modules_ts(module_name: str, kebab_name: str, description: str):
    """modules.ts에 새 모듈 추가"""
    content = FRONTEND_MODULES_TS.read_text()

    # 이미 존재하는지 확인
    if f"id: '{kebab_name}'" in content:
        print(f"[Frontend] modules.ts에 이미 존재: {kebab_name}")
        return

    # 새 모듈 항목
    new_module = f'''  {{
    id: '{kebab_name}',
    name: '{description}',
    description: '{description} 기능',
    href: '/{kebab_name}',
    icon: '📄',
    enabled: true,
  }},
]'''

    # 마지막 ] 앞에 삽입
    content = content.replace("\n]", f",\n{new_module}")
    content = content.replace(",,", ",")  # 중복 쉼표 제거

    FRONTEND_MODULES_TS.write_text(content)
    print("[Frontend] modules.ts 업데이트 완료")


def update_api_ts(kebab_name: str, camel_name: str):
    """api.ts endpoints에 새 모듈 추가"""
    api_ts_path = PROJECT_ROOT / "frontend" / "src" / "lib" / "api.ts"
    content = api_ts_path.read_text()

    # 이미 존재하는지 확인
    if f"{camel_name}:" in content:
        print(f"[Frontend] api.ts에 이미 존재: {camel_name}")
        return

    # endpoints 객체에 추가
    new_endpoint = f"  {camel_name}: '/{kebab_name}',\n}}"

    # 마지막 } 찾아서 교체
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "smallClaims:" in line or (line.strip() == "}" and i > 5):
            if lines[i].strip() == "}":
                lines[i] = new_endpoint
            else:
                lines.insert(i + 1, f"  {camel_name}: '/{kebab_name}',")
            break

    api_ts_path.write_text("\n".join(lines))
    print("[Frontend] api.ts 업데이트 완료")


def remove_module(module_name: str):
    """모듈 삭제"""
    import shutil

    kebab_name = to_kebab_case(module_name)

    # Backend
    backend_path = BACKEND_MODULES / module_name
    if backend_path.exists():
        shutil.rmtree(backend_path)
        print(f"[Backend] 모듈 삭제: {module_name}")

    # Frontend app
    app_path = FRONTEND_APP / kebab_name
    if app_path.exists():
        shutil.rmtree(app_path)
        print(f"[Frontend] 페이지 삭제: {kebab_name}")

    # Frontend features
    features_path = FRONTEND_FEATURES / kebab_name
    if features_path.exists():
        shutil.rmtree(features_path)
        print(f"[Frontend] 기능 삭제: {kebab_name}")

    print("\n[주의] modules.ts와 api.ts에서 수동으로 해당 항목을 삭제하세요.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "remove" and len(sys.argv) >= 3:
        module_name = sys.argv[2]
        remove_module(module_name)
    elif len(sys.argv) >= 3:
        module_name = sys.argv[1]
        description = sys.argv[2]

        print(f"\n=== 모듈 추가: {module_name} ===\n")
        create_backend_module(module_name, description)
        create_frontend_module(module_name, description)
        print("\n완료! 서버를 재시작하세요.")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
