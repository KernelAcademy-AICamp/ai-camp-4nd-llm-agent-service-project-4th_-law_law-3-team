"""
Upstage Solar API 연결 테스트

실행: uv run python evaluation/tools/test_solar_api.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv()

UPSTAGE_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"


async def test_solar_api():
    """Solar API 연결 테스트"""
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key:
        print("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 UPSTAGE_API_KEY=your_key 추가 필요")
        return False

    print(f"✓ API 키 확인됨: {api_key[:10]}...")

    # 모델 목록
    models_to_test = [
        "solar-pro3-260126",
        "solar-pro3",
    ]

    for model in models_to_test:
        print(f"\n테스트 모델: {model}")
        print("-" * 40)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "안녕하세요. 간단한 테스트입니다. '테스트 성공'이라고만 답해주세요."}
            ],
            "temperature": 0.1,
            "max_tokens": 50,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"→ 요청 전송 중... ({UPSTAGE_API_URL})")
                response = await client.post(
                    UPSTAGE_API_URL,
                    headers=headers,
                    json=payload,
                )

                print(f"→ 응답 코드: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    print(f"✅ 성공!")
                    print(f"   모델: {result.get('model', 'N/A')}")
                    print(f"   응답: {content}")
                    print(f"   토큰: prompt={usage.get('prompt_tokens', 'N/A')}, "
                          f"completion={usage.get('completion_tokens', 'N/A')}")
                    return True
                else:
                    print(f"❌ 실패: {response.status_code}")
                    print(f"   응답: {response.text[:500]}")

        except httpx.TimeoutException:
            print("❌ 타임아웃 (30초 초과)")
        except httpx.RequestError as e:
            print(f"❌ 요청 오류: {e}")
        except Exception as e:
            print(f"❌ 예외 발생: {type(e).__name__}: {e}")

    return False


async def test_question_generation():
    """질문 생성 테스트"""
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        return False

    print("\n" + "=" * 50)
    print("질문 생성 테스트")
    print("=" * 50)

    model = os.getenv("UPSTAGE_MODEL", "solar-pro3-260126")

    test_prompt = """다음 판례의 판시사항을 보고, 일반인이 법률 상담 시 물어볼 만한 자연스러운 질문을 생성하세요.

[판시사항]
타인의 불법행위로 인하여 손해를 입은 피해자가 손해배상을 청구하는 경우, 가해자의 불법행위와 피해자의 손해 사이에 상당인과관계가 있어야 한다.

[사건명]
손해배상청구사건

요구사항:
1. 법률 전문용어 대신 일상적인 표현 사용
2. 구체적인 상황을 가정한 질문
3. 1개의 질문만 생성

출력 형식 (JSON):
{"question": "...", "key_points": ["...", "..."], "category": "민사|형사|행정|헌법|노동|상사|조세|기타", "query_type": "단순조회|개념검색|비교검색|참조추적|시간검색|복합검색"}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"→ 모델: {model}")
            print(f"→ 요청 전송 중...")
            response = await client.post(
                UPSTAGE_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                print(f"✅ 응답 수신!")
                print(f"\n생성된 응답:\n{content}")

                # JSON 파싱 테스트
                import json
                try:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        parsed = json.loads(json_str)
                        print(f"\n✅ JSON 파싱 성공:")
                        print(f"   질문: {parsed.get('question', 'N/A')}")
                        print(f"   키포인트: {parsed.get('key_points', [])}")
                        print(f"   카테고리: {parsed.get('category', 'N/A')}")
                        return True
                except json.JSONDecodeError as e:
                    print(f"\n⚠️ JSON 파싱 실패: {e}")
            else:
                print(f"❌ 실패: {response.status_code}")
                print(f"   {response.text[:500]}")

    except Exception as e:
        print(f"❌ 오류: {e}")

    return False


if __name__ == "__main__":
    print("=" * 50)
    print("Upstage Solar API 테스트")
    print("=" * 50)

    success = asyncio.run(test_solar_api())

    if success:
        asyncio.run(test_question_generation())

    print("\n" + "=" * 50)
    print("테스트 완료")
    print("=" * 50)
