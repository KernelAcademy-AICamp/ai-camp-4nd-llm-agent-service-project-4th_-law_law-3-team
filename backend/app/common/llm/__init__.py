"""
LLM 프로바이더 모듈

환경 변수 LLM_PROVIDER에 따라 적절한 LLM을 선택합니다.
- openai (기본값): OpenAI GPT 모델
- anthropic: Anthropic Claude 모델
- google: Google Gemini 모델

LangChain/LangGraph와 호환되는 ChatModel 인터페이스를 제공합니다.

Usage:
    from app.common.llm import get_chat_model, get_llm_config

    # ChatModel 인스턴스 가져오기
    llm = get_chat_model()

    # LangChain 스타일 사용
    response = llm.invoke([HumanMessage(content="안녕하세요")])

    # LangGraph에서 사용
    from langgraph.graph import StateGraph
    graph = StateGraph(...)
    graph.add_node("agent", lambda state: llm.invoke(state["messages"]))
"""

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """
    환경 설정에 따라 적절한 ChatModel 인스턴스 반환

    Args:
        provider: LLM 프로바이더 (openai, anthropic, google)
        model: 모델명 (없으면 config에서)
        temperature: 생성 온도
        **kwargs: 추가 인자 (각 프로바이더별 옵션)

    Returns:
        BaseChatModel 인스턴스
    """
    provider_name = provider or getattr(settings, "LLM_PROVIDER", "openai")
    provider_name = str(provider_name).lower()

    if provider_name == "anthropic":
        return _get_anthropic_model(model, temperature, **kwargs)
    elif provider_name == "google":
        return _get_google_model(model, temperature, **kwargs)
    else:
        # 기본값: OpenAI
        return _get_openai_model(model, temperature, **kwargs)


def _get_openai_model(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """OpenAI ChatModel 생성"""
    from langchain_openai import ChatOpenAI

    model_name = str(model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"))

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=str(settings.OPENAI_API_KEY),  # type: ignore[arg-type]
        **kwargs,
    )


def _get_anthropic_model(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Anthropic ChatModel 생성"""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic 패키지가 필요합니다. "
            "설치: uv add langchain-anthropic"
        )

    model_name = str(model or getattr(settings, "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    api_key = str(getattr(settings, "ANTHROPIC_API_KEY", ""))

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    return ChatAnthropic(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,  # type: ignore[arg-type]
        **kwargs,
    )


def _get_google_model(
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Google Gemini ChatModel 생성"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai 패키지가 필요합니다. "
            "설치: uv add langchain-google-genai"
        )

    model_name = model or getattr(settings, "GOOGLE_MODEL", "gemini-3-flash-preview")
    api_key = getattr(settings, "GOOGLE_API_KEY", "")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs,
    )


def get_llm_config() -> dict[str, Any]:
    """현재 LLM 설정 정보 반환"""
    provider = getattr(settings, "LLM_PROVIDER", "openai")

    config = {
        "provider": provider,
    }

    if provider == "openai":
        config["model"] = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "anthropic":
        config["model"] = getattr(settings, "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    elif provider == "google":
        config["model"] = getattr(settings, "GOOGLE_MODEL", "gemini-1.5-flash")

    return config


# Export
__all__ = [
    "get_chat_model",
    "get_llm_config",
]
