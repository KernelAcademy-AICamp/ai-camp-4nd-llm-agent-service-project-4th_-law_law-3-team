"""
공통 모듈 (Deprecated)

이 모듈은 이관 중입니다:
- database → app.core.database
- llm → app.tools.llm
- vectorstore → app.tools.vectorstore
- graph_service → app.tools.graph
- agent_router → app.multi_agent.routing
- chat_service → 삭제됨 (law_service.py로 이관 후 삭제)
"""

import warnings

warnings.warn(
    "app.common 모듈은 deprecated 되었습니다. "
    "새 위치: core/, tools/, services/, multi_agent/",
    DeprecationWarning,
    stacklevel=2,
)
