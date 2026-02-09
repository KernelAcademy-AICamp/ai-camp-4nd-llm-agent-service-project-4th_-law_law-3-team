import pytest
from app.multi_agent.agents.small_claims_agent import SmallClaimsAgent, SmallClaimsStep

@pytest.mark.asyncio
async def test_flow_to_document_generation():
    agent = SmallClaimsAgent()
    
    # Setup session at DEMAND_LETTER step
    initial_session = {
        "step": SmallClaimsStep.DEMAND_LETTER,
        # Assume some info is already gathered
        "recipient": "홍길동",
        "sender": "김철수",
        "claim_amount": 1000000 
    }
    
    # User asks for help drafting
    message = "내용증명 작성 도와줘"
    
    result = await agent.process(message, session_data=initial_session)
    
    # Should transition to a new step for gathering missing info (or direct generation if ready)
    # Let's assume we need a new step GATHER_DOC_INFO
    assert result.session_data["step"] == "gather_doc_info"
    assert "수신인 주소" in result.message # Should ask for missing info (address in this case)

@pytest.mark.asyncio
async def test_invoke_document_service(tmp_path):
    agent = SmallClaimsAgent()
    
    # Setup session with ALL required info gathered
    session_data = {
        "step": SmallClaimsStep.GATHER_DOC_INFO,
        "recipient": "홍길동",
        "sender": "김철수",
        "content": "돈 갚으세요",
        "claim_amount": 1000000,
        # Assume user just provided the last piece of info
    }
    
    # User provides the last info (e.g. date or confirms)
    message = "서울시 강남구 역삼동 (수신인 주소)"
    
    # We need to mock DocumentService or check if the result contains the file action
    # For integration test style, we can check if ActionType.FILE is returned
    
    # The agent logic should detect that all info is present found in message or session
    # and call generate_demand_letter
    
    result = await agent.process(message, session_data=session_data)
    
    # Should have a LINK action for download
    actions = result.actions
    has_download_action = any(a["type"] == "link" and a["label"] == "내용증명 다운로드" for a in actions)
    
    # Or check if message says "생성되었습니다"
    assert "생성되었습니다" in result.message or has_download_action
