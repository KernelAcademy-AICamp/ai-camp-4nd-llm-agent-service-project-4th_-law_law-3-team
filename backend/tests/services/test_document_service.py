from app.services.document_service import DocumentService
import os

def test_render_template():
    service = DocumentService()
    data = {
        "recipient": "홍길동",
        "sender": "김철수",
        "content": "돈 갚으세요."
    }
    # This method doesn't exist yet, so it should fail (Red)
    result = service.generate_demand_letter(data, format="text")
    
    assert "홍길동 귀하" in result
    assert "김철수" in result
    assert "돈 갚으세요" in result

def test_generate_pdf(tmp_path):
    service = DocumentService()
    data = {
        "recipient": "홍길동",
        "sender": "김철수",
        "content": "돈 갚으세요."
    }
    output_path = tmp_path / "demand_letter.pdf"
    
    # This should fail because format="pdf" is not implemented yet
    result_path = service.generate_demand_letter(data, format="pdf", output_path=str(output_path))
    
    assert os.path.exists(result_path)
    assert result_path.endswith(".pdf")
    # Check if file size is greater than 0
    assert os.path.getsize(result_path) > 0

def test_missing_data():
    service = DocumentService()
    # Missing 'content'
    data = {
        "recipient": "홍길동",
        "sender": "김철수"
    }
    
    import pytest
    with pytest.raises(ValueError) as excinfo:
        service.generate_demand_letter(data)
    
    assert "필수 데이터 누락" in str(excinfo.value)
