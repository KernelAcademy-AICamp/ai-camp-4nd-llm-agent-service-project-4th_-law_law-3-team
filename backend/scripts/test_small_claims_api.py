import requests
import json

BASE_URL = "http://localhost:8000/api/small-claims"

def test_evidence_checklist():
    print("Testing GET /evidence-checklist/fraud...")
    try:
        response = requests.get(f"{BASE_URL}/evidence-checklist/fraud")
        response.raise_for_status()
        print("✅ Success:", response.json())
    except Exception as e:
        print("❌ Failed:", e)

def test_related_cases():
    print("\nTesting GET /related-cases/fraud...")
    try:
        response = requests.get(f"{BASE_URL}/related-cases/fraud")
        response.raise_for_status()
        print("✅ Success:", response.json())
    except Exception as e:
        print("❌ Failed:", e)

def test_generate_document():
    print("\nTesting POST /generate-document...")
    payload = {
        "document_type": "demand_letter",
        "case_info": {
            "dispute_type": "fraud",
            "plaintiff_name": "홍길동",
            "plaintiff_address": "서울시 강남구",
            "defendant_name": "김사기",
            "amount": 500000,
            "description": "중고나라론 사기 당함",
            "incident_date": "2024-02-01"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-document", json=payload)
        response.raise_for_status()
        result = response.json()
        print("✅ Success:")
        print("Title:", result.get("title"))
        print("PDF URL:", result.get("pdf_url"))
        print("DOCX URL:", result.get("docx_url"))
        print("Text Content Preview:", result.get("content")[:50] + "...")
    except Exception as e:
        print("❌ Failed:", e)
        if hasattr(e, 'response') and e.response:
             print("Response content:", e.response.text)

if __name__ == "__main__":
    test_evidence_checklist()
    test_related_cases()
    test_generate_document()
