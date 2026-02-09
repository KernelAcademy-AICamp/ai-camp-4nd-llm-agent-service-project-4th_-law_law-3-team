# TDD Plan: Small Claims Document Generation

## DocumentService Tests
- [x] `test_render_template`: `generate_demand_letter` should render Jinja2 template with provided data (returns string content)
- [x] `test_generate_pdf`: `generate_demand_letter` should create a PDF file at the specified path
- [x] `test_missing_data`: `generate_demand_letter` should raise ValueError if required fields are missing

## Agent Logic Tests
- [x] `test_flow_to_document_generation`: `small_claims_agent` should transition to document generation step when all info is gathered
- [x] `test_invoke_document_service`: `small_claims_agent` should call `DocumentService` with correct data
