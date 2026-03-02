"""웹툰 스토리보드 AI 파이프라인 패키지

Chain 1: panel_planner — 대본 → 장면 분할 (Solar Pro2)
Chain 2: prompt_builder — 장면 → 이미지 프롬프트 (텍스트 조합)
Chain 3: image_generator — 이미지 생성 (Gemini 3 Pro Image)
"""
