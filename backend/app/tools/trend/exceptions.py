"""트렌드 도구 예외 클래스"""


class TrendSourceError(Exception):
    """트렌드 소스 API 호출 실패"""


class TrendScoringError(Exception):
    """트렌드 스코어링 처리 실패"""
