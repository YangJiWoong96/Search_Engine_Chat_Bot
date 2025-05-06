# api/schemas.py
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """사용자 쿼리를 받는 요청 모델"""

    query: str


class AnswerResponse(BaseModel):
    """챗봇 답변을 반환하는 응답 모델"""

    answer: str
