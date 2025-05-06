import logging
import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException

# 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # api
app_root_dir = os.path.dirname(current_dir)  # app
sys.path.append(app_root_dir)

# settings 모듈
try:
    from config import settings
except ImportError as e:
    print(f"[ERROR] settings import 실패: {e}.")
    settings = None  # 임시 설정

# pipeline 모듈
try:
    from core.pipeline import run_pipeline, agent, llm
except ImportError as e:
    print(f"[ERROR] pipeline import 실패: {e}.")
    run_pipeline = None
    agent = None
    llm = None

# schemas 모듈
try:
    from .schemas import QueryRequest, AnswerResponse
except ImportError as e:
    print(f"[ERROR] schemas import 실패: {e}.")
    QueryRequest = None
    AnswerResponse = None


log_level = logging.INFO
if settings and hasattr(settings, "LOG_LEVEL"):
    log_level_str = getattr(logging, settings.LOG_LEVEL, "INFO")
    log_level = log_level_str if isinstance(log_level_str, int) else logging.INFO
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# FastAPI 앱
# 앱 메타데이터
app = FastAPI(
    title="Conversational RAG API",
    description="Processes user queries using LangChain, search engines, and LLMs.",
    version="1.0.0",
)


# API 엔드포인트 정의
@app.post(
    "/process",
    response_model=AnswerResponse if AnswerResponse else None,
    summary="Process User Query",
    description="Receives a user query, processes it through the RAG pipeline, and returns the answer.",
    tags=["Chatbot"],
)  # API 문서 그룹화
async def process_query_endpoint(request: QueryRequest if QueryRequest else None):
    if not QueryRequest or not AnswerResponse:
        raise HTTPException(status_code=500, detail="API schema definition error.")
    if not request or not request.query or not request.query.strip():
        logger.warning("Received invalid request: query is empty.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Received API request for query: '{request.query}'")

    if not run_pipeline:  # 파이프라인 함수 로드 실패 시
        logger.error("Pipeline function is not available.")
        raise HTTPException(
            status_code=500, detail="Internal server error: Pipeline unavailable."
        )

    try:
        # 핵심 파이프라인
        final_answer = await run_pipeline(request.query)
        logger.info(
            f"Processed query successfully via API. Answer length: {len(final_answer)}"
        )
        return AnswerResponse(answer=final_answer)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            f"API 상 쿼리 파싱 에러 '{request.query}' : {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="An unexpected internal server error occurred."
        )


@app.get(
    "/health",
    summary="Health Check",
    description="Checks if the API and its core components (LLM, Agent) are operational.",
    tags=["Health"],
)
async def health_check():

    # core.pipeline 모듈에서 로드
    llm_ok = llm is not None
    agent_ok = agent is not None

    if llm_ok and agent_ok:
        return {
            "status": "ok",
            "message": "API is running and core components seem initialized.",
        }
    else:
        details = []
        if not llm_ok:
            details.append("LLM 초기화 실패")
        if not agent_ok:
            details.append("Agent 초기화 실패")
        logger.error(f"Health check 실패: {', '.join(details)}")
        # 서비스 준비 안됨 상태 반환
        raise HTTPException(
            status_code=503, detail=f"서버 이용불가: {', '.join(details)}"
        )


# uvicorn - 도커 테스트 용

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # 환경 변수 - 포트번호
    logger.info(f"FastAPI 서버 Uvicorn 실행. 포트넘버: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)  # app 모듈 직접 전달
