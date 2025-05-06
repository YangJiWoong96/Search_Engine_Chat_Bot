# core/pipeline.py
import os
import re
import asyncio
import logging
from urllib.parse import urlparse

# 환경설정
# config/settings.py
from config import settings

# 랭체인 라이브러리
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain_core.agents import AgentAction
from langchain.schema import BaseMessage

# search 폴더의 각 엔진 파일
try:
    from search.ces import CesEngine
    from search.naver import NaverEngine
    from search.serpapi import SerpapiEngine
except ImportError as e:
    logging.error(f"검색엔진 import 실패: {e}. 각 'search' 엔진 확인 필요")
    CesEngine, NaverEngine, SerpapiEngine = None, None, None

# 유틸
# utils/helpers.py
try:
    from utils.helpers import (
        _extract_and_process_item,
        format_search_results,
        parse_agent_observation,
    )
except ImportError as e:
    logging.error(f"helper import 실패: {e}. utils 모듈 확인 필요")
    # None 반환
    _extract_and_process_item = None
    format_search_results = None
    parse_agent_observation = None

logger = logging.getLogger(__name__)

# 1. LLM 초기화
llm = None
if settings.OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            streaming=False,
            temperature=0.0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        logger.info(f"ChatOpenAI LLM 모델 초기화: {settings.OPENAI_MODEL}")
    except Exception as e:
        logger.error(f"ChatOpenAI LLM 모델 초기화 실패: {e}", exc_info=True)
else:
    logger.error("OPENAI_API_KEY 에러. 초기화 실패")

# 2. 검색 엔진 초기화
serp, naver, ces = None, None, None
try:
    if SerpapiEngine:
        serp = SerpapiEngine()
    if NaverEngine:
        naver = NaverEngine()
    if CesEngine:
        ces = CesEngine()
    logger.info("Search engines 인스턴스 생성")
except Exception as e:
    logger.error(f"Search engines 인스턴스 생성 에러: {e}", exc_info=True)

# 3. LLMChain Prompt 정의
# 1) search decide chain
decide_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["query"],
        template="""
다음 사용자 질의에 대해,
- 의미를 알 수 없거나, LLM만으로 즉시 정확히 답변할 수 있으면, 'NO_SEARCH'
- 사용자 질의가 검색을 명시했거나, LLM만으로 답변할 수 없다면, (최신 정보·수치·통계·주가 등) 'SEARCH'
를 반드시 출력하라.

질의: 'ㅇ'
답변: NO_SEARCH

질의: '파이썬 리스트 컴프리헨션이 뭐야?'
답변: NO_SEARCH

질의: '엔비디아 최신 주가 알려줘'
답변: SEARCH

질의: 'RAG 기초에 대해 검색해'
답변: SEARCH

질의: {query}
답변:""",
    ),
    output_key="decision",
)

# refine_chain (RouterChain)
# 2) query 목적별 Chain

# 정보형 쿼리
prompt_question = PromptTemplate(
    input_variables=["input"],
    template="""사용자의 질문형 쿼리를 웹 검색 엔진에서 좋은 결과를 얻을 수 있도록, **핵심 키워드 중심의 간결한 검색 구문**으로 재작성하라.

조건:
- 원본 질문의 핵심 의도와 중요한 명사/개념은 반드시 유지하라.
- '어떻게', '왜', '무엇', '언제', '어디서', '인지' 등 의문형 표현 대신, 검색 결과에 해당 내용이 포함될 만한 키워드 조합으로 바꿔라.
- 검색에 불필요한 조사, 부사, 구문은 최대한 제거하되, 키워드의 의미가 왜곡되지 않도록 주의하라.
- 최종 결과는 검색 엔진 입력에 바로 사용될 수 있어야 한다.

예시:
쿼리: '왜 금리가 계속 오르고 있나요?'
재작성된 쿼리: '최근 금리 인상 원인'

쿼리: '챗GPT는 어떻게 작동하나요?'
재작성된 쿼리: '챗GPT 작동 원리 및 기술'

쿼리: '요즘 미국 달러 환율이 왜 이렇게 낮아?'
재작성된 쿼리: '최근 미국 달러 환율 하락 이유 분석'

쿼리: {input}
재작성된 쿼리:""",
)
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# 지시형 쿼리
prompt_keyword = PromptTemplate(
    input_variables=["input"],
    template="""사용자 쿼리에서 검색 목적(무엇을 하고자 하는지)을 파악하고, 웹 검색 엔진에서 **정확하고 효율적인 결과**를 얻을 수 있는 **명확하고 구체적인 검색 구문**으로 재작성하라. 이 쿼리는 주로 특정 정보, 방법, 대상 찾기 등 지시적인 성격을 가진다.

조건:
- **검색 목적 달성**에 필요한 핵심 키워드(주로 명사)를 반드시 포함하라.
- 원본 쿼리에 포함된 **중요한 제약 조건이나 특정 대상** (예: 특정 버전, 특정 지역, 특정 기간 등)이 있다면 검색 구문에 반영하라.
- 불필요한 미사여구, 감탄사, 접속사, 일반적인 질문 표현 ('알려줘', '궁금해' 등)은 제거하여 간결하게 만들어라.
- 최종 결과는 검색 엔진에 바로 입력하기 좋은 형태여야 한다.

예시:
쿼리: '유튜브 썸네일 만드는 최신 방법 알려줘'
재작성된 쿼리: '유튜브 썸네일 제작 최신 가이드'

쿼리: '파이썬 3.10 버전으로 웹 크롤링 하는 기초적인 법 알려줘'
재작성된 쿼리: '파이썬 3.10 웹 크롤링 기초'

쿼리: '면접용 1분 자기소개서 잘 쓰는 팁 알려줘'
재작성된 쿼리: '면접 1분 자기소개 작성 팁'

쿼리: {input}
재작성된 쿼리:""",
)
chain_keyword = LLMChain(llm=llm, prompt=prompt_keyword)

# 탐색형 쿼리 - 일반
prompt_general = PromptTemplate(
    input_variables=["input"],
    template="""사용자의 쿼리가 넓은 주제를 탐색하거나, 사례/추천/비교/동향 등을 찾는 성격일 때, 웹 검색 엔진에서 **관련성 높고 다양한 정보**를 찾는데 효과적인 **구체화된 검색 문장**으로 재작성하라.

조건:
- 쿼리에 숨겨진 사용자 의도(예: 최신 정보 찾기, 장단점 비교, 구체적인 사용 사례, 모범 사례 학습 등)를 파악하여 검색 문장에 반영하라. 이를 위해 "최신 동향", "장단점 비교", "구체적인 사례", "활용 방안", "모범 사례", "가이드라인" 등의 구문을 적절히 추가할 수 있다.
- 너무 포괄적이거나 모호한 쿼리는 **핵심 주제를 유지하면서 좀 더 구체적인 방향**으로 재구성하라. (예: 'AI 발전' -> '최신 AI 기술 동향 및 활용 사례')
- 최종 결과는 검색 엔진 입력에 반드시 적합한 형태이어야 하며, 자연스러운 문장 형태를 유지해도 좋다.
- 원본 쿼리의 핵심 주제에서 절대로 벗어나지 않도록 주의하라.

예시:
쿼리: '챗GPT를 활용한 재미있는 사례 알려줘'
재작성된 쿼리: '챗GPT 창의적인 활용 사례 모음'

쿼리: '요즘 인기 있는 AI 서비스 뭐가 있어?'
재작성된 쿼리: '최신 인기 AI 서비스 종류 및 특징 비교'

쿼리: '재택근무 잘하는 방법이나 사례 있을까?'
재작성된 쿼리: '재택근무 생산성 향상 방법 및 성공 사례'

쿼리: '기후 변화 영향'
재작성된 쿼리: '기후 변화가 환경과 사회에 미치는 영향 분석'

쿼리: {input}
재작성된 쿼리:""",
)
chain_general = LLMChain(llm=llm, prompt=prompt_general)

# 기본형 쿼리
prompt_basic = PromptTemplate(
    input_variables=["input"],
    template="""사용자의 쿼리가 매우 짧거나, 문법적으로 오류가 있거나, 의미가 불명확하여 다른 방식으로 처리하기 어려울 때, **최대한 원본의 핵심 단어를 유지하면서 검색 엔진에 입력 가능한 최소한의 키워드 구문**으로 재작성하라. 

조건:
- 원본 쿼리에 나타난 **가장 중요한 명사 또는 키워드**를 식별하고 유지하라.
- 불필요한 감탄사, 중복 단어, 명백한 오타 등 노이즈를 제거하라.
- **임의로 장소, 시간, 구체적인 맥락을 과도하게 추측하거나 추가하지 마라.** (예: '날씨' -> '오늘 서울 날씨' X, '날씨 정보' O)
- 검색이 가능하도록 최소한의 단어를 조합하되, 원본의 의미를 크게 왜곡하지 마라.
- 최종 결과는 간결한 키워드 또는 키워드 구문 형태여야 한다.

예시:
쿼리: '이거 왜이럼???????'
재작성된 쿼리: '문제 원인 또는 해결 방법'

쿼리: '서울 날씨 알려줭'
재작성된 쿼리: '오늘 서울 날씨 정보'

쿼리: '엔비디아 주가 얼마임?'
재작성된 쿼리: '엔비디아 주가'

쿼리: {input}
재작성된 쿼리:""",
)
chain_basic = LLMChain(llm=llm, prompt=prompt_basic)


# 3) LLMRouterChain 설정
prompt_infos_for_router = [
    {
        "name": "keyword_rewrite",
        "description": "쿼리가 '~하는 법', '설치 방법', '구매처 찾기' 등 **구체적인 행동이나 대상에 대한 직접적인 정보 요청**일 때 사용. 결과는 간결한 키워드/명사구 형태.",
        "keywords": [
            "방법",
            "팁",
            "찾기",
            "구매",
            "설치",
            "만들기",
            "요청",
        ],  # 키워드 부여시 해당 query에 점수 부여 후, 내부 알고리즘을 통해 선정
    },
    {
        "name": "question_rewrite",
        "description": "쿼리가 '왜', '어떻게', '무엇', '언제', '차이점' 등 **명확한 의문사를 포함하거나 원인/이유/정의 등을 묻는 질문**일 때 사용. 결과는 질문의 핵심 주제를 나타내는 검색 구문 형태.",
        "keywords": [
            "왜",
            "어떻게",
            "무엇",
            "언제",
            "어디서",
            "정의",
            "원인",
            "이유",
            "비교",
        ],
    },
    {
        "name": "general_rewrite",
        "description": "쿼리가 특정 주제에 대한 **사례, 추천, 비교, 최신 동향, 전반적인 정보 탐색** 등 넓은 범위의 정보를 찾거나 주제가 다소 모호할 때 사용. 결과는 탐색 의도를 반영하여 약간 구체화된 문장 형태.",
        "keywords": ["사례", "추천", "비교", "동향", "트렌드", "종류", "영향", "전망"],
    },
    {
        "name": "basic_rewrite",
        "description": "쿼리가 **매우 짧거나, 의미가 불명확하거나, 문법 오류가 심하거나, 위의 다른 유형으로 분류하기 어려울 때** 사용하는 최종 안전 장치(Fallback). 최소한의 정제만 거친 키워드 형태로 재작성.",
        "keywords": ["단순 키워드", "오류 포함", "의미 불명확", "Fallback"],
    },
]
destinations = "\n".join(
    [f'{p["name"]}: {p["description"]}' for p in prompt_infos_for_router]
)
router_template_str = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate(
    template=router_template_str,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

llm_router_chain = LLMRouterChain.from_llm(llm, router_prompt, verbose=True)

# 4) MultiPromptChain 설정
destination_chains = {
    "keyword_rewrite": chain_keyword,
    "question_rewrite": chain_question,
    "general_rewrite": chain_general,
    "basic_rewrite": chain_basic,
}
default_chain = chain_basic
refine_chain = MultiPromptChain(
    router_chain=llm_router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)


# 5) Search Engine choose chain (SequentialChain)

# 5-1) 쿼리 분석 chain
analyze_prompt_template = """\
다음 쿼리를 분석하여 검색 엔진 선택에 유의미한 핵심 속성들을 도출하라.  
각 속성은 하나의 명확한 값 또는 요약 구문으로 기술하라.

쿼리: {refined_query}

분석 결과:
- 최신성 요구 수준: [매우 높음 (실시간/수시간 내), 높음 (최근/수일 내), 중간 (최근 정보 선호), 낮음 (시간 상관 없음)]
- 지역 중심성: [한국 특정, 특정 해외 지역, 전 세계적, 지역 무관]
- 정보 유형: [뉴스/기사, 블로그/리뷰/카페글, 지식인/커뮤니티, 기술 문서/논문, 금융/주가/환율 데이터, 날씨 데이터, 제품/쇼핑 정보, 간단한 정의/개념, 기타]
- 탐색 깊이: [얕음 (간단 확인), 보통 (대략적 개요), 깊음 (비교/사례/리뷰 등)]
- 쿼리 난이도/명확성: [명확함, 다소 모호함, 매우 모호함]
- 핵심 주제/키워드: [주제를 간결하게 요약하라]
"""

analyze_properties_prompt = PromptTemplate(
    input_variables=["refined_query"], template=analyze_prompt_template
)
analyze_properties_chain = LLMChain(
    llm=llm, prompt=analyze_properties_prompt, output_key="analysis_result"
)

# 5-2) 엔진 선택 Chain

select_engine_template = """\
다음은 사용자 쿼리와 그에 대한 분석 결과이다. 아래 조건을 기준으로, 가장 적합한 검색 엔진 하나만 선택하라.

[조건]

1. SerpAPI로 선택:
- 최신성 요구 수준이 '매우 높음' 또는 '높음'  
- 또는 정보 유형이 '금융/주가/환율 데이터', '날씨 데이터', '뉴스/기사', '실시간 트렌드'

2. Naver로 선택:
- 지역 중심성이 '한국 특정'이며 정보 유형이 한국인의 관점에서 '뉴스/기사', '블로그/리뷰/카페글', '지식인/커뮤니티', '제품/쇼핑 정보' 등
- 또는 쿼리 난이도가 '다소 모호함'이면서 한국 대상 정보일 때

3. CES로 선택:
- 정보 유형이 정확도가 높고, 최신성 요구 수준이 '높음' 또는 탐색 깊이가 '깊음'이거나 주제가 학문적/글로벌할 때
- 포괄적인 '기술 문서/논문', '간단한 정의/개념', 또는 지역 중심성이 '해외/전세계'

4. 애매하거나 판단 어려운 경우 기본적으로 Naver 선택

쿼리: {refined_query}
분석 결과:
{analysis_result}

최종 선택 엔진 (반드시 SerpAPI, Naver, CES 중 하나만 따옴표 없이 출력):
"""

select_engine_prompt = PromptTemplate(
    input_variables=["refined_query", "analysis_result"],
    template=select_engine_template,
)
select_engine_chain = LLMChain(
    llm=llm, prompt=select_engine_prompt, output_key="engine_name"
)
# 5-3) SequentialChain 연결
choose_chain = SequentialChain(
    chains=[analyze_properties_chain, select_engine_chain],
    input_variables=["refined_query"],
    output_variables=["engine_name"],
    verbose=True,
)

# 6) no_search_chain
no_search_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["query"],
        template="사용자 질의에 대해 간결하고 명확하게 20자 이내로 답변하라.\n질의: {query}\n답변: ",
    ),
    output_key="answer",
)

# 7) search_answer_chain (HTML Content와 refine_query를 받아서 적절하게 요약)
search_answer_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["content", "refined_query"],
        template="""
아래 검색 결과 본문(content)과 원본 검색 쿼리(refined_query)를 참고하여, 사용자의 쿼리에 대한 답변이 될 수 있도록 본문의 핵심 내용을 **최소 3-4문장 이상의 충분한 길이로 상세하게 요약**하라.

조건:
- 반드시 본문 내용에만 기반하여 작성하라.
- 쿼리와 직접적으로 관련 없는 부가 정보나 광고성 문구는 제거하라.
- 원본의 중요한 사실, 수치, 개념 등은 반드시 유지하면서 자연스럽게 설명하라.
- 출처 및 URL(`https://...` 형식)은 반드시 포함하라.

쿼리: {refined_query}

본문:
{content}

요약:""",
    ),
    output_key="summary",
)

# 8) fact check chain
fact_check_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["answer", "history"],
        template="""\
너는 꼼꼼한 팩트 검증기이다. 너의 최종 목표는 사용자가 질문한 내용에 대해 사실에 기반하고 명확하며, 반드시 출처 정보를 포함하는 답변을 생성하는 것이다. 
아래 '검토 대상 답변'을 '검토 참고 정보'와 비교하여 사실 관계를 확인하고, 필요한 경우 수정하여 최종적으로 정제된 답변을 생성하라.

검토 및 정제 지침:
1.  **[검토 대상 답변]**과 **[검토 참고 정보]** (특히 '[검색된 본문]' 섹션)를 **문장 단위로 비교**하여 사실 관계의 일치 여부를 확인하라.
2.  **불일치/오류 식별:** [검토 대상 답변]에서 [검색된 본문] 내용과 다르거나, 부정확하거나, 사용자의 원래 질문과 관련 없는 정보를 식별하라.
3.  **수정 및 정제:** 식별된 오류를 수정하고, 불필요한 내용은 제거하며, 문맥을 자연스럽게 다듬어라. 모든 내용은 반드시 [검색된 본문] 정보에 근거해야 한다.
4.  **출처 추출 및 확인:** {answer}에 포함된 **모든 유효한 URL**들을 반드시 식별하고 추출하라. 이 URL들은 최종 답변의 근거이다. (`https://...` 형식)
5.  **최종 답변 생성:** 수정 및 정제된 답변 본문 뒤에, **반드시 다음 형식으로 추출된 모든 출처 URL 목록을 포함**하여 최종 결과물을 작성하라.
주의 사항: 
-URL을 작성할 때, 반드시 현재 수정 및 정제된 답변과 관련이 있는 URL인지 확인하고 해당하는 URL만을 작성하라. 
-수정 및 정제된 답변 본문에 URL이 포함되어있지 않다면, URL은 ''을 반환하라. 

**출력 형식 (매우 중요):**
ChatBot: [수정 및 정제된 최종 답변 본문 내용...]

출처:
- [추출된 첫 번째 URL]
- [추출된 두 번째 URL]
- ... (추출된 모든 URL 나열)

검토 대상 답변:
{answer}

[검토 참고 정보]  
- Observation: 에이전트가 검색을 통해 얻은 본문  
- History: 사용자와의 이전 대화 기록
{history}

---
# 최종 출력 (위의 '출력 형식'을 반드시 준수하라):
ChatBot:
""",
    ),
    output_key="checked_answer",
)

if all(
    [
        decide_chain,
        refine_chain,
        choose_chain,
        no_search_chain,
        search_answer_chain,
        fact_check_chain,
        llm,
        parse_agent_observation,  # Agent 헬퍼 함수 로드
    ]
):
    logger.info("LLM Chains 및 컴포넌트 초기화 성공")
else:
    # 실패 chain 로그
    missing = [
        name
        for name, obj in {
            "decide_chain": decide_chain,
            "refine_chain": refine_chain,
            "choose_chain": choose_chain,
            "no_search_chain": no_search_chain,
            "search_answer_chain": search_answer_chain,
            "fact_check_chain": fact_check_chain,
            "llm": llm,
            "parse_agent_observation": parse_agent_observation,
        }.items()
        if obj is None
    ]
    logger.error(f"LLM Chains or 컴포넌트 초기화 실패: {', '.join(missing)} 누락")


# 4. Tool 함수 정의

# Search Engine


# serapi
async def run_serpapi_async(query):
    if not serp:
        return ("SerpAPI 엔진 초기화 실패", [])
    if not _extract_and_process_item or not format_search_results:
        return ("헬퍼 함수 임포트 실패", [])

    try:
        search_result = await asyncio.to_thread(serp.search, query)
        # handle_response 결과가 특별한 정보(날씨, 주가 등)이면 answer box이기 때문에 따로 링크 필요없음
        handled_result = await asyncio.to_thread(serp.handle_response, search_result)
        is_generic_web_search = handled_result.startswith("웹 검색")
        is_no_result = handled_result == "검색 결과 없음."

        if not is_generic_web_search and not is_no_result:
            return (handled_result, [])
        elif is_no_result:
            return (handled_result, [])  # 결과 없음

        # 일반 웹 검색 결과 처리 (organic_results)
        logger.info("run_serpapi_async: Processing organic_results standard way.")
        items = []
        if "organic_results" in search_result:
            items = [
                {"title": i.get("title", ""), "link": i.get("link", "")}
                for i in search_result.get("organic_results", [])
                if i.get("link")
            ]
        if not items:
            return ("검색 결과 없음.", [])

        tasks = [_extract_and_process_item(serp, item) for item in items]
        results = await asyncio.gather(*tasks)
        valid_texts = [text for text, link in results if text]
        valid_links = [link for text, link in results if link]  # 링크 리스트

        observation_string = format_search_results(
            valid_texts, valid_links
        )  # Agent에게 전달할 문자열
        return (observation_string, valid_links)  # 출력을 위해 튜플로

    except Exception as e:
        logger.error(f"run_serpapi_async 에러: {e}", exc_info=True)
        return (f"SerpAPI 검색 처리 중 오류 발생: {e}", [])  # 에러 시 빈 문자열


# naver
async def run_naver_async(query):
    if not naver:
        return ("Naver 엔진 초기화 실패", [])
    if not _extract_and_process_item or not format_search_results:
        return ("헬퍼 함수 임포트 실패", [])
    try:
        items = await asyncio.to_thread(naver.search, query)
        if not items:
            return ("네이버 검색 결과 없음", [])
        tasks = [_extract_and_process_item(naver, item) for item in items]
        results = await asyncio.gather(*tasks)
        valid_texts = [text for text, link in results if text]
        valid_links = [link for text, link in results if link]  # 링크 리스트
        observation_string = format_search_results(valid_texts, valid_links)
        return (observation_string, valid_links)  # 출력을 위해 튜플로
    except Exception as e:
        logger.error(f"run_naver_async 에러: {e}", exc_info=True)
        return (f"네이버 검색 처리 중 오류 발생: {e}", [])


# ces
async def run_ces_async(query):
    if not ces:
        return ("CES 엔진 초기화 실패", [])
    if not _extract_and_process_item or not format_search_results:
        return ("헬퍼 함수 임포트 실패", [])
    try:
        items = await asyncio.to_thread(ces.search, query)
        if not items:
            return ("CES 검색 결과 없음", [])
        tasks = [_extract_and_process_item(ces, item) for item in items]
        results = await asyncio.gather(*tasks)
        valid_texts = [text for text, link in results if text]
        valid_links = [link for text, link in results if link]  # 링크 리스트

        observation_string = format_search_results(valid_texts, valid_links)
        return (observation_string, valid_links)  # 출력을 위해 튜플로
    except Exception as e:
        logger.error(f"run_ces_async 에러: {e}", exc_info=True)
        return (f"CES 검색 처리 중 오류 발생: {e}", [])


# 5. Agent 호출을 위한 Tool 목록 정의
tools = []
if all([serp, naver, ces]):  # 검색 엔진 초기화
    # 비동기 실행
    tools = [
        Tool(
            name="serpapi_search",
            func=lambda q: asyncio.run(run_serpapi_async(q)),
            coroutine=run_serpapi_async,
            description="주가, 날씨, 실시간 정보, 일반 웹 검색 등 다양한 최신 정보를 검색할 때 사용. 쿼리를 입력받아 검색 결과 '본문'과 '출처' URL이 포함된 텍스트를 반환.",
        ),
        Tool(
            name="naver_search",
            func=lambda q: asyncio.run(run_naver_async(q)),
            coroutine=run_naver_async,
            description="한국 관련 뉴스, 정부 기관, 블로그, 지식인, 쇼핑 등 한국 특화 정보 검색 시 사용. 쿼리를 입력받아 여러 결과의 '본문'과 '출처' URL을 통합하여 반환.",
        ),
        Tool(
            name="ces_search",
            func=lambda q: asyncio.run(run_ces_async(q)),
            coroutine=run_ces_async,
            description="기술 블로그, 해외 논문, 특정 웹사이트 등 보다 정제된 일반 웹 검색이 필요할 때 사용. 쿼리를 입력받아 여러 결과의 '본문'과 '출처' URL을 통합하여 반환.",
        ),
    ]
    logger.info("Search Tools 초기화 성공")
else:
    logger.error("하나 이상의 검색 엔진 초기화 실패로 Tools 목록이 비어있음")

# 6. 대화 이력 메모리 설정
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
logger.info("메모리 초기화")

# 7. Agent 초기화
agent = None
if tools and llm:
    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",  # ReAct 에이전트
            memory=memory,
            handle_parsing_errors="Agent 파싱 오류 발생: 출력 형식 오류. 재시도 시작",  # 파싱 오류 재시도 처리
            max_iterations=5,  # 최대 5번 재시도
            verbose=True,  # 상세 로그 출력
        )
        logger.info("Agent 초기화 성공")
    except Exception as e:
        logger.error(f"Agent 초기화 실패: {e}", exc_info=True)
else:
    logger.error("Agent 초기화 실패: Tools 또는 LLM이 미준비 상태")


# 8. 파이프라인 정의 (메커니즘)
async def run_pipeline(query: str) -> str:
    """
    사용자 질의 처리 파이프라인 (Agent 사용)
    - 쿼리 정제
    - 알맞은 Search Engine 선택
    - Content 추출,
    - Content 전처리
    - Content 요약/팩트체크
    """

    # 8-1. 필수 chain 및 Agent 컴포넌트 초기화 및 확인 & 검색 여부 판단

    if (
        not llm
        or not decide_chain
        or not no_search_chain
        or not refine_chain
        or not choose_chain
        or not agent  # Agent 확인
        or not search_answer_chain  # 요약
        or not fact_check_chain  # 팩트체크
        or not parse_agent_observation  # Content 파서
    ):
        return "챗봇 초기화 오류 발생."

    # 검색에 들어간다면,,,
    decision = "SEARCH"
    try:
        decision_result = await decide_chain.ainvoke({"query": query})
        decision = decision_result.get("decision", "SEARCH").strip().upper()
        logger.info(f"Decision: {decision}")
    except Exception as e:
        decision = "SEARCH"
        logger.error(f"Decide chain error: {e}", exc_info=True)

    # 검색이 필요없다면,,,
    if decision == "NO_SEARCH":
        try:
            if memory:
                memory.chat_memory.add_user_message(query)
            no_search_result = await no_search_chain.ainvoke({"query": query})
            answer = no_search_result.get("answer", "답변 생성 불가").strip()[:50]
            if memory:
                memory.chat_memory.add_ai_message(answer)
            return answer
        except Exception as e:
            return "간단 답변 생성 에러"

    # --- SEARCH 경로 처리 ---
    # 컴포넌트 확인 (Agent, 요약, 팩트체크 포함)
    if (
        not agent
        or not refine_chain
        or not choose_chain
        or not search_answer_chain
        or not fact_check_chain
    ):
        missing_search = [
            name
            for name, obj in {
                "agent": agent,
                "refine_chain": refine_chain,
                "choose_chain": choose_chain,
                "search_answer_chain": search_answer_chain,
                "fact_check_chain": fact_check_chain,
            }.items()
            if obj is None
        ]
        logger.error(f" 경로 설정 확인 필요!: {', '.join(missing_search)}")
        # Fallback 시 no_search_chain 호출 로직으로...
        try:
            logger.warning("Search 엔진 설정 확인 필요!!")
            if memory:
                memory.chat_memory.add_user_message(query)
            fallback_result = await no_search_chain.ainvoke({"query": query})
            fallback_answer = fallback_result.get("answer", "답변 오류").strip()[:20]
            if memory:
                memory.chat_memory.add_ai_message(fallback_answer)
            return fallback_answer
        except Exception as e_fb:
            logger.error(f"Fallback No Search 에러: {e_fb}")
            return "답변 생성 중 오류 발생."

    # 변수 정의
    refined = query
    engine_name = "ces"  # default
    final_answer_str = ""  # Agent 최종 답변
    original_source_links = []
    # 요약 및 팩트체크용
    agent_observation_for_factcheck = ""
    extracted_content = ""
    summary = ""
    checked_summary = ""

    # 2. 쿼리 재작성
    try:
        refine_result = await refine_chain.ainvoke({"input": query})
        if isinstance(refine_result, dict):
            refined = refine_result.get("text", query).strip()
        elif isinstance(refine_result, str):
            refined = refine_result.strip()
        if not refined:
            refined = query
        logger.info(f"Refined Query: {refined}")
    except Exception as e:
        refined = query
        logger.error(f"Refine chain 에러: {e}", exc_info=True)

    # 3. 검색 엔진 선택
    try:
        choose_result = await choose_chain.ainvoke({"refined_query": refined})
        engine_name_raw = choose_result.get("engine_name", "CES").strip()
        engine_name = re.sub(r"[^a-zA-Z]+", "", engine_name_raw).lower()
        if engine_name not in ["serpapi", "naver", "ces"]:
            engine_name = "ces"
            logger.warning(f"엔진 이름 부정확: {engine_name_raw}")
        logger.info(f"선택된 엔진: {engine_name}")
    except Exception as e:
        engine_name = "ces"
        logger.error(f"Choose chain 에러: {e}", exc_info=True)

    # 4. Agent 실행 및 출력을 위한 원본 링크 리스트 저장
    tool_map = {
        "serpapi": "serpapi_search",
        "naver": "naver_search",
        "ces": "ces_search",
    }
    selected_tool_name = tool_map.get(engine_name, "ces_search")
    agent_input = f"'{refined}' 쿼리에 대해 '{selected_tool_name}' 도구를 사용하여 관련 정보를 검색하고 그 결과를 말하라."
    agent_input_dict = {"input": agent_input}

    try:
        logger.info(f"다음에 대해 Agent 실행중 : {agent_input}")
        agent_result = await agent.ainvoke(
            agent_input_dict, return_intermediate_steps=True
        )
        final_answer_str = agent_result.get("output", "").strip()  # Agent 최종 답변
        intermediate_steps = agent_result.get("intermediate_steps", [])

        # 마지막 '검색 도구' 실행 결과에서 observation 문자열과 링크 리스트 분리 저장
        if isinstance(intermediate_steps, list):
            for step in reversed(intermediate_steps):
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation_tuple = step
                    if isinstance(action, AgentAction) and hasattr(action, "tool"):
                        tool_name_in_action = action.tool
                        if tool_name_in_action in [
                            "serpapi_search",
                            "naver_search",
                            "ces_search",
                        ]:
                            if (
                                isinstance(observation_tuple, tuple)
                                and len(observation_tuple) == 2
                            ):
                                obs_str, src_links = observation_tuple
                                if isinstance(obs_str, str) and isinstance(
                                    src_links, list
                                ):
                                    agent_observation_for_factcheck = (
                                        obs_str  # 팩트체크용
                                    )
                                    original_source_links = src_links  # 원본 링크
                                    logger.info(
                                        f"본문 (len:{len(obs_str)}) and {len(src_links)} links from tool '{tool_name_in_action}'."
                                    )
                                    logger.debug(f"링크: {original_source_links}")
                                    break
                            else:
                                logger.warning(
                                    f"본문 타입 warning '{tool_name_in_action}'"
                                )
            if not original_source_links and intermediate_steps:
                logger.warning("원본 링크 추출 실패")
        if not agent_observation_for_factcheck:
            agent_observation_for_factcheck = final_answer_str  # Fallback

    except Exception as e:
        logger.error(f"Agent 실행 에러: {e}", exc_info=True)
        final_answer_str = f"Agent 실행 중 에러 발생: {e}"
        agent_observation_for_factcheck = ""
        original_source_links = []

    # 사용자 메시지 메모리 저장
    if memory:
        memory.chat_memory.add_user_message(query)

    # 5. 결과 파싱 - Agent 최종 답변에서 본문 추출
    extracted_content = final_answer_str  # 기본값은 Agent 원본 답변
    if final_answer_str and parse_agent_observation:
        parsed_body, _ = parse_agent_observation(final_answer_str)
        if parsed_body:
            extracted_content = parsed_body
        else:
            logger.warning("본문 파싱 실패")
    elif not final_answer_str:
        extracted_content = "(Agent 최종 답변 없음)"

    # 6. 답변 요약
    summary = extracted_content  # 파싱된 내용 또는 Agent 원본 답변
    if (
        extracted_content
        and not any(
            err_msg in extracted_content
            for err_msg in ["오류", "실패", "없음", "불가", "죄송합니다"]
        )
        and search_answer_chain
    ):
        logger.info("요약중...")
        try:
            summary_result = await search_answer_chain.ainvoke(
                {"content": extracted_content, "refined_query": refined}
            )
            temp_summary = summary_result.get("summary", "").strip()
            if temp_summary:
                summary = temp_summary
                logger.info(f"요약 성공 (len: {len(summary)}).")
            else:
                logger.warning("요약 빈 문자열 반환!")
        except Exception as e:
            logger.error(f"요약 체인 에러: {e}", exc_info=True)
    else:
        if not search_answer_chain:
            logger.warning("요약 체인 에러")
        logger.info("요약 스킵")

    # 팩트 체크
    checked_summary = summary  # observation 본문의 요약 결과
    logger.info(f"팩트 체크 시작 (len: {len(summary)})...")
    if fact_check_chain:
        try:
            history_text = "(이전 대화 없음)"
            if memory:
                history_messages = memory.chat_memory.messages[-2:]  # 메모리 고민 필요
                history_text = "\n".join(
                    [
                        f"{type(m).__name__}: {m.content}"
                        for m in history_messages
                        if isinstance(m, BaseMessage)
                    ]
                )
                if not history_text:
                    history_text = "(이전 대화 없음)"

            # 팩트체크 기준은 agent_observation_for_factcheck 사용
            agent_observation_body = "(검색된 본문 없음)"
            if isinstance(agent_observation_for_factcheck, str):  # 타입 확인
                stripped_observation = agent_observation_for_factcheck.strip()
                if stripped_observation:
                    agent_observation_body = stripped_observation
                else:
                    agent_observation_body = "(검색된 본문 내용 없음)"
            else:
                logger.warning(
                    f"agent_observation_for_factcheck 타입 에러: {type(agent_observation_for_factcheck)}"
                )

            combined_history = f"[검색된 본문]\n{agent_observation_body[:2000]}\n\n[최근 대화 기록]\n{history_text}"

            checked_result = await fact_check_chain.ainvoke(
                {"answer": summary, "history": combined_history}
            )
            temp_checked = checked_result.get("checked_answer", "").strip()

            if temp_checked and not any(
                err_msg in temp_checked
                for err_msg in ["오류", "정보 확인 불가", "수정 불가"]
            ):
                checked_summary = temp_checked  # 팩트체크 결과 반영
                logger.info(f"팩트 체크 진행 완료: {len(checked_summary)}")
            else:
                logger.warning(f"팩트 체크 실패. fallback")
        except Exception as e:
            logger.error(f"팩트 체크 에러: {e}", exc_info=True)
    else:
        logger.warning("팩트 체크 에러. fallback")

    # 8. 최종 답변 출력 탬플릿
    final_answer_body = checked_summary  # 팩트체크 완료된 (요약된) content

    final_answer_with_links = final_answer_body  # 최종 출력 content에 link 첨부를 위해

    if original_source_links:  # 이전에 저장한 link 덧붙이기
        unique_links = list(
            dict.fromkeys(
                link for link in original_source_links if link and isinstance(link, str)
            )
        )
        if unique_links:
            sources_text = "\n\n출처:\n" + "\n".join(
                [f"- {link}" for link in unique_links]
            )
            final_answer_with_links += sources_text
        else:
            logger.info("링크 추출 결과 없음")
    else:
        logger.info("링크 추출 실패. 스킵")

    # 최종 답변 메모리 저장
    if memory:
        try:
            # 메모리에는 링크 포함된 최종본을 저장
            memory.chat_memory.add_ai_message(final_answer_with_links)
        except Exception as mem_e:
            logger.error(f"대화 이력 추가 에러: {mem_e}")

    logger.info(
        f"Pipeline 실행완료. Final answer length: {len(final_answer_with_links)}"
    )
    # 링크가 따로 덧붙여진 최종 문자열만을 반환
    return final_answer_with_links
