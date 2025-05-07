## 인터넷 검색 기반 실시간 질의응답 챗봇 구조

* 이 챗봇은 사용자 질문을 받아, 검색이 필요한 경우 인터넷에서 실시간으로 정보를 수집하고, LLM이 요약·검증해서 최종 답변을 생성하는 RAG (Retrieval-Augmented Generation)? 시스템이다.

* 핵심 파일: core/pipeline.py
    * 모든 메커니즘을 통제. 
    * LangChain으로 구성된 체인과 Agent를 orchestrate. - 실험용 : 과업에 비해 조금 무거운 감이 있음 

#### 주요구성 
    * 환경설정 : env. 로 관리 
        * GCP - credential.json(API 관련), CSE ID key, Naver Client ID key, SerpAPI key, GPT key 필요 
    * ChatOpenAI(GPT) - LLM Chain 구성 
    * Search Engine : Basic Engine(골격),SerpapiEngine, NaverEngine, CesEngine

* Agent 
    * Search Engine을 Tool로 관리
    * ConversationBufferMemory history 관리 - 현재 활용 X
    * LangChain Agent(ReAct)
    * Tool.search() → 검색
    * Tool.extract_text() → HTML 추출
    * Tool.extract_main_text_from_html() → 전처리
    * preprocess_html() → 전처리

* 구조도

```
📦project-root
├── core/
│   └── pipeline.py              전체 파이프라인 제어 (검색 여부 판단~최종 응답 생성)
├── search/
│   ├── base_engine.py           모든 검색엔진의 공통 인터페이스
│   ├── ces.py                   Google CSE API + Selenium 기반 
│   ├── naver.py                 Naver API + 블로그/뉴스 본문 추출 특화
│   └── serpapi.py               SerpAPI + AnswerBox(UI)/KnowledgeGraph 우선 파싱
├── utils/
│   ├── helpers.py               비동기 검색 실행 및 결과 파싱/정제 함수들
│   └── html_processor.py        HTML 본문 텍스트 정제 (readability, fallback 포함)
├── api/
│   ├── main.py                  FastAPI 서버 실행부 (/process, /health API 제공)
│   └── schemas.py               Pydantic 기반 요청/응답 모델 정의
├── web/
│   └── app.py                   Streamlit UI (입력 → 백엔드 호출 → 응답 출력)
├── config/
│   └── settings.py              .env 환경설정 
├── docker/
│   └── backend.Dockerfile       Selenium 포함된 백엔드 Docker 이미지
├── .env                         API 키 및 설정값 저장
└── README.md                    
```


* 간단 Pipeline
```
사용자 질문
  ↓
run_pipeline()
  ├─ decide_chain
  ├─ (검색 필요 시) refine_chain(query) → choose_chain(search engine)
  ├─ Agent 실행 (검색 + 정제)
  ├─ parse → 요약 → 팩트체크
  └─ 최종 답변 + 출처 조립
  ```