# web/app.py
import streamlit as st
import requests
import os
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 백엔드 API 환경설정
# 환경 변수에서 API URL을 가져오거나 기본값 사용
# Docker Compose 사용 시 서비스 이름 사용 가능 (예: 'http://backend:8000')
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")
PROCESS_ENDPOINT = f"{BACKEND_URL}/process"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"  # Health check 엔드포인트 추가


# API 연결 환경 확인
def check_api_health():
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return True, response.json().get("message", "API is running.")
        else:
            return (
                False,
                f"API health check failed with status {response.status_code}: {response.text}",
            )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to connect to API at {HEALTH_ENDPOINT}: {e}"


# Streamlit 사용자 UI 구성
st.set_page_config(page_title="검색엔진 챗봇", page_icon="", layout="wide")
st.title("🤖 인터넷 검색 엔진을 활용한 실시간 질의 Chat Bot")

# 백앤드 API 상태 확인
api_ok, api_status_msg = check_api_health()
if api_ok:
    st.success(f"백엔드 API 연결 성공: {api_status_msg}")
else:
    st.error(f"백엔드 API 연결 실패: {api_status_msg}")
    st.warning(
        "백엔드 서버가 실행 중인지, API_URL 환경 변수가 올바르게 설정되었는지 확인하세요."
    )
    # API 연결 실패시 STOP
    st.stop()

st.markdown("---")

# 대화 기록 초기화 UI
if st.button("대화 기록 초기화"):
    st.session_state.messages = []
    st.success("대화 기록이 초기화되었습니다.")
    st.rerun()  # 화면 새로고침

# 세션 상태 초기화 UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 UI
prompt = st.chat_input("여기에 질문을 입력하세요...")

if prompt:
    # 사용자 query 표시 및 기록
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 로딩중 표시 및 API 호출
    with st.spinner("AI가 답변을 생성하고 있습니다... 잠시만 기다려주세요."):
        try:
            # FastAPI에 POST 요청
            response = requests.post(
                PROCESS_ENDPOINT,
                json={"query": prompt},
                timeout=180,  # 검색엔진 호출 시간 고려
            )
            response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

            # 응답 성공
            api_response = response.json()
            answer = api_response.get("answer")

            if answer:
                # 챗봇 응답 표시 및 기록
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                # 응답은 성공. But, answer 키가 없는 경우
                st.error("오류: API로부터 유효한 답변을 받지 못했습니다.")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "오류: 답변 형식이 잘못되었습니다.",
                    }
                )

        # 오류 처리 - timeout
        except requests.exceptions.Timeout:
            st.error("오류: 백엔드 서버 응답 시간 초과. 잠시 후 다시 시도해주세요.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "오류: 응답 시간 초과"}
            )
        except requests.exceptions.HTTPError as http_err:
            error_detail = "알 수 없는 오류"
            try:  # 상세 error 파싱 시도
                error_detail = http_err.response.json().get(
                    "detail", http_err.response.text
                )
            except:
                error_detail = http_err.response.text
            st.error(
                f"오류: API 요청 실패 (HTTP {http_err.response.status_code}): {error_detail}"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"오류: API 요청 실패 ({http_err.response.status_code})",
                }
            )
        except requests.exceptions.RequestException as req_err:
            st.error(f"오류: 백엔드 서버 연결 실패. 서버 주소를 확인하세요: {req_err}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "오류: 서버 연결 실패"}
            )
        except Exception as e:
            st.error(f"알 수 없는 오류 발생: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "오류: 알 수 없는 문제 발생"}
            )
