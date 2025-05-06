# config/settings.py
import os
import logging
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)

if env_path and os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    logging.info(f".env 로드 성공: {env_path}")
else:
    logging.warning(".env 로드 실패")

# 환경변수
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CSE_ID = os.getenv("CSE_ID")
GOOGLE_APPLICATION_CREDENTIALS_FILENAME = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

NAVER_CLIENT_ID = os.getenv("CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("CLIENT_SECRET")

SERPAPI_API_KEY = os.getenv("Serp_API_KEY")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# settings.py 절대 경로
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))  # /path/to/app/config
APP_ROOT_DIR = os.path.dirname(CONFIG_DIR)  # /path/to/app

# Google Credentials JSON 파일 절대 경로
GOOGLE_CREDENTIALS_PATH = None
if GOOGLE_APPLICATION_CREDENTIALS_FILENAME:
    possible_path = os.path.join(APP_ROOT_DIR, GOOGLE_APPLICATION_CREDENTIALS_FILENAME)
    if os.path.exists(possible_path):
        GOOGLE_CREDENTIALS_PATH = possible_path
        logging.info(
            f"Google credentials JSON 파일 절대 경로 : {GOOGLE_CREDENTIALS_PATH}"
        )
    else:
        logging.error(
            f"Google credentials JSON 파일 '{GOOGLE_APPLICATION_CREDENTIALS_FILENAME}' not found in app root: {APP_ROOT_DIR}"
        )
else:
    logging.warning("GOOGLE_APPLICATION_CREDENTIALS JSON 파일 불러오기 실패")

# 설정값 누락 확인
missing_keys = []
if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")
if not CSE_ID:
    missing_keys.append("CSE_ID")
if not GOOGLE_CREDENTIALS_PATH:
    missing_keys.append("GOOGLE_APPLICATION_CREDENTIALS file path")
if not NAVER_CLIENT_ID:
    missing_keys.append("NAVER_CLIENT_ID")
if not NAVER_CLIENT_SECRET:
    missing_keys.append("NAVER_CLIENT_SECRET")
if not SERPAPI_API_KEY:
    missing_keys.append("SERPAPI_API_KEY")

if missing_keys:
    logging.warning(f"설정 누락 에러: {', '.join(missing_keys)}")
