import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .base import SearchEngine
from readability import Document
import logging

logger = logging.getLogger(__name__)


class SerpapiEngine(SearchEngine):
    def __init__(self):
        env_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)
        if env_path and os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            logger.info(f".env 파일 로드 : {env_path}")
        else:
            logger.warning(".env 로드 실패")

        self.api_key = os.getenv("Serp_API_KEY")

    def search(self, query: str):
        if not self.api_key:
            print("[SerpAPI] API 키 에러")
            return {}

        params = {"q": query, "api_key": self.api_key, "engine": "google", "num": 1}

        try:
            response = requests.get(
                "https://serpapi.com/search", params=params, timeout=20
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[SerpAPI] 요청 실패: {e}")
            return {}
        except Exception as e:
            print(f"[SerpAPI] 기타 에러: {e}")
            return {}

    def extract_text(self, url: str) -> str:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("user-agent=Mozilla/5.0")
        driver = None

        try:
            chromedriver_path = "/usr/local/bin/chromedriver"
            service = Service(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(35)
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return driver.page_source
        except Exception as e:
            print(f"[HTML] 셀레니움 추출 에러: {e}")
            try:
                resp = requests.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
                )
                resp.raise_for_status()
                return resp.text
            except Exception as req_e:
                print(f"[HTTP] requests 요청 실패: {req_e}")
                return ""
        finally:
            if driver:
                driver.quit()

    # 본문 추출 우선순위
    def extract_main_text_from_html(self, html: str) -> str:
        # 1) Readability
        try:
            doc = Document(html)
            main_html = doc.summary()
            text = BeautifulSoup(main_html, "html.parser").get_text(
                separator="\n", strip=True
            )
            return self._clean_text(text)
        except Exception:
            pass

        # 2) 기존 fallback - 아래 태그 제거 후, 본문 전체 추출
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "form", "nav", "aside"]
        ):
            tag.decompose()
        texts = soup.find_all(text=True)
        visible = filter(
            lambda el: el.parent.name
            not in ["style", "script", "head", "meta", "[document]"]
            and el.strip(),
            texts,
        )
        lines = []
        for t in visible:
            line = (
                t.strip()
                .replace("\u200d", "")
                .replace("\xa0", " ")
                .replace("\ufeff", "")
            )
            line = re.sub(r"\s+", " ", line)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        # 공통 간단 전처리
        return re.sub(r"\s+", " ", text).strip()

    # API 형식 별 추출 - Only SerpAPI
    def handle_response(self, response_json):
        # Query에 따른 동기적 흐름
        # 1) Answer Box
        if "answer_box" in response_json:
            box = response_json["answer_box"]

            # 날씨
            if box.get("type") == "weather_result":
                loc = box.get("location", "정보 없음")
                cond = box.get("weather", "정보 없음")
                temp = box.get("temperature", "정보 없음")
                unit = box.get("unit", "")
                return (
                    f"날씨 정보\n"
                    f"지역: {loc}\n"
                    f"날씨: {cond}\n"
                    f"기온: {temp} {unit}"
                )
            # 주가
            if box.get("type") == "finance_results" and "price" in box:
                return (
                    f"주가 정보 (SerpAPI - answer_box)\n"
                    f"종목: {box.get('title','N/A')} ({box.get('stock','N/A')})\n"
                    f"거래소: {box.get('exchange','N/A')}\n"
                    f"현재가: {box.get('price','N/A')} {box.get('currency','')}\n"
                    f"전일 종가: {box.get('previous_close','N/A')}"
                )
            # 간단 정보 -
            answer = (
                box.get("answer")
                or box.get("snippet")
                or " / ".join(box.get("highlighted_words", []))
                or box.get("title")
                or "정보 없음"
            )
            return f"간단 정보\n{answer}"

        # 2) Knowledge Graph
        if "knowledge_graph" in response_json:
            kg = response_json["knowledge_graph"]
            title = kg.get("title", "정보 없음")
            desc = kg.get("description", "설명 없음")
            return f"지식 카드\n{title}: {desc}"

        # 3) Organic Results – 일반 웹 검색
        if "organic_results" in response_json and response_json["organic_results"]:
            item = response_json["organic_results"][0]
            title = item.get("title", "")
            link = item.get("link", "")
            html = self.extract_text(link)
            text = self.extract_main_text_from_html(html)
            return f"웹 검색\n제목: {title}\n링크: {link}\n\n본문:\n{text}..."

        return "검색 결과 없음."
