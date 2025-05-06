import time
import os
import re
import json
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .base import SearchEngine
from readability import Document


class CesEngine(SearchEngine):
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        search_dir_path = os.path.dirname(current_file_path)
        app_root_path = os.path.dirname(search_dir_path)
        self.SERVICE_ACCOUNT_FILE = os.path.join(
            app_root_path, "google-search-api.json"
        )

        self.SCOPES = ["https://www.googleapis.com/auth/cse"]
        self.CSE_ID = ""

        try:
            if not os.path.exists(self.SERVICE_ACCOUNT_FILE):
                print(f"[CES] 서비스 계정 파일 호출 에러: {self.SERVICE_ACCOUNT_FILE}")
                raise FileNotFoundError(
                    f"계정 파일 호출 에러: {self.SERVICE_ACCOUNT_FILE}"
                )
            credentials = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
            )
            self.service = build("customsearch", "v1", credentials=credentials)
            print(f"[CES] 계정 초기화 성공 : {self.SERVICE_ACCOUNT_FILE}")
        except FileNotFoundError as fnf_e:
            print(fnf_e)
            self.service = None
        except Exception as e:
            print(f"[CES] 계정 초기화 실패: {e}")
            self.service = None

    def search(self, query, start=1, num_results=1):
        if not self.service:
            print("[CES] 검색 엔진 초기화 실패")
            return []

        try:
            res = (
                self.service.cse()
                .list(q=query, cx=self.CSE_ID, start=start, num=num_results)
                .execute()
            )
            items = res.get("items", [])
            return [
                {"title": i.get("title", ""), "link": i.get("link", "")}
                for i in items
                if i.get("link")
            ]
        except Exception as e:
            print(f"[CES] 검색 엔진 구조 에러: {e}")
            return []

    def extract_text(self, url):
        driver = self._create_driver()
        try:
            driver.set_page_load_timeout(35)
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return driver.page_source
        except Exception as e:
            print(f"[CES] 검색 엔진 추출 에러 {e}")
            return ""
        finally:
            driver.quit()

    # 환경설정
    def _create_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
        chromedriver_path = "/usr/local/bin/chromedriver"
        service = Service(executable_path=chromedriver_path)
        return webdriver.Chrome(service=service, options=chrome_options)

    # HTML 본문 텍스트 추출
    def extract_main_text_from_html(self, html):
        # 1) Readability 우선 추출
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

        text_elements = soup.find_all(text=True)

        def is_visible(element):
            return element.parent.name not in [
                "style",
                "script",
                "head",
                "meta",
                "[document]",
            ] and bool(element.strip())

        visible_texts = filter(is_visible, text_elements)
        cleaned_lines = []
        for t in visible_texts:
            line = (
                t.strip()
                .replace("\u200d", "")
                .replace("\xa0", " ")
                .replace("\ufeff", "")
            )
            line = re.sub(r"\s+", " ", line)
            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
