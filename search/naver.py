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


class NaverEngine(SearchEngine):
    def __init__(self):
        env_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)
        if env_path and os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            logger.info(f".env 파일 로드 성공: {env_path}")
        else:
            logger.warning(".env 파일 로드 실패")

        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            logger.error("[Naver] CLIENT_ID or CLIENT_SECRET 계정 에러")
        # 검색 서비스 키워드 매핑 (동적 서비스 선택)
        self.service_map = {
            "news": ["뉴스", "기사", "보도", "언론"],
            "book": ["책", "도서", "출판"],
            "encyc": ["백과사전", "사전", "백과"],
            "cafearticle": ["카페", "카페글", "카페 포스트"],
            "kin": ["지식인", "질문", "답변"],
            "webkr": ["웹문서", "사이트", "페이지"],
            "image": ["이미지", "사진", "그림"],
            "shop": ["쇼핑", "상품", "조회"],
            "doc": ["전문자료", "논문", "리포트"],
            "adult": ["성인", "19금", "성인물"],
            "errata": ["오타", "교정", "정정"],
        }
        # 기본 웹 검색 엔진
        self.fallback_service = "webkr"
        self.num_results = 3  # 최대 호출 웹

    def detect_service(self, query: str) -> str:
        text = query.lower()
        for svc, keywords in self.service_map.items():
            for kw in keywords:
                if kw in text:
                    return svc
        return self.fallback_service

    def search(self, query: str, service: str = None):
        # 동적 서비스 결정
        service_id = (
            service if service in self.service_map or service == "webkr" else None
        )
        if not service_id:
            service_id = self.detect_service(query) if not service else service

        url = f"https://openapi.naver.com/v1/search/{service_id}.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {"query": query, "display": self.num_results}  # 1

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            return [
                {
                    "title": re.sub("<.*?>", "", item.get("title", "")),
                    "link": item.get("link", ""),
                }
                for item in items
                if item.get("link")
            ]

        except Exception as e:
            print(f"[Naver] API 호출 에러 {e}")
            return []

    # 환경설정
    def extract_text(self, url: str) -> str:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("user-agent=Mozilla/5.0")
        driver = None
        try:
            # Dockerfile ChromeDriver 경로
            chromedriver_path = "/usr/local/bin/chromedriver"
            service = Service(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(35)
            driver.get(url)

            # 네이버 블로그 대기
            if "blog.naver.com" in url:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "iframe"))
                )
            # 네이버 뉴스 대기
            elif "news.naver.com" in url or "n.news.naver.com" in url:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#newsEndContents, #articleBodyContents")
                    )
                )
            else:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            return driver.page_source

        except Exception as e:
            print(f"[HTML] 셀레니움 추출 실패: {e}")
            return ""
        finally:
            # driver 제대로 생성되었을때만
            if driver:
                driver.quit()

    # content 추출 우선순위
    def extract_main_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # 1) 사이트별 지정 변수명으로 추출
        selectors = [
            "#newsEndContents",
            "#articleBodyContents",
            "article",
            ".news_read_area",
            ".article_body",
            ".news-content",
            "#content",
            ".post-content",
        ]
        for sel in selectors:
            container = soup.select_one(sel)
            if container:
                return self._clean_text(container.get_text(separator="\n", strip=True))
        # 2) Readability
        try:
            doc = Document(html)
            main_html = doc.summary()
            return self._clean_text(BeautifulSoup(main_html, "html.parser").get_text())
        except Exception:
            pass
        # 3) 기본 fallback
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
        return self._clean_text("\n".join(t.strip() for t in visible))

    def _clean_text(self, text: str) -> str:
        text = text.replace("\u200d", "").replace("\xa0", " ").replace("\ufeff", "")
        return re.sub(r"\s+", " ", text).strip()
