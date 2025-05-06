import re
import asyncio
from urllib.parse import urlparse
import logging

# utils 폴더 내의 html_processor 모듈에서 def preprocess_html 호출
try:
    from .html_processor import preprocess_html
except ImportError:
    logging.error("html_processor 모듈에서 def preprocess_html 확인 요망")

    # 만약에 못 받아오면, 아래로
    def preprocess_html(text: str, url: str = "") -> str:

        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:2000]


logger = logging.getLogger(__name__)


# _extract_and_process_item 함수
async def _extract_and_process_item(engine, item):
    """개별 item(dict)을 받아 text와 link를 비동기로 추출 및 처리"""
    link = item.get("link")
    title = item.get("title", "")
    if not link:
        logger.debug(f"Item '{title}' has no link.")
        return None, None
    logger.debug(f"Processing item: '{title}' - {link}")
    try:
        # engine.extract_text 는 blocking I/O (requests, selenium) 사용 가정
        html = await asyncio.to_thread(engine.extract_text, link)
        if html:
            # engine.extract_main_text_from_html 은 CPU-bound (파싱) + I/O 가정
            main_text = await asyncio.to_thread(
                engine.extract_main_text_from_html, html
            )
            # preprocess_html 은 CPU-bound (정규식 처리) 가정
            processed_text = await asyncio.to_thread(
                preprocess_html, main_text, url=link  # preprocess_html 호출
            )
            if processed_text:
                logger.info(f"다음 링크에서 HTML 추출 성공: {link}")
                return f"--- 문서 ({title}) ---\n{processed_text}", link
            else:
                logger.warning(f"다음 링크의 전처리 결과가 빈 문자열: {link}")
        else:
            logger.warning(f"다음 링크에서 HTML 추출 실패: {link}")
        return None, None
    except Exception as e:
        logger.error(f"다음 구조의 파싱 에러 '{title}' ({link}): {e}", exc_info=True)
        return None, None


# format_search_results 함수
def format_search_results(processed_texts: list, links: list) -> str:
    """추출된 텍스트 리스트와 링크 리스트를 지정된 문자열 형식으로 변환"""
    if not processed_texts:
        logger.info("No processed texts to format.")
        return "관련 내용을 찾거나 추출하지 못했습니다."
    content_str = "\n\n".join(processed_texts)

    unique_links = []
    if links:
        # 링크 중복 제거 및 순서 유지
        unique_links = list(
            dict.fromkeys(link for link in links if link)
        )  # None 제거 포함

    if unique_links:
        link_list_str = "\n".join([f"- {link}" for link in unique_links])
        logger.info(
            f"Formatted search results with {len(processed_texts)} texts and {len(unique_links)} unique links."
        )
        return f"본문:\n{content_str}\n\n출처:\n{link_list_str}"
    else:
        logger.info(
            f"Formatted search results with {len(processed_texts)} texts and no links."
        )
        return f"본문:\n{content_str}"


# parse_agent_observation 함수
def parse_agent_observation(observation: str) -> tuple[str, list[str]]:
    """
    Agent가 반환한 문자열에서 본문 내용과 출처 링크 리스트를 분리합니다.
    URL 끝에 붙은 불필요한 문자(조사, 구두점, Markdown 닫는 괄호 등) 제거 로직을 강화합니다.
    """
    content = observation
    links = []
    if not isinstance(observation, str):
        logger.warning("[Parser] Input is not a string.")
        return "파싱 불가", []

    logger.debug(
        f"[Parser] Parsing observation (length: {len(observation)}):\n'''{observation[:200]}...'''"
    )

    try:
        # 1. URL 추출
        markdown_links = re.findall(r"\[.*?\]\((https?://.*?)\)", observation)
        plain_links = re.findall(r"(?<!\]\()(https?://[^\s\"'<>]+)", observation)
        found_links_raw = list(dict.fromkeys(markdown_links + plain_links))
        logger.debug(
            f"[Parser] Raw extracted links (Markdown + Plain): {found_links_raw}"
        )

        # 2. 링크 정제
        cleaned_links = []
        # 정규식: 선택적 닫는 괄호 + 선택적 공백 + 선택적 (조사 + 선택적 공백 + 선택적 구두점) 반복 + 문자열 끝
        trailing_junk_regex = re.compile(
            r"(\)?\s*(?:에서|이?와|이?과|으?로|의|이|가|은|는)?\s*[.,;\'\"]*)+$"
        )
        # 단순 후행 괄호/공백/구두점 제거용
        simple_trailing_junk_regex = re.compile(r"[)\s.,;\'\"]+$")

        for link in found_links_raw:
            if not link:
                continue
            cleaned = link.strip()
            original_cleaned = cleaned
            cleaned = trailing_junk_regex.sub("", cleaned)
            cleaned = simple_trailing_junk_regex.sub("", cleaned)

            try:
                result = urlparse(cleaned)
                if all([result.scheme, result.netloc]):
                    if cleaned not in cleaned_links:
                        cleaned_links.append(cleaned)
            except ValueError:
                logger.warning(f"[Parser] 잘못된 URL 형식 예외 처리: {cleaned}")

        links = cleaned_links
        logger.info(f"[Parser] 최종 추출된 링크 수: {len(links)}")

        # 3. 본문 내용 추출 로직
        content_part = observation
        match_source = re.search(
            r"(.*?)(?:\n*\s*(?:출처|Sources)\s*:\s*\n*)(.*)",
            observation,
            re.DOTALL | re.IGNORECASE,
        )
        if match_source:
            content_part = match_source.group(1).strip()

        if re.match(r"^본문\s*:\s*\n?", content_part, re.IGNORECASE):
            content = re.sub(
                r"^본문\s*:\s*\n?", "", content_part, flags=re.IGNORECASE
            ).strip()
        else:
            content = content_part

        final_answer_match = re.search(
            r"Final Answer:\s*(.*)", observation, re.DOTALL | re.IGNORECASE
        )
        if final_answer_match:
            possible_content = final_answer_match.group(1).strip()
            if links and len(possible_content) < len(content_part):
                content = possible_content 

            elif not links and possible_content:
                content = possible_content
                logger.info("[Parser] 'Final Answer:' 본문에서 링크 추출 실패")

    except Exception as e:
        logger.error(f"[Parser] observation 파싱 에러: {e}", exc_info=True)
        content = observation.strip()
        links = []  # 파싱 실패 시 링크는 비움

    if not content:
        content = "결과에서 유효한 내용을 찾지 못했습니다."

    logger.debug(f"[Parser] 최종 본문 길이: {len(content)}")
    return content, links
