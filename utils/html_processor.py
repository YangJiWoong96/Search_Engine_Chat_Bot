import re


def preprocess_html(text: str, url: str = "") -> str:

    # 네이버 블로그일 경우 SmartEditor 2.x~4.x 구조 기반의 본문에서만 추출 시도
    def extract_naver_blog_body(text):
        # SmartEditor 4.0: SE-TEXT 기반
        se_blocks = re.findall(r"SE-TEXT\s*{(.*?)}\s*SE-TEXT", text, re.DOTALL)
        if se_blocks:
            blocks = [re.sub(r"<[^>]+>", "", b).strip() for b in se_blocks if b.strip()]
            return "\n".join(blocks)

        # SmartEditor 2.x~3.x: postViewArea or se2_textView 기반
        match = re.search(
            r'<div[^>]+id="postViewArea"[^>]*>(.*?)</div>', text, re.DOTALL
        )
        if match:
            inner_html = match.group(1)
            return re.sub(r"<[^>]+>", "", inner_html)

        return None  # 구조 파싱 실패

    # 네이버 블로그일 경우, 위 구조 기반 content 추출
    if "blog.naver.com" in url:
        body_text = extract_naver_blog_body(text)
        if body_text:
            text = body_text

    # 전체 적용 텍스트 전처리

    # 태그 제거
    cleaned = re.sub(r"<[^>]+>", "", text)
    # 특수문자 제거
    cleaned = re.sub(r"[^\w가-힣\s\.,\?\!]", "", cleaned, flags=re.UNICODE)
    # 연속 공백 하나로
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # 2000자 길이 clipping
    return cleaned[:2000]
