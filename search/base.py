from abc import ABC, abstractmethod


# 검색 엔진의 기본 골격


class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str) -> list:
        """
        사용자 질의를 받아 검색 결과 리스트를 반환
        각 추출 결과는 {"title": str, "link": str} 형식의 dict로 구성
        """
        pass

    @abstractmethod
    def extract_text(self, url: str) -> str:
        """
        주어진 URL에서 HTML 전체 소스를 반환
        이후 전처리 후 → LLM 전달에 활용
        """
        pass
