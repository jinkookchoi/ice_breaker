# from langchain_community.utilities import SerpAPIWrapper
# class CustomSerpAPIWrapper(SerpAPIWrapper):
#     def __init__(self):
#         super(CustomSerpAPIWrapper, self).__init__()
#
#     @staticmethod
#     def _process_response(res: dict) -> str:
#         """Process response from SerpAPI."""
#         if "error" in res.keys():
#             raise ValueError(f"Got error from SerpAPI: {res['error']}")
#         if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
#             toret = res["answer_box"]["answer"]
#         elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
#             toret = res["answer_box"]["snippet"]
#         elif (
#             "answer_box" in res.keys()
#             and "snippet_highlighted_words" in res["answer_box"].keys()
#         ):
#             toret = res["answer_box"]["snippet_highlighted_words"][0]
#         elif (
#             "sports_results" in res.keys()
#             and "game_spotlight" in res["sports_results"].keys()
#         ):
#             toret = res["sports_results"]["game_spotlight"]
#         elif (
#             "knowledge_graph" in res.keys()
#             and "description" in res["knowledge_graph"].keys()
#         ):
#             toret = res["knowledge_graph"]["description"]
#         elif "snippet" in res["organic_results"][0].keys():
#             toret = res["organic_results"][0]["link"]
#
#         else:
#             toret = "No good search result found"
#         return toret
#
#
# def get_profile_url(name: str):
#     """Searches for Linkedin or twitter Profile Page."""
#     search = CustomSerpAPIWrapper()
#     res = search.run(f"{name}")
#     return res

from typing import Union
from langchain_community.tools.tavily_search import TavilySearchResults
from loguru import logger
import urllib.parse


# Note. Travily AI api key
# https://app.tavily.com/home
def get_profile_url_tavily(name: str) -> Union[str, None]:
    """Searches for Linkedin or twitter Profile Page."""
    logger.info(name)
    search = TavilySearchResults(max_results=5)
    res = search.run(f"{urllib.parse.quote(name)}")
    logger.debug(res)

    for r in res:
        logger.debug(r.get("url"))

    # Note. old code
    # url: Union[str, None] = res[0].get("url", None) if res else None
    # return url

    # Match linkedin.com/in URL
    _include: str='linkedin.com/in'
    for r in res:
        url: Union[str, None] = r.get("url", None)
        logger.debug(f"Found URL: {url}")
        if url is not None and _include in url:
            logger.debug(f"Matched URL: {url}")
            return url
    
    logger.debug(f"No matching URL found for inclusion: {_include}")

    return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    url = get_profile_url_tavily("jinkook choi")
    logger.debug(url)
