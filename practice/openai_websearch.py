import json
import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv(".env.local")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

user_query = "국립중앙박물관에서 가을에 보기 좋은 전시물을 알려줘."

response = tavily.search(
    query=user_query,
    topic="general",
    search_depth="advanced",
    max_results=5,
    include_answer="advanced",
    include_domains=["museum.go.kr", "www.museum.go.kr"],
)

print("[Tavily answer]")
print(response.get("answer", ""))
print("\n[Tavily raw]")
print(json.dumps(response, indent=2, ensure_ascii=False))

print("\n[Sources]")
for result in response.get("results", []):
    print(f"{result['title']}: {result['url']}")
