import logging
from typing import Dict, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from tavily import TavilyClient

from .openai_llm import openai_gpt_5_4 as openai_llm
from .openai_prompt_templates import history_based_prompt

logger = logging.getLogger(__name__)
tavily: TavilyClient | None = None


def get_tavily_client() -> TavilyClient:
    global tavily
    if tavily is None:
        tavily = TavilyClient()
    return tavily


class Category(BaseModel):
    nationality: str = Field(description="유물의 국가. 예: 한국, 중국, 일본")
    period: str = Field(
        description="유물의 시대. 예: 신라, 고려, 조선. 통일신라는 신라로 표기"
    )
    genre: str = Field(
        description="유물 장르. 예: 건축, 조각(불상), 공예, 회화, 서예, 복식, 과학기술"
    )


tools = [
    {
        "name": "search_relics_by_period_and_genre",
        "description": "사용자가 시대와 장르를 모두 지정해 전시물 검색을 요청한 경우 사용",
        "parameters": Category.model_json_schema(),
    },
    {
        "name": "search_historical_facts",
        "description": "역사적 사실이나 배경 설명이 필요한 질문에 대해 참고 정보를 검색",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 엔진에 넣을 핵심 질의어",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def search_relics_by_period_and_genre(
    search_condition: dict, database: dict
) -> tuple[dict, str]:
    results = {}
    for relic_id, relic_data in database.items():
        relic_category: dict = relic_data["category"]
        if relic_category == search_condition:
            results[relic_id] = relic_data
    message = (
        f"요청하신 전시물이 {len(results)}건 검색되었습니다. [다음] 버튼으로 이동해 주세요."
        if len(results) > 0
        else "요청하신 전시물의 검색 결과가 없습니다."
    )
    return results, message


def search_historical_facts(query: str) -> tuple[list, list]:
    tavily_response = get_tavily_client().search(
        query=query,
        include_domains=["ko.wikipedia.org", "encykorea.aks.ac.kr"],
        max_results=3,
        search_depth="advanced",
    )
    logger.info(f"[query] {query}")
    references: list[tuple[str, str]] = []
    contents: list[str] = []
    for result in tavily_response["results"]:
        references.append((result["title"], result["url"]))
        contents.append(result["content"])
    return references, contents


class ToolData(TypedDict):
    type: Literal["relics", "facts"]
    items: dict | list[tuple[str, str] | bool]


def use_tools(
    messages: list, database: dict
) -> tuple[Optional[ToolData], Optional[Dict[str, str]]]:
    payload = openai_llm.choose_tool(messages=messages, tools=tools)
    if not payload or payload.get("type") != "tool_call":
        return None, None

    logger.info(f"[tool_payload] {payload}")
    tool_name = payload["name"]
    tool_args = payload.get("arguments", {})
    tool_data, message_dict = None, None

    if tool_name == "search_relics_by_period_and_genre":
        data, message = search_relics_by_period_and_genre(tool_args, database)
        tool_data = {"type": "relics", "items": data}
        message_dict = {"role": "assistant", "content": message}
    elif tool_name == "search_historical_facts":
        data, message = search_historical_facts(tool_args["query"])
        tool_data = {"type": "facts", "items": data}
        message_dict = {
            "role": "user",
            "content": history_based_prompt.format(history_facts=message),
        }

    if tool_data:
        logger.info(f"[tool_data type] {tool_data['type']}")
    logger.info(f"[message_dict] {message_dict}")
    return tool_data, message_dict
