import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .openai_prompt_templates import (
    system_prompt as default_system_prompt,
    tool_system_prompt as default_tool_system_prompt,
)

logger = logging.getLogger(__name__)


def build_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )


def get_model(default: str = "gpt-5.4") -> str:
    load_dotenv()
    return os.getenv("OPENAI_MODEL", default)


def extract_first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("첫 JSON 값이 객체가 아닙니다.")
    return payload


class OpenAILLM:
    def __init__(self, model_name: str, system_prompt: str, tool_system_prompt: str):
        self.client = build_client()
        self.model = model_name
        self.system_prompt = system_prompt
        self.tool_system_prompt = tool_system_prompt

    def create_response_text(
        self,
        messages: list,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        try:
            openai_messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                *messages,
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[openai create_response_text error] {e}")
            raise

    def choose_tool(
        self,
        messages: list,
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_system_prompt: str | None = None,
    ) -> dict | None:
        try:
            tool_summaries: list[str] = []
            for tool in tools:
                tool_summaries.append(
                    json.dumps(
                        {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("input_schema")
                            or tool.get("parameters", {}),
                        },
                        ensure_ascii=False,
                    )
                )

            system_text = (
                (tool_system_prompt or self.tool_system_prompt)
                + "\n\n"
                + "당신은 도구 선택기입니다. 반드시 JSON 객체 하나만 반환하세요. "
                + '도구를 써야 하면 {"type":"tool_call","name":"...","arguments":{...}} 를 반환하고, '
                + '도구가 필요 없으면 {"type":"no_tool"} 를 반환하세요.\n'
                + "사용 가능한 도구 목록:\n"
                + "\n".join(tool_summaries)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_text},
                    *messages,
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            payload = extract_first_json_object(content)
            if payload.get("type") == "no_tool":
                return None
            return payload
        except Exception as e:
            logger.error(f"[openai choose_tool error] {e}")
            raise


openai_gpt_5_4 = OpenAILLM(
    model_name=get_model("gpt-5.4"),
    system_prompt=default_system_prompt,
    tool_system_prompt=default_tool_system_prompt,
)
