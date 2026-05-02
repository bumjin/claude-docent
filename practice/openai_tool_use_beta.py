import json

from openai_common import build_client, get_model
from openai_tool_helpers import run_local_tool_call_loop


def get_weather(location: str, unit: str = "celsius") -> dict:
    return {
        "location": location,
        "temperature": 20 if unit == "celsius" else 68,
        "unit": unit,
        "condition": "맑음",
        "humidity": 60,
    }


tools = [
    {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "도시 이름. 예: Seoul",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "온도 단위",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
    }
]

tool_repository = {"get_weather": get_weather}

client = build_client()
model = get_model()

final_answer, tool_trace = run_local_tool_call_loop(
    client=client,
    model=model,
    user_prompt="서울 날씨가 어떤지 간단히 알려줘.",
    tools=tools,
    tool_repository=tool_repository,
)

print("[도구 호출 내역]")
print(json.dumps(tool_trace, indent=2, ensure_ascii=False))
print("\n[최종 응답]")
print(final_answer)
