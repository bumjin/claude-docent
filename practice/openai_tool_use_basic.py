import json
from datetime import datetime

import pytz

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


def get_time(timezone: str) -> dict:
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return {
            "timezone": timezone,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone_offset": current_time.strftime("%z"),
            "timezone_name": tz.zone,
        }
    except pytz.exceptions.UnknownTimeZoneError:
        return {"error": f"알 수 없는 시간대입니다: {timezone}"}


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
    },
    {
        "name": "get_time",
        "description": "IANA 시간대 기준 현재 시간을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "예: Asia/Seoul",
                }
            },
                "required": ["timezone"],
                "additionalProperties": False,
            },
    },
]

tool_repository = {
    "get_weather": get_weather,
    "get_time": get_time,
}

client = build_client()
model = get_model()

final_answer, tool_trace = run_local_tool_call_loop(
    client=client,
    model=model,
    user_prompt="서울의 날씨와 Asia/Seoul 현재 시간을 함께 알려줘.",
    tools=tools,
    tool_repository=tool_repository,
)

print("[도구 호출 내역]")
print(json.dumps(tool_trace, indent=2, ensure_ascii=False))
print("\n[최종 응답]")
print(final_answer)
