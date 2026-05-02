import json
from pprint import pprint

from openai_common import build_client, get_model
from openai_tool_helpers import _extract_first_json_object


schema = {
    "type": "object",
    "properties": {
        "applicants": {
            "type": "array",
            "description": "지원자 객체 배열",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "이름"},
                    "gender": {
                        "type": "string",
                        "enum": ["M", "F"],
                        "description": "지원자의 성별",
                    },
                    "age": {"type": ["integer", "null"], "description": "나이"},
                    "major": {"type": "string", "description": "전공"},
                },
                "required": ["name", "gender", "age", "major"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["applicants"],
    "additionalProperties": False,
}

user_message = """
이번 대학교 입시 지원자들에 대해 다음 정보를 정리해 주세요.
먼저 김지우는 여자이고 나이는 스물다섯이며 전공은 컴퓨터공학입니다.
박현우는 남자이고 서른 살이며 전공은 산업공학이라고 했습니다.
마지막으로 이서연은 여자이고 스물셋이며 경영학을 전공한다고 합니다.
""".strip()

client = build_client()
model = get_model()

system_prompt = f"""
당신은 구조화된 정보 추출기입니다.
반드시 JSON만 반환하세요. 설명 문장, 마크다운, 코드 블록을 넣지 마세요.
아래 JSON Schema를 만족하는 객체 하나만 반환하세요.

JSON Schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
)

content = response.choices[0].message.content or ""
payload = _extract_first_json_object(content)
pprint(payload["applicants"])
