import json
from pprint import pprint
from typing import Literal

from pydantic import BaseModel, Field

from openai_common import build_client, get_model
from openai_tool_helpers import _extract_first_json_object


class Applicant(BaseModel):
    name: str = Field(description="이름")
    gender: Literal["M", "F"] = Field(description="지원자의 성별")
    age: int | None = Field(default=None, description="나이")
    major: str = Field(description="전공")


class Applicants(BaseModel):
    applicants: list[Applicant] = Field(description="지원자 목록")


client = build_client()
model = get_model()

user_message = """
이번 대학교 입시 지원자들에 대해 다음 정보를 정리해 주세요.
먼저 김지우는 여자이고 나이는 스물다섯이며 전공은 컴퓨터공학입니다.
박현우는 남자이고 서른 살이며 전공은 산업공학이라고 했습니다.
마지막으로 이서연은 여자이고 스물셋이며 경영학을 전공한다고 합니다.
""".strip()

schema_json = json.dumps(Applicants.model_json_schema(), ensure_ascii=False)
system_prompt = f"""
당신은 구조화된 정보 추출기입니다.
반드시 JSON 객체 하나만 반환하세요.
설명 문장, 코드 블록, 마크다운은 넣지 마세요.
아래 JSON Schema를 만족해야 합니다.

JSON Schema:
{schema_json}
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
parsed = Applicants.model_validate(payload)

pprint([app.model_dump() for app in parsed.applicants])
