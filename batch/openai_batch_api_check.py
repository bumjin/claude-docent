import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

class Category(BaseModel):
    id: str = Field(description="유물 id")
    image_description: str = Field(
        description=(
            "입력된 텍스트 정보에 의존하지 말고, 이미지에 실제로 보이는 내용만 한국어로 "
            "자연스럽게 3문장 이상 묘사하세요."
        )
    )
    nationality: str = Field(description="유물의 국가. 예: 한국, 중국, 일본")
    period: str = Field(
        description="유물의 시대. 예: 신라, 고려, 조선. 통일신라는 신라로 표기하세요."
    )
    genre: str = Field(
        description="유물 장르. 예: 불상, 공예, 회화, 서예, 복식, 과학기술"
    )


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


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_base64_data

relic_index_path = ROOT_DIR / "data" / "database" / "relic_index.json"
with relic_index_path.open("r", encoding="utf-8") as f:
    relic_index_json = json.load(f)


client = build_client()
model = get_model()

schema_json = json.dumps(Category.model_json_schema(), ensure_ascii=False)
system_prompt = f"""
당신은 박물관 유물 분류기입니다.
반드시 JSON 객체 하나만 반환하세요.
설명 문장, 코드 블록, 마크다운은 금지합니다.
이미지 설명은 이미지에 실제로 보이는 것만 적고, 입력 텍스트의 서술을 베끼지 마세요.
아래 JSON Schema를 만족해야 합니다.

JSON Schema:
{schema_json}
""".strip()

category_data = []
for relic_id, relic_info in relic_index_json.items():
    image_path = (
        ROOT_DIR
        / "data"
        / "database"
        / relic_id
        / Path(relic_info["img"]).name
    )
    base64_data = get_base64_data(image_path)
    user_text = (
        f"id:{relic_id}\n"
        f"{relic_info['label']}\n"
        f"{relic_info['content']}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image_data": base64_data},
                ],
            },
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content or ""
    payload = extract_first_json_object(content)
    parsed = Category.model_validate(payload)
    category_data.append(parsed.model_dump())

    if len(category_data) >= 2:
        break


for item in category_data:
    print(json.dumps(item, ensure_ascii=False, indent=2))
