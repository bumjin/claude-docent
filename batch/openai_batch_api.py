import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class Category(BaseModel):
    id: str = Field(description="유물 id")
    image_description: str = Field(
        description=(
            "입력 텍스트에 의존하지 말고 이미지에 실제로 보이는 내용만 한국어로 "
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
    base_url = (
        os.getenv("OPENAI_BATCH_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://127.0.0.1:10532/v1"
    )
    return OpenAI(
        base_url=base_url,
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


def classify_relic(relic_id: str, relic_info: dict, system_prompt: str, client: OpenAI, model: str) -> Category:
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
    return Category.model_validate(payload)


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

jsonl_path = BASE_DIR / "openai_batch_requests.jsonl"
selected_items: list[tuple[str, dict]] = []
with jsonl_path.open("w", encoding="utf-8") as f:
    for idx, (relic_id, relic_info) in enumerate(relic_index_json.items()):
        selected_items.append((relic_id, relic_info))
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
        request_body = {
            "custom_id": relic_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image", "image_data": base64_data},
                        ],
                    },
                ],
            },
        }
        f.write(json.dumps(request_body, ensure_ascii=False) + "\n")
        if idx >= 1:
            break

print(f"배치 요청 파일 생성: {jsonl_path}")
print(f"base_url={client.base_url}")

with jsonl_path.open("rb") as f:
    try:
        uploaded_file = client.files.create(file=f, purpose="batch")
    except Exception as exc:
        print(
            "로컬 호환 서버가 OpenAI Files API(`/v1/files`)를 지원하지 않아 "
            "순차 처리 폴백으로 전환합니다."
        )
        category_results = []
        for relic_id, relic_info in selected_items:
            parsed = classify_relic(relic_id, relic_info, system_prompt, client, model)
            category_results.append(parsed)
            print(json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2))

        for parsed in category_results:
            category_data = parsed.model_dump()
            relic_id = category_data["id"]
            del category_data["id"]
            relic_index_json[relic_id]["image_description"] = category_data["image_description"]
            del category_data["image_description"]
            relic_index_json[relic_id]["category"] = category_data

        with relic_index_path.open("w", encoding="utf-8") as out_file:
            json.dump(relic_index_json, out_file, ensure_ascii=False, indent=4)

        print(f"폴백 결과 반영 완료: {relic_index_path}")
        raise SystemExit(0) from exc

print(f"업로드 파일 ID: {uploaded_file.id}")

try:
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"source": "openai_batch_api.py"},
    )
except Exception as exc:
    raise RuntimeError(
        "로컬 호환 서버가 OpenAI Batch API(`/v1/batches`)를 지원하지 않거나 "
        "배치 생성 요청을 처리하지 못했습니다."
    ) from exc

print(f"배치 작업 ID: {batch.id}")

terminal_statuses = {"completed", "failed", "expired", "cancelled"}
while True:
    batch = client.batches.retrieve(batch.id)
    print(f"배치 상태: {batch.status}")
    if batch.status in terminal_statuses:
        break
    time.sleep(10)

if batch.status != "completed":
    raise RuntimeError(
        f"배치 작업이 완료되지 않았습니다. status={batch.status}, "
        f"error_file_id={batch.error_file_id}"
    )

if not batch.output_file_id:
    raise RuntimeError("배치 결과 파일 ID가 없습니다.")

result_bytes = client.files.content(batch.output_file_id).content
result_lines = result_bytes.decode("utf-8").strip().splitlines()
print(f"결과 건수: {len(result_lines)}")

for line in result_lines[:2]:
    result = json.loads(line)
    custom_id = result["custom_id"]
    content = result["response"]["body"]["choices"][0]["message"]["content"]
    payload = extract_first_json_object(content)
    parsed = Category.model_validate(payload)
    category_data = parsed.model_dump()
    relic_id = category_data["id"]
    del category_data["id"]
    relic_index_json[relic_id]["image_description"] = category_data["image_description"]
    del category_data["image_description"]
    relic_index_json[relic_id]["category"] = category_data
    print(json.dumps({"custom_id": custom_id, **parsed.model_dump()}, ensure_ascii=False, indent=2))

with relic_index_path.open("w", encoding="utf-8") as f:
    json.dump(relic_index_json, f, ensure_ascii=False, indent=4)

print(f"결과 반영 완료: {relic_index_path}")
