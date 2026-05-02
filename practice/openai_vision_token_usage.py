import base64
import json
import os
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


def get_base64_data(file_path: str) -> str:
    img = Image.open(file_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}, 추정 토큰 수: {(width * height) // 750}")

    buffer = BytesIO()
    img.save(buffer, format=img.format or "JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_client() -> tuple[OpenAI, str]:
    load_dotenv()
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    return client, model


def main() -> None:
    image_file = "data/database/348/image.jpg"
    client, model = build_client()
    image_base64 = get_base64_data(image_file)

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_data": image_base64,
                    },
                    {
                        "type": "text",
                        "text": "이 이미지를 한국어 3문장으로 요약해 주세요.",
                    },
                ],
            }
        ],
    )

    print("모델 응답 ===\n" + (response.choices[0].message.content or ""))
    if response.usage:
        print("토큰 사용량 ===\n" + json.dumps(response.usage.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
