import base64
import os
import pathlib
import tempfile
import webbrowser
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


image_file = "data/leaflet/전시해설프로그램.png"
json_file = "data/leaflet/guide_program.json"

prompt = """
전달한 전시 안내 이미지를 읽고 JSON 데이터로 구조화해 주세요.

조건:
1. 이미지의 핵심 섹션을 JSON 형태로 정리합니다.
2. 최상위 키는 "기획전시관 안내", "큐레이터 투어", "기획전시관 예약 안내" 세 가지로 구성합니다.
3. 응답은 JSON 본문만 반환합니다.
""".strip()


def show_image(img: Image.Image) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp, "PNG")
        tmp.flush()
        file_url = pathlib.Path(tmp.name).resolve().as_uri()
        webbrowser.open_new_tab(file_url)


def get_base64_data(file_path: str) -> str:
    img = Image.open(file_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}")
    show_image(img)

    buffer = BytesIO()
    img.save(buffer, format=img.format or "PNG")
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
    client, model = build_client()
    image_base64 = get_base64_data(image_file)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        stop=["</json>"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image_data": image_base64,
                    },
                ],
            }
        ],
    )

    json_str = response.choices[0].message.content.strip().replace("<json>", "")
    print(json_str)
    with open(json_file, "w", encoding="utf-8") as file:
        file.write(json_str)


if __name__ == "__main__":
    main()
