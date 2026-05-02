import base64
import os
import pathlib
import tempfile
import webbrowser
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


def show_image(img: Image.Image) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp, "JPEG")
        tmp.flush()
        file_url = pathlib.Path(tmp.name).resolve().as_uri()
        webbrowser.open_new_tab(file_url)


def get_base64_data(file_path: str) -> str:
    img = Image.open(file_path)
    width, height = img.size
    print(f"이미지 크기: {width}x{height}")
    show_image(img)

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
    image_file = "data/database/348/bon001958-000-0001.jpg"
    client, model = build_client()
    image_base64 = get_base64_data(image_file)

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
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
                        "text": "이 유물을 한국어로 간단히 묘사해 주세요.",
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
