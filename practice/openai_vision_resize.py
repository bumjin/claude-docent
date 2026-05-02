import base64
import json
import os
import pathlib
import sys
import tempfile
import time
import webbrowser
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


MAX_PIXELS = 1_150_000


def show_image(img: Image.Image) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp, "JPEG")
        tmp.flush()
        file_url = pathlib.Path(tmp.name).resolve().as_uri()
        webbrowser.open_new_tab(file_url)


def resize(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = img.size
    while w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size
    return img


def encode_image(path: str, do_resize: bool) -> str:
    with Image.open(path) as img:
        w, h = img.size
        print(f"원본: {w}x{h}px, 추정 토큰 수: {(w * h) // 750}")
        show_image(img)
        if do_resize:
            img = resize(img)
            w, h = img.size
            print(f"리사이즈 후: {w}x{h}px, 추정 토큰 수: {(w * h) // 750}")
            show_image(img)
        else:
            print("리사이즈 건너뜀")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_client() -> tuple[OpenAI, str]:
    load_dotenv()
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    return client, model


def main() -> None:
    do_resize = len(sys.argv) > 1 and sys.argv[1] == "1"
    image_file = "data/resized_relic_348.jpg"
    encoded_image = encode_image(image_file, do_resize)

    client, model = build_client()
    t0 = time.perf_counter()

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
                        "image_data": encoded_image,
                    },
                    {"type": "text", "text": "이미지를 한국어로 설명해 주세요."},
                ],
            }
        ],
    )

    print("\n=== 모델 응답 ===\n" + (response.choices[0].message.content or ""))
    if response.usage:
        print("\n=== 토큰 사용량 ===\n" + json.dumps(response.usage.model_dump(), ensure_ascii=False, indent=2))
    print(f"\n[LLM 전체 수행 시간] {time.perf_counter() - t0:.3f}초")


if __name__ == "__main__":
    main()
