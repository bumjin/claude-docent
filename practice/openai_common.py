import os

from dotenv import load_dotenv
from openai import OpenAI


def build_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )


def get_model(default: str = "gpt-5.4") -> str:
    load_dotenv()
    return os.getenv("OPENAI_MODEL", default)
