import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# https://github.com/EvanZhouDev/openai-oauth/
# codex provides a simple way to obtain an access token for the OpenAI API. You can use it to authenticate your requests without having to manually generate and manage API keys. 
# npx @openai/codex login
# npx openai-oauth

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {
            "role": "user",
            "content": "Hello, OpenAI?",
        }
    ],
)

print(response.choices[0].message.content)
