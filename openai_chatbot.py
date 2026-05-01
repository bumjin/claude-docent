import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

st.set_page_config(page_title="OpenAI 챗봇", page_icon="💬", layout="wide")

sidebar_text = """
### 💬 OpenAI 챗봇입니다.

### 사용 방법
- 무엇이든 편하게 질문해 주세요.
- 이전 대화 내용을 이어서 답변합니다.

### 예시 질문
- 오늘 점심 메뉴 추천해줘.
- Python으로 CSV 읽는 방법 알려줘.
- 여행 일정 짜줘.
- 간단한 영어 문장 교정해줘.
"""

with st.sidebar:
    st.markdown(sidebar_text)


@st.cache_resource
def get_client():
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("메시지를 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            messages=[
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ],
        )
        response_text = response.choices[0].message.content or ""
    except Exception as error:
        st.error(f"오류가 발생했습니다: {error}")
        response_text = "응답을 생성하는 중 오류가 발생했습니다."

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )

    with st.chat_message("assistant"):
        st.markdown(response_text)
