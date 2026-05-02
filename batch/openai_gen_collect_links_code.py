import os

from dotenv import load_dotenv
from openai import OpenAI


base_url = "https://www.museum.go.kr/MUSEUM/contents/M0504000000.do"
query_string = "?startCount=<number>&searchId=treasure&schM=list"

tag_samples = """
<li class="card">
    <a href="?schM=view&relicId=348" class="img-box">
        <img src="/relic_image//PS01001001/bon001/2016/1124093013018/700/bon001958-000-0001.jpg" alt="sample" onerror="this.src='/ux/content/museum/images/onerror.png';">
    </a>
    <div class="txt">
        <a href="?schM=view&relicId=348">sample relic title</a>
    </div>
</li>

<li class="card">
    <a href="?schM=view&relicId=349" class="img-box">
        <img src="/relic_image//PS01001001/bon001/2017/0508160735009/700/bon001959-00-01.jpg" alt="sample" onerror="this.src='/ux/content/museum/images/onerror.png';">
    </a>
    <div class="txt">
        <a href="?schM=view&relicId=349">sample relic title</a>
    </div>
</li>
""".strip()

prompt = f"""
BeautifulSoup를 사용해 국립중앙박물관 페이지에서 유물 상세 링크를 수집하는 Python 코드를 작성해 주세요.

[base_url]
{base_url}

[query_string]
{query_string}

[tag_samples]
{tag_samples}

조건:
1. 전체 37페이지를 순회합니다.
2. 각 페이지에는 12개의 카드가 있으며 `startCount=0, 12, 24 ...` 방식으로 증가합니다.
3. 각 카드에서 `?schM=view&relicId=348` 형태의 상세 query string을 추출합니다.
4. 결과 형식은 `[{{"page": 1, "query": "?schM=view&relicId=348"}}, ...]` 입니다.
5. 서버 부하를 피하기 위해 페이지마다 1초 대기합니다.
6. 결과는 `data/relic_links.json` 파일로 저장합니다.
7. 코드만 반환하고, ```python ``` 코드펜스는 포함하지 마세요.
""".strip()


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
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
