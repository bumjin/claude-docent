import os

from dotenv import load_dotenv
from openai import OpenAI


base_url = "https://www.museum.go.kr/MUSEUM/contents/M0504000000.do"

image_file_sample = """
<div class="thumbs-area">
    <div class="swiper-container gallery-thumbs">
        <div class="swiper-wrapper">
            <div class="swiper-slide">
                <img src="/relic_image//PS01001001/bon001/2016/1124093013018/700/bon001958-000-0001.jpg" alt="sample image">
            </div>
        </div>
    </div>
</div>
""".strip()

label_sample = """
<div class="outview-tit-box">
    <div class="label-list">
        <span class="label-type purple-2">중요</span>
    </div>
    <div class="outview">
        <strong class="outveiw-tit">유물명</strong>
        <ul class="outview-list">
            <li><strong>시대</strong><p>통일신라</p></li>
            <li><strong>분류</strong><p>불교 조각</p></li>
            <li><strong>크기</strong><p>높이 270.0cm</p></li>
        </ul>
    </div>
</div>
""".strip()

content_sample = """
<div class="view-info-cont view-info-cont2">
    <p>유물 설명 본문...</p>
    <div class='codeView01 codeCopyright line'>
        <img src='https://www.kogl.or.kr/open/web/images/images_2014/codetype/new_img_opentype01.png' alt='공공누리'>
    </div>
</div>
""".strip()

relic_index_sample = """
{
    "348": {
        "url": "https://www.museum.go.kr/MUSEUM/contents/M0504000000.do?schM=view&searchId=treasure&relicId=348",
        "img": "/relic_image//PS01001001/bon001/2016/1124093013018/700/bon001958-000-0001.jpg",
        "label": {
            "명칭": "유물명",
            "시대": "통일신라",
            "분류": "불교 조각",
            "크기": "높이 270.0cm"
        },
        "content": "유물 설명 본문...",
        "copyright_img": "https://www.kogl.or.kr/open/web/images/images_2014/codetype/new_img_opentype01.png"
    }
}
""".strip()

prompt = f"""
유물 상세 페이지들을 순회해서 JSON 데이터베이스와 이미지 폴더를 만드는 Python 코드를 작성해 주세요.

[base_url]
{base_url}

[html_image_file_sample]
{image_file_sample}

[html_label_sample]
{label_sample}

[html_content_sample]
{content_sample}

[relic_index_sample]
{relic_index_sample}

요구사항:
1. `data/relic_links.json` 파일을 읽습니다.
2. 각 항목은 `{{"page": 1, "query": "?schM=view&relicId=348"}}` 형태입니다.
3. `base_url + query`로 상세 URL을 만들고 HTML을 요청합니다.
4. BeautifulSoup로 다음 정보를 추출합니다.
   - key: `relicId`
   - url
   - 첫 번째 대표 이미지 경로
   - 라벨 정보 dict
   - 본문 설명 text
   - 저작권 이미지 URL
5. 결과 전체는 `relic_index_json`에 담습니다.
6. 서버 부하를 피하기 위해 요청마다 1초 대기합니다.
7. `data/database` 폴더를 새로 만들고, 그 안에 `relic_index.json`을 저장합니다.
8. 각 `relicId`별 하위 폴더를 만들고 `relic_data.json`과 대표 이미지 파일을 저장합니다.
9. 저작권 이미지가 `https://www.kogl.or.kr/open/web/images/images_2014/codetype/new_img_opentype01.png` 가 아닌 유물은 제외하는 필터 함수도 포함합니다.
10. 코드만 반환하고, ```python ``` 코드펜스는 포함하지 마세요.
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
