---
name: openai-compatible-server
description: Use when working against a local or partial OpenAI-compatible server where Responses API features such as tool calling, structured output, or parse helpers return empty output or missing parsed results
---

# OpenAI Compatible Server

## Overview
로컬 OpenAI 호환 서버는 `models.list()`와 기본 `chat.completions`는 동작해도, `responses.create()`, `responses.parse()`, `text.format=json_schema`, 네이티브 tool calling을 부분만 구현한 경우가 많다.

이 프로젝트에서는 서버가 `completed`를 반환하면서도 `output=[]`, `output_text=""`, `output_parsed=None`을 돌려줄 수 있으므로, 공식 OpenAI 예제를 그대로 쓰지 말고 로컬 서버 대응 패턴으로 바꾼다.

## When to Use
- `responses.create()`는 성공인데 `response.output_text`가 빈 문자열일 때
- `responses.parse()` 호출 후 `output_parsed`가 `None`일 때
- tool 정의를 넘겨도 `tool_calls` 없이 일반 대화만 반환할 때
- structured output을 기대했는데 JSON 본문이 비어 있을 때
- 로컬 서버가 `OPENAI_BASE_URL=http://127.0.0.1:10532/v1` 같은 호환 엔드포인트일 때

다음 경우에는 이 스킬을 쓰지 않는다.
- 실제 OpenAI 공식 엔드포인트를 사용하고 있고 `Responses API` 기능이 정상 동작할 때
- 단순 텍스트 질의만 필요해 JSON 추출이나 도구 호출이 전혀 없을 때

## Core Pattern
공식 패턴:
```python
response = client.responses.parse(...)
parsed = response.output_parsed
```

로컬 서버 대응 패턴:
```python
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)
payload = _extract_first_json_object(response.choices[0].message.content or "")
```

핵심 원칙:
- 네이티브 tool calling 대신 `JSON only` 프롬프트를 사용한다.
- structured output 대신 스키마를 프롬프트에 넣고 JSON을 직접 파싱한다.
- Pydantic 검증이 필요하면 `model_validate()`를 후처리로 적용한다.

## Quick Reference
- 도구 호출: `run_local_tool_call_loop()` 사용
- JSON 추출: `_extract_first_json_object()` 사용
- structured output: `chat.completions` + JSON Schema 문자열 프롬프트
- Pydantic 파싱: `MyModel.model_validate(payload)`
- 실행 검증: `uv run python practice/<file>.py`

참고 구현:
- `practice/openai_tool_helpers.py`
- `practice/openai_tool_use_basic.py`
- `practice/openai_multi_tool_use.py`
- `practice/openai_multi_tool_error.py`
- `practice/openai_tool_use_beta.py`
- `practice/openai_structured_output.py`
- `practice/openai_applicants.py`

## Implementation
### 1. 도구 호출
모델이 반드시 아래 둘 중 하나만 반환하도록 강제한다.

```json
{"type":"tool_call","name":"get_weather","arguments":{"location":"Seoul","unit":"celsius"}}
```

```json
{"type":"final","answer":"최종 답변"}
```

파이썬 쪽에서는:
- 첫 JSON 객체를 파싱한다.
- `type == "tool_call"`이면 로컬 함수를 실행한다.
- 결과를 다시 모델에 넣어 다음 `tool_call` 또는 `final`을 받는다.

### 2. structured output
`responses.parse()`를 쓰지 말고:
- `model_json_schema()` 또는 수동 스키마를 JSON 문자열로 만든다.
- 시스템 프롬프트에 "JSON 객체 하나만 반환"을 명시한다.
- 응답을 직접 파싱한다.
- 필요하면 Pydantic으로 검증한다.

예시:
```python
schema_json = json.dumps(Applicants.model_json_schema(), ensure_ascii=False)
system_prompt = f"""
반드시 JSON 객체 하나만 반환하세요.
아래 JSON Schema를 만족해야 합니다.
{schema_json}
""".strip()
```

## Common Mistakes
- `responses.create()`가 성공했으니 structured output도 될 거라고 가정하는 실수
- `response.output_text`가 비어 있는데 바로 `json.loads()` 하는 실수
- `responses.parse()`의 `output_parsed`를 `None` 체크 없이 사용하는 실수
- tool 스키마를 넘기면 로컬 서버도 자동으로 `tool_calls`를 준다고 믿는 실수
- 예전 실패 예제를 그대로 두고 문자열 인코딩이 깨진 상태를 방치하는 실수

## Verification
- 먼저 `uv run python <file>`로 실제 실패를 재현한다.
- 수정 후 같은 명령으로 다시 실행한다.
- 도구 호출 예제는 도구 호출 내역과 최종 답변이 둘 다 출력되는지 본다.
- structured output 예제는 파싱 결과가 리스트/딕셔너리로 정상 출력되는지 본다.
