import json
from typing import Any


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("첫 JSON 값이 객체가 아닙니다.")
    return payload


def run_function_call_loop(
    client,
    model: str,
    instructions: str | None,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_repository: dict[str, Any],
):
    tool_trace: list[dict[str, Any]] = []

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_items,
        tools=tools,
        tool_choice="auto",
    )

    while True:
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not function_calls:
            return response, tool_trace

        input_items.extend(response.output)

        for function_call in function_calls:
            args = json.loads(function_call.arguments)
            result = tool_repository[function_call.name](**args)
            tool_trace.append(
                {
                    "name": function_call.name,
                    "arguments": args,
                    "result": result,
                }
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=input_items,
            tools=tools,
            tool_choice="auto",
        )


def run_local_tool_call_loop(
    client,
    model: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    tool_repository: dict[str, Any],
    max_iterations: int = 5,
):
    tool_summaries: list[str] = []
    for tool in tools:
        properties = tool["parameters"]["properties"]
        required = tool["parameters"].get("required", [])
        tool_summaries.append(
            json.dumps(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "arguments": properties,
                    "required": required,
                },
                ensure_ascii=False,
            )
        )

    system_prompt = (
        "You are a local tool router for a server that does not support native tool calling. "
        "Reply with JSON only. "
        'Use exactly one of these shapes: {"type":"tool_call","name":"...","arguments":{...}} '
        'or {"type":"final","answer":"..."}. '
        "If the user asked for multiple pieces of information, call every needed tool before returning a final answer. "
        "Do not return final until you have enough information to fully answer the request. "
        "When answering finally, write the answer in Korean. "
        "Available tools:\n"
        + "\n".join(tool_summaries)
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_trace: list[dict[str, Any]] = []

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        message = response.choices[0].message
        content = message.content or ""
        payload = _extract_first_json_object(content)

        if payload.get("type") == "final":
            return payload.get("answer", ""), tool_trace

        if payload.get("type") != "tool_call":
            raise ValueError(f"알 수 없는 응답 형식입니다: {payload}")

        tool_name = payload["name"]
        arguments = payload.get("arguments", {})
        if tool_name not in tool_repository:
            raise KeyError(f"등록되지 않은 도구입니다: {tool_name}")

        result = tool_repository[tool_name](**arguments)
        tool_trace.append(
            {
                "name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool result:\n"
                    + json.dumps(
                        {
                            "name": tool_name,
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                    + "\nIf more information is still needed, return another tool_call JSON. "
                    + 'Only when the user request is fully covered, return {"type":"final","answer":"..."} in Korean.'
                ),
            }
        )

    raise RuntimeError("도구 호출 반복 횟수를 초과했습니다.")
