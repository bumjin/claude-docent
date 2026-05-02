# AGENTS

## 기본 규칙
- 응답과 산출물 문서는 항상 한국어로 작성한다.

## OpenAI 호환 서버
- 이 프로젝트에서 로컬 OpenAI 호환 서버를 다룰 때는 `skills/openai-compatible-server/SKILL.md`를 먼저 참고한다.
- `Responses API`의 `tool calling`, `structured output`, `responses.parse()`가 비어 있거나 부분 동작하면 공식 OpenAI 패턴을 고집하지 말고, 프로젝트의 로컬 서버 대응 패턴으로 구현한다.
