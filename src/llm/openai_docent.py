import logging

from ..relics import Relics, SearchedRelics
from ..utils import get_base64_data
from .openai_llm import openai_gpt_5_4 as openai_llm
from .openai_prompt_templates import guide_instruction, revisit_instruction
from .openai_tools import ToolData, use_tools

logger = logging.getLogger(__name__)


class ExceptionHandler:
    @staticmethod
    def overflow(messages: list, relics: Relics) -> Relics:
        if isinstance(relics, SearchedRelics):
            messages.append(
                {
                    "role": "assistant",
                    "content": "검색된 전시물을 모두 소개했습니다. 다음 전시물로 넘어가겠습니다.",
                }
            )
            relics.original.index += 1
            return relics.original

        messages.append(
            {
                "role": "assistant",
                "content": "준비한 전시물을 모두 소개했습니다. 오늘도 즐거운 관람이 되었길 바랍니다.",
            }
        )
        return relics

    @staticmethod
    def underflow(messages: list, relics: Relics):
        messages.append({"role": "assistant", "content": "첫 번째 작품입니다."})
        relics.index = 0


class InstructionHandler:
    def __init__(self):
        self.last_guide_id = ""

    def add_guide(self, relics: Relics, messages: list) -> None:
        self._remove_previous_guide(messages)
        guide_instruction_prompt = guide_instruction.format(
            label=relics.current["label"],
            content=relics.current["content"],
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": guide_instruction_prompt},
                    {"type": "image", "image_data": get_base64_data(relics.current["img_path"])},
                ],
            }
        )
        self.last_guide_id = relics.current_id

    def _remove_previous_guide(self, messages: list) -> None:
        for idx in reversed(range(len(messages))):
            content = messages[idx]["content"]
            if not isinstance(content, list):
                continue
            first = content[0]
            if not isinstance(first, dict):
                continue
            text_message = first.get("text", "")
            if "[전시물 정보]" in text_message:
                messages.pop(idx)
                break

    def check_and_add(self, relics: Relics, messages: list) -> None:
        if self.last_guide_id == relics.current_id:
            return
        self.add_guide(relics, messages)
        messages.append({"role": "user", "content": revisit_instruction})


class OpenAIDocentBot:
    greeting_message = """
안녕하세요. 저는 박물관 도슨트 뮤즈입니다.
전시 작품 설명부터 감상 포인트까지 함께 안내해 드릴게요.
그럼 첫 번째 전시물을 소개하겠습니다.
""".strip()

    def __init__(self):
        self.messages = []
        self.relics = Relics()
        self.instruction = InstructionHandler()

    def greet(self) -> str:
        return self.greeting_message

    def _present_relic(self) -> None:
        self.instruction.add_guide(self.relics, self.messages)
        response_message = openai_llm.create_response_text(messages=self.messages)
        self.messages.append({"role": "assistant", "content": response_message})
        self.relics.set_presented(True)

    def move(self, is_next: bool) -> None:
        if is_next:
            try:
                self.relics.next()
            except IndexError:
                self.relics = ExceptionHandler.overflow(self.messages, self.relics)
        else:
            try:
                self.relics.previous()
            except ValueError:
                ExceptionHandler.underflow(self.messages, self.relics)

        if not self.relics.is_presented():
            self._present_relic()

    def answer(self, user_input: str) -> tuple[list, str]:
        self.instruction.check_and_add(self.relics, self.messages)
        self.messages.append({"role": "user", "content": user_input})
        conversation = self.get_conversation()
        tool_data: ToolData | None = None
        message_dict: dict[str, str] | None = None
        tool_data, message_dict = use_tools(conversation, self.relics.original_database)
        references: list = []

        match tool_data:
            case {"type": "relics", "items": items}:
                if len(items) > 0:
                    self.relics = SearchedRelics(items, self.relics.original)
                self.messages.append(message_dict)
                response_message = message_dict["content"]
            case {"type": "facts", "items": references}:
                self.messages.append(message_dict)
                response_message = openai_llm.create_response_text(messages=self.messages)
                self.messages.append({"role": "assistant", "content": response_message})
            case _:
                response_message = openai_llm.create_response_text(messages=self.messages)
                self.messages.append({"role": "assistant", "content": response_message})

        return references, response_message

    def get_conversation(self) -> list[dict[str, str]]:
        conversation = []
        for message in self.messages:
            if isinstance(message["content"], list):
                text_message = ""
                for item in message["content"]:
                    if isinstance(item, dict) and "text" in item:
                        text_message = item["text"]
                        break
            else:
                text_message = message["content"]
            text_message = text_message.strip()
            if text_message.startswith("[전시물 정보]"):
                continue
            conversation.append({"role": message["role"], "content": text_message})
        return conversation
