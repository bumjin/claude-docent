import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from relics import Relics
from utils import get_base64_data

from .prompt_templates import (
    guide_instruction,
    revisit_instruction,
    system_prompt,
)

logger = logging.getLogger(__name__)


def build_client() -> tuple[OpenAI, str]:
    load_dotenv()
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:10532/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    return client, model


class DocentBot:
    def __init__(self, model_name: str | None = None):
        client, default_model = build_client()
        self.client = client
        self.model = model_name or default_model
        self.messages = []
        self.relics = Relics()
        self.last_guide_id = ""

    def _create_response(self) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.messages,
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def _add_guide_instruction(self) -> None:
        guide_instruction_prompt = guide_instruction.format(
            label=self.relics.current["label"],
            content=self.relics.current["content"],
        )
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_data": get_base64_data(self.relics.current["img_path"]),
                    },
                    {"type": "text", "text": guide_instruction_prompt},
                ],
            }
        )
        self.last_guide_id = self.relics.current_id

    def _present_relic(self) -> None:
        self._add_guide_instruction()
        response_message = self._create_response()
        self.messages.append({"role": "assistant", "content": response_message})
        self.relics.set_presented(True)

    def _check_and_add(self) -> None:
        if self.last_guide_id == self.relics.current_id:
            return
        self._add_guide_instruction()
        self.messages.append({"role": "user", "content": revisit_instruction})

    def _overflow(self) -> None:
        self.messages.append(
            {"role": "assistant", "content": "준비한 유품은 모두 소개했습니다."}
        )

    def _underflow(self) -> None:
        self.messages.append({"role": "assistant", "content": "첫 번째 유품입니다."})
        self.relics.index = 0

    def move(self, is_next: bool) -> None:
        if is_next:
            try:
                self.relics.next()
            except IndexError:
                self._overflow()
        else:
            try:
                self.relics.previous()
            except ValueError:
                self._underflow()

        if not self.relics.is_presented():
            self._present_relic()

    def answer(self, user_input: str) -> str:
        self._check_and_add()
        self.messages.append({"role": "user", "content": user_input})
        response_message = self._create_response()
        self.messages.append({"role": "assistant", "content": response_message})
        return response_message

    def get_conversation(self):
        conversation = []
        for message in self.messages:
            if isinstance(message["content"], list):
                text_message: str = message["content"][1]["text"]
            else:
                text_message = message["content"]
            text_message = text_message.strip()
            if text_message.startswith("<system_command>"):
                continue
            conversation.append({"role": message["role"], "content": text_message})
        return conversation
