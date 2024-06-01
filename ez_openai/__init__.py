import json
import os
import time
from typing import Callable, Any, Generator
from openai.types.beta.threads import Message
from openai.types.beta.threads import MessageDelta
from openai.types.beta.assistant_stream_event import ThreadRunRequiresAction
from openai.lib.streaming import AssistantStreamManager
from openai.lib.streaming import AssistantEventHandler


import openai

from .decorator import openai_function  # noqa

DEFAULT_MODEL = "gpt-4o"


class Conversation:
    """
    A conversation, with multiple messages.

    This is roughly what OpenAI calls a thread.
    """

    def __init__(self, assistant: "Assistant", functions: dict[str, Callable]) -> None:
        self._assistant = assistant
        self._functions = functions
        self.__thread = None

    @property
    def _client(self):
        return self._assistant._client

    @property
    def _thread(self):
        if self.__thread is None:
            raise ValueError(
                "Cannot work with an uninitialized conversation. Either specify a "
                "conversation ID or call .create()."
            )
        return self.__thread

    @_thread.setter
    def _thread(self, thread):
        self.__thread = thread

    @property
    def id(self) -> str:
        return self._thread.id

    def get(self, id) -> "Conversation":
        self._thread = self._assistant._client.beta.threads.retrieve(id)
        return self

    def create(self) -> "Conversation":
        self._thread = self._assistant._client.beta.threads.create()
        return self

    def delete(self) -> None:
        self._assistant._client.beta.threads.delete(self.id)

    def _gather_content(self, message, image_url, image_file):
        content = []
        file = None
        if message is not None:
            content.append({"type": "text", "text": message})
        if image_url is not None:
            content.append({"type": "image_url", "image_url": {"url": image_url}})  # type: ignore
        if image_file is not None:
            file = self._client.files.create(
                file=open(image_file, "rb"), purpose="assistants"
            )
            content.append(
                {
                    "type": "image_file",
                    "image_file": {"file_id": file.id},  # type: ignore
                }
            )
        return content

    def _gather_tool_outputs(self, last_run: ThreadRunRequiresAction):
        tool_outputs = []
        for fn_call in last_run.required_action.submit_tool_outputs.tool_calls:
            # Run the functions, one by one, and collect the results.
            function = fn_call.function
            r = self._functions[function.name](**json.loads(function.arguments))
            tool_outputs.append({"tool_call_id": fn_call.id, "output": json.dumps(r)})
        return tool_outputs

    def ask(
        self,
        message: str | None,
        image_url: str | None = None,
        image_file: bytes | None = None,
    ) -> str:
        content = self._gather_content(message, image_url, image_file)

        self._client.beta.threads.messages.create(self.id, role="user", content=content)

        last_run = self._client.beta.threads.runs.create(
            thread_id=self.id, assistant_id=self._assistant.id
        )

        while True:
            while last_run.status in ("queued", "in_progress"):
                last_run = self._client.beta.threads.runs.retrieve(
                    thread_id=self._thread.id, run_id=last_run.id
                )
                time.sleep(1)

            if last_run.status == "requires_action":  # type: ignore[attr-defined]
                tool_outputs = self._gather_tool_outputs(last_run)

                last_run = self._client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.id,
                    run_id=last_run.id,
                    tool_outputs=tool_outputs,
                )

            elif last_run.status == "completed":  # type: ignore[attr-defined]
                thread_messages = self._client.beta.threads.messages.list(
                    self._thread.id, limit=4
                )
                response = thread_messages.data[0].content[0].text.value
                return response
            elif last_run.status == "failed":
                raise ValueError(
                    f"ERROR: Got unknown run status: {last_run.last_error.message}"
                )

    def ask_stream(
        self,
        message: str | None,
        image_url: str | None = None,
        image_file: bytes | None = None,
    ) -> Generator[MessageDelta, None, Message]:
        content = self._gather_content(message, image_url, image_file)

        self._client.beta.threads.messages.create(
            self.id,
            role="user",
            content=content,
        )

        stream_manager: AssistantStreamManager[AssistantEventHandler] = (
            self._client.beta.threads.runs.stream(
                thread_id=self.id,
                assistant_id=self._assistant.id,
            )
        )

        tool_outputs = []
        while True:
            with stream_manager as stream:
                while True:
                    event = next(stream)
                    match event.event:
                        case "thread.message.delta":
                            yield event.data
                        case "thread.message.completed":
                            return event.data
                        case "thread.run.requires_action":
                            tool_outputs = self._gather_tool_outputs(event.data)
                            # Break the inner loop to reset the stream
                            break
                        case _:
                            continue

            # Create a new stream with the updated tool outputs
            if tool_outputs:
                stream_manager = (
                    self._client.beta.threads.runs.submit_tool_outputs_stream(
                        thread_id=self.id,
                        run_id=event.data.id,
                        tool_outputs=tool_outputs,
                    )
                )
                tool_outputs = []


class Assistant:
    def __init__(
        self,
        api_key: str = "",
        functions: None | list[Callable] = None,
    ) -> None:
        """Initialize the assistant."""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise AssertionError(
                "ERROR: api_key parameter or OPENAI_API_KEY environment variable not "
                "provided, cannot continue without an API key."
            )

        self._client = openai.OpenAI(api_key=api_key)
        self.__assistant = None

        if not functions:
            functions = []
        self._functions = {fn.__name__: fn for fn in functions}

    @property
    def id(self) -> str:
        return self._assistant.id

    @property
    def conversation(self) -> Conversation:
        return Conversation(assistant=self, functions=self._functions)

    @property
    def _assistant(self):
        if self.__assistant is None:
            raise ValueError(
                "Cannot work with an uninitialized assistant. Either specify an "
                "assistant ID or call .create()."
            )
        return self.__assistant

    @_assistant.setter
    def _assistant(self, assistant):
        self.__assistant = assistant

    @classmethod
    def get(
        cls,
        id: str,
        functions: None | list[Callable] = None,
        api_key: str = "",
    ) -> "Assistant":
        """Retrieve a previously-created assistant by ID."""
        assistant = cls(api_key=api_key, functions=functions)
        assistant._assistant = assistant._client.beta.assistants.retrieve(id)
        return assistant

    @classmethod
    def get_and_modify(
        cls,
        id: str,
        name: str,
        instructions: str = "",
        model=DEFAULT_MODEL,
        temperature: float | None = None,
        response_format: Any = None,
        functions: None | list[Callable] = None,
        api_key: str = "",
    ) -> "Assistant":
        """Retrieve a previously-created assistant, and modify it to the parameters."""
        assistant = cls.get(id, functions=functions, api_key=api_key)
        params = {
            "instructions": instructions,
            "name": name,
            "tools": [fn._openai_fn for fn in assistant._functions.values()],  # type: ignore
            "model": model,
        }
        if response_format:
            params["response_format"] = response_format
        if temperature:
            params["temperature"] = temperature
        assistant._client.beta.assistants.update(id, **params)
        return assistant

    @classmethod
    def create(
        cls,
        name: str,
        instructions: str = "",
        model=DEFAULT_MODEL,
        temperature: float = 1.0,
        response_format: Any = None,
        functions: None | list[Callable] = None,
        api_key: str = "",
    ) -> "Assistant":
        """Create an assistant."""
        assistant = cls(api_key=api_key, functions=functions)
        params = {
            "instructions": instructions,
            "name": name,
            "model": model,
            "temperature": temperature,
            "tools": [fn._openai_fn for fn in assistant._functions.values()],  # type: ignore
        }
        if response_format:
            params["response_format"] = response_format
        assistant._assistant = assistant._client.beta.assistants.create(**params)
        return assistant

    def delete(self):
        self._client.beta.assistants.delete(self.id)
