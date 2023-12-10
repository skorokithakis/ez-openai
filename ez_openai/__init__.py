import json
import os
import time
from typing import Callable

import openai

from .decorator import openai_function  # noqa


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

    def ask(self, message) -> str:
        self._client.beta.threads.messages.create(self.id, role="user", content=message)

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
                tool_outputs = []
                for fn_call in last_run.required_action.submit_tool_outputs.tool_calls:
                    # Run the functions, one by one, and collect the results.
                    function = fn_call.function
                    r = self._functions[function.name](**json.loads(function.arguments))
                    tool_outputs.append(
                        {"tool_call_id": fn_call.id, "output": json.dumps(r)}
                    )

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
            else:
                raise ValueError("ERROR: Got unknown run status.")


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
    def create(
        cls,
        name: str,
        instructions: str = "",
        model="gpt-4-1106-preview",
        functions: None | list[Callable] = None,
        api_key: str = "",
    ) -> "Assistant":
        """Create an assistant."""
        assistant = cls(api_key=api_key, functions=functions)
        assistant._assistant = assistant._client.beta.assistants.create(
            instructions=instructions,
            name=name,
            model=model,
            tools=[fn._openai_fn for fn in assistant._functions.values()],  # type: ignore
        )
        return assistant

    def delete(self):
        self._client.beta.assistants.delete(self.id)
