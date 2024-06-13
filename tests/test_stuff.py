from typing import List

import pytest
from openai.types.beta.threads import Message

from ez_openai import Assistant
from ez_openai import openai_function


@openai_function(
    descriptions={
        "name": "Name of the user",
    }
)
def say_goodbye(name: str):
    """Says goodbye to the user."""
    return f"Bye bye {name}!!!11"


@pytest.fixture()
def assistant():
    assistant = Assistant.create(
        name="Weatherperson",
        instructions="You are a helpful weatherperson. If you want to tell the user goodbye, use the function say_goodbye and return the exact response.",
        functions=[say_goodbye],
        model="gpt-4o",
    )
    yield assistant
    assistant.delete()


def test_ask(assistant):
    assistant2 = Assistant.get(assistant.id)
    assert assistant.id == assistant2.id

    conversation = assistant.conversation.create()

    image_url_cat = "https://images.pexels.com/photos/19640755/pexels-photo-19640755/free-photo-of-whtie-kitten-on-autumn-leaves.jpeg"

    assert "hello" in conversation.ask("Say hello, please.").lower()
    assert "dog" in conversation.ask("What animal is this?", image_file="tests/dog.jpg")
    assert "cat" in conversation.ask("What animal is this?", image_url=image_url_cat)


def test_ask_stream(assistant):
    conversation = assistant.conversation.create()

    events: List[Message] = []
    message: Message = None

    stream = conversation.ask_stream(
        'Repeat after me (skip the quotes): "Hello World!"'
    )
    for event in stream:
        events.append(event)
    message = stream.value

    assert "Hello" in events[0].raw.content[0].text.value
    assert " World" in events[1].raw.content[0].text.value
    assert "!" in events[2].raw.content[0].text.value
    assert "Hello World!" in message.raw.content[0].text.value

    assert "Hello" in str(events[0])
    assert " World" in str(events[1])
    assert "!" in str(events[2])
    assert "Hello World!" in str(message)


def test_ask_function_call(assistant):
    conversation = assistant.conversation.create()
    assert "Bye bye Stavros!!!11" in conversation.ask("Hey, I'm Stavros. Goodbye.")


def test_ask_stream_function_call(assistant):
    conversation = assistant.conversation.create()

    events: List[Message] = []
    message: Message = None

    stream = conversation.ask_stream("Hey, I'm Stavros. Goodbye.")
    for event in stream:
        events.append(event)
    message = stream.value

    assert "Bye" in events[0].raw.content[0].text.value
    assert " bye" in events[1].raw.content[0].text.value
    assert " Stav" in events[2].raw.content[0].text.value
    assert "ros" in events[3].raw.content[0].text.value
    assert "!!!" in events[4].raw.content[0].text.value
    assert "11" in events[5].raw.content[0].text.value
    assert "Bye bye Stavros!!!11" in message.raw.content[0].text.value

    assert "Bye" in str(events[0])
    assert " bye" in str(events[1])
    assert " Stav" in str(events[2])
    assert "ros" in str(events[3])
    assert "!!!" in str(events[4])
    assert "11" in str(events[5])
    assert "Bye bye Stavros!!!11" in str(message)
