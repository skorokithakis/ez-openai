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

    message = conversation.ask("Say hello, please.")
    assert "hello" in message.text.lower()
    assert "hello" in message.raw.content[0].text.value.lower()
    assert "hello" in str(message).lower()

    message = conversation.ask("What animal is this?", image_file="tests/dog.jpg")
    assert "dog" in message.text.lower()
    assert "dog" in message.raw.content[0].text.value.lower()
    assert "dog" in str(message).lower()

    message = conversation.ask("What animal is this?", image_url=image_url_cat)
    assert "cat" in message.text.lower()
    assert "cat" in message.raw.content[0].text.value.lower()
    assert "cat" in str(message).lower()


def test_ask_addtional_instructions(assistant):
    assistant2 = Assistant.get(assistant.id)
    assert assistant.id == assistant2.id

    conversation = assistant.conversation.create()
    image_url_cat = "https://images.pexels.com/photos/19640755/pexels-photo-19640755/free-photo-of-whtie-kitten-on-autumn-leaves.jpeg"

    message = conversation.ask(
        "Say hello, please.", additional_instructions="Reply in CAPS."
    )
    assert "HELLO" in message.text
    assert "HELLO" in message.raw.content[0].text.value
    assert "HELLO" in str(message)

    message = conversation.ask(
        "What animal is this?",
        image_file="tests/dog.jpg",
        additional_instructions="Reply in CAPS.",
    )
    assert "DOG" in message.text
    assert "DOG" in message.raw.content[0].text.value
    assert "DOG" in str(message)

    message = conversation.ask(
        "What animal is this?",
        image_url=image_url_cat,
        additional_instructions="Reply in CAPS.",
    )
    assert "CAT" in message.text
    assert "CAT" in message.raw.content[0].text.value
    assert "CAT" in str(message)


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

    assert "Hello" in events[0].text
    assert " World" in events[1].text
    assert "!" in events[2].text
    assert "Hello World!" in message.text

    assert "Hello" in str(events[0])
    assert " World" in str(events[1])
    assert "!" in str(events[2])
    assert "Hello World!" in str(message)


def test_ask_stream_additional_instructions(assistant):
    conversation = assistant.conversation.create()

    events: List[Message] = []
    message: Message = None

    stream = conversation.ask_stream(
        'Repeat after me (skip the quotes): "Hello World!"',
        additional_instructions="Reply in CAPS.",
    )
    for event in stream:
        events.append(event)
    message = stream.value

    assert "HEL" in events[0].raw.content[0].text.value
    assert "LO" in events[1].raw.content[0].text.value
    assert " WORLD" in events[2].raw.content[0].text.value
    assert "!" in events[3].raw.content[0].text.value
    assert "HELLO WORLD!" in message.raw.content[0].text.value

    assert "HEL" in events[0].text
    assert "LO" in events[1].text
    assert " WORLD" in events[2].text
    assert "!" in events[3].text
    assert "HELLO WORLD!" in message.text

    assert "HEL" in str(events[0])
    assert "LO" in str(events[1])
    assert " WORLD" in str(events[2])
    assert "!" in str(events[3])
    assert "HELLO WORLD!" in str(message)


def test_ask_function_call(assistant):
    conversation = assistant.conversation.create()
    message = conversation.ask("Hey, I'm Stavros. Goodbye.")
    assert "Bye bye Stavros!!!11" in message.text
    assert "Bye bye Stavros!!!11" in message.raw.content[0].text.value
    assert "Bye bye Stavros!!!11" in str(message)


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

    assert "Bye" in events[0].text
    assert " bye" in events[1].text
    assert " Stav" in events[2].text
    assert "ros" in events[3].text
    assert "!!!" in events[4].text
    assert "11" in events[5].text
    assert "Bye bye Stavros!!!11" in message.text

    assert "Bye" in str(events[0])
    assert " bye" in str(events[1])
    assert " Stav" in str(events[2])
    assert "ros" in str(events[3])
    assert "!!!" in str(events[4])
    assert "11" in str(events[5])
    assert "Bye bye Stavros!!!11" in str(message)
