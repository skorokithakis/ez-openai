import pytest
from ez_openai import Assistant, openai_function
from openai.types.beta.threads import Message
from typing import List


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
    try:
        while True:
            event = next(stream)
            events.append(event)
    except StopIteration as e:
        message = e.value

    assert "Hello" in events[0].delta.content[0].text.value
    assert " World" in events[1].delta.content[0].text.value
    assert "!" in events[2].delta.content[0].text.value
    assert "Hello World!" in message.content[0].text.value


def test_ask_function_call(assistant):
    conversation = assistant.conversation.create()
    assert "Bye bye Stavros!!!11" in conversation.ask("Hey, I'm Stavros. Goodbye.")


def test_ask_stream_function_call(assistant):
    conversation = assistant.conversation.create()

    events: List[Message] = []
    message: Message = None

    stream = conversation.ask_stream("Hey, I'm Stavros. Goodbye.")
    try:
        while True:
            event = next(stream)
            events.append(event)
    except StopIteration as e:
        message = e.value

    assert "Bye" in events[0].delta.content[0].text.value
    assert " bye" in events[1].delta.content[0].text.value
    assert " Stav" in events[2].delta.content[0].text.value
    assert "ros" in events[3].delta.content[0].text.value
    assert "!!!" in events[4].delta.content[0].text.value
    assert "11" in events[5].delta.content[0].text.value
    assert "Bye bye Stavros!!!11" in message.content[0].text.value
