from ez_openai import Assistant
import pytest


@pytest.fixture()
def assistant():
    assistant = Assistant.create(
        name="Weatherperson", instructions="You are a helpful weatherperson."
    )
    yield assistant
    assistant.delete()


def test_answer(assistant):
    assistant2 = Assistant.get(assistant.id)
    assert assistant.id == assistant2.id

    conversation = assistant.conversation.create()

    assert "hello" in conversation.ask("Say hello, please.").lower()
    assert "dog" in conversation.ask("What animal is this?", image_file="tests/dog.jpg")
    assert "dog" in conversation.ask(
        "What animal is this?",
        image_url="https://loremflickr.com/cache/resized/164_364237614_e93038a73d_c_512_512_nofilter.jpg",
    )
