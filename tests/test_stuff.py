from ez_openai import Assistant


def test_answer():
    # To create a new one:
    assistant = Assistant.create(
        name="Weatherperson",
        instructions="You are a helpful weatherperson.",
        temperature=2,
    )
    # To use a previously-created assistant:
    assistant2 = Assistant.get(assistant.id)
    assert assistant.id == assistant2.id

    # Delete it when you're done.
    assistant.delete()
