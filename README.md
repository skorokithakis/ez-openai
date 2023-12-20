# Ez OpenAI

My opinion of the `openai` Python library is best illustrated by the fact that if you
ask ChatGPT about it, it will usually hallucinate a more reasonable API. So, I wrote
this library, because if I had to manually poll for a tool update again I would
instigate the robot uprising myself.


## Installation

Run this somewhere:

```
pip install ez-openai
```


## Usage

### Basic usage

Using Ez OpenAI is (hopefully) straightforward, otherwise I've failed at the one thing
I've set out to make:

```python
from ez_openai import Assistant

# To use a previously-created assistant:
ass = Assistant.get("asst_someassistantid")

# To create a new one:
ass = Assistant.create(
    name="Weatherperson",
    instructions="You are a helpful weatherperson.",
)

# You can store the ID for later.
assistant_id = ass.id

# Delete it when you're done.
ass.delete()
```

### Function calling

No more wizardry, just plain Python functions:

```python
from ez_openai import Assistant, openai_function
from enum import Enum

@openai_function(descriptions={
        "city": "The city to get the weather for.",
        "unit": "The temperature unit , either `c` or `f`.",
    })
def get_weather(city: str, unit: Enum("unit", ["c", "f"])):
    """Get the weather for a given city, and in the given unit."""
    # ...do some magic here to get the weather...
    print(f"I'm getting the weather for {city} woooooo")
    return {"temperature": 26, "humidity": "60%"}


ass = Assistant.create(
    name="Weatherperson",
    instructions="You are a helpful weatherperson.",
    functions=[get_weather]
)

# Or, if you already have one, you can fetch it (but still
# need to specify the functions).
ass = Assistant.get("asst_O5ZAsccgOOtgjrcgHhUMloSA", functions=[get_weather])

conversation = ass.conversation.create()

# Similarly, you can store the ID to fetch later:
old_conversation = ass.conversation.get(old_conversation.id)

# The library will handle all the background function calls itself:
conversation.ask("Hi, what's the weather like in Thessaloniki and Athens right now?")
> I'm getting the weather for Thessaloniki woooooo
> I'm getting the weather for Athens woooooo
> "The weather today in both Thessaloniki and Athens is quite similar, with a
   temperature of 26°C and a humidity level at 60%. Enjoy a pleasant and comfortable
   day!"
```

Because assistants change (eg if you want to add some more functions), and it's tedious
to create new ones every time, there's a helper method that will update an assistant
with new functions/instructions:

```python
from ez_openai import Assistant

ass = Assistant.get_and_modify(
    id="asst_someassistantid",
    name="Weatherperson",
    instructions="These are your new instructions.",
    functions=[get_weather, some_new_function]
)
```

gg ez
