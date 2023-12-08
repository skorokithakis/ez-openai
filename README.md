# Easy OpenAI

My opinion of the `openai` Python library is best illustrated by the fact that, if you
ask ChatGPT about it, it will usually hallucinate a more reasonable API. So, I wrote
this library, because if I had to manually poll for a tool update again I would
instigate the robot uprising myself.

## Installation

Run this somewhere:

```
pip install easy-openai
```


## Usage

### Basic usage

Using Easy OpenAI is (hopefully) straightforward, otherwise I've failed at the one thing
I've set out to make:

```python
from easy_openai import Assistant

# To use a previously-created assistant:
ass = Assistant("asst_someassistantid)

# To create a new one:
ass = Assistant.create(system="Some system instructions.")

# You can store the ID for later.
assistant_id = ass.id

# Delete it when you're done.
ass.delete()
```

### Function calling

No more wizardry, just plain Python functions:

```python
from easy_openai import Assistant, openai_function

@openai_function(descriptions={
        "city": "The city to get the weather for.",
        "unit": "The temperature unit , either `c` or `f`.",
    })
def get_weather(city: str, unit: str):
    # ...do some magic here to get the weather...
    print("I'm getting the weather wooooooooo")
    return {"temperature": 26, "unit": "c"}


ass = Assistant.create(
    system="You are a helpful weatherperson.",
    functions=[get_weather]
)

# The library will handle all the background function calls itself:
ass.ask("Hi, what's the weather like in Thessaloniki right now?")
> I'm getting the weather wooooooooo
> "It's 26 degrees centigrade right now in Thessaloniki."
```
