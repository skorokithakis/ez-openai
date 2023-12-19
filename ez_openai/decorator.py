import enum
import inspect
from functools import wraps
from typing import Any


def _openai_type_for_python_type(param) -> dict[str, Any]:
    type_map = {
        "bool": "boolean",
        "int": "integer",
        "str": "string",
    }

    type_dict: dict[str, Any] = {}
    if isinstance(param.annotation, enum.EnumMeta):
        type_dict["type"] = "string"
        type_dict["enum"] = [x.name for x in param.annotation]  # type: ignore
    else:
        type_dict["type"] = type_map.get(
            param.annotation.__name__, param.annotation.__name__
        )
    return type_dict


def openai_function(descriptions=dict[str, str]):
    def outer(f):
        arguments = {}
        required_arguments = []
        sig = inspect.signature(f)

        for param in sig.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError(f"Parameter is untyped: {f.__name__}({param.name})")

            if param.name not in descriptions:  # type: ignore
                raise ValueError(
                    f"Parameter has no description: {f.__name__}({param.name})"
                )

            arguments[param.name] = _openai_type_for_python_type(param)
            if param.name in descriptions:
                arguments[param.name]["description"] = descriptions[param.name]
            if param.default == inspect.Parameter.empty:
                required_arguments.append(param.name)

        fn_dict = {
            "type": "function",
            "function": {
                "name": f.__name__,
                "description": (f.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "required": required_arguments,
                    "properties": arguments,
                },
            },
        }

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._openai_fn = fn_dict
        return wrapper

    return outer
