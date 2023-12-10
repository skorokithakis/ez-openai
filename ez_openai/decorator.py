import inspect
from functools import wraps

TYPE_MAP = {"str": "string"}


def openai_function(descriptions=dict[str, str]):
    def outer(f):
        arguments = {}
        required_arguments = []
        sig = inspect.signature(f)

        for param in sig.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError(f"Parameter is untyped: {f.__name__}({param.name})")

            if param.name not in descriptions:
                raise ValueError(
                    f"Parameter has no description: {f.__name__}({param.name})"
                )

            arguments[param.name] = {
                "type": TYPE_MAP.get(
                    param.annotation.__name__, param.annotation.__name__
                ),
                "description": descriptions[param.name],
            }
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
