from typing import Any


def first_output(output: Any) -> Any:
    """Return the first item if a module returns a tuple/list, else return as-is."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
