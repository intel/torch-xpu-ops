"""JSON extraction utilities."""


def extract_json(text: str) -> str:
    """Extract the first JSON object from text."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    raise ValueError("No JSON object found in text")
