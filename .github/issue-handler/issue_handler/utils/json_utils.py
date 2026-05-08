"""JSON extraction utilities."""
import json


def extract_json(text: str) -> str:
    """Extract the first JSON object from text.

    Uses json.JSONDecoder.raw_decode which handles braces inside
    string values correctly, unlike naive brace counting.
    """
    decoder = json.JSONDecoder()
    # Find the first '{' and try raw_decode from there
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON object found in text")
