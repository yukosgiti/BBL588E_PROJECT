from typing import List
CHARS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def plate_cleaner(input_str: str) -> str:
    output = input_str.upper()

    # Fix when the boundries act as a letter.
    if output.startswith("1 "):
        output = output[2:]
    output = "".join(filter(lambda i: i in CHARS, output))
    return output