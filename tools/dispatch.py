"""
Tool-call execution helpers. Guarantees every tool_call id gets a reply,
so the message history never contains dangling tool_calls (API 400).
"""

import json
from typing import Any, Callable, Dict, List

TRUNCATED_MSG = (
    "Error: your response was cut off (output token limit) so the tool arguments "
    "are incomplete and nothing was executed. Write shorter files: put at most "
    "about 3000 words in one call, then add more with mode='append' in later calls."
)


def _tool_message(call: Dict[str, Any], content: str) -> Dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": call["function"]["name"],
        "content": str(content),
    }


def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    finish_reason: str,
    messages: List[Dict[str, Any]],
    tool_map: Dict[str, Callable],
    log: Callable[[str], None] = print,
) -> Dict[str, bool]:
    """
    Run each tool call and append one tool message per call to `messages`.
    Never raises. Returns {"compress_requested": bool}; the caller compresses
    at the start of the next iteration, never mid-loop.
    """
    flags = {"compress_requested": False}
    for call in tool_calls:
        name = call["function"]["name"]
        try:
            if finish_reason == "length":
                result = TRUNCATED_MSG
            else:
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    args = None
                    result = f"Error: invalid JSON arguments for '{name}': {e}. Send valid JSON."
                else:
                    if not isinstance(args, dict):
                        result = f"Error: arguments for '{name}' must be a JSON object."
                    elif name == "compress_context":
                        flags["compress_requested"] = True
                        result = "Context will be compressed before the next step."
                    elif name not in tool_map:
                        result = f"Error: Unknown tool '{name}'"
                    else:
                        log(f"\n  → {name}")
                        result = tool_map[name](**args)
        except Exception as e:  # tool bug or bad argument names
            result = f"Error running tool '{name}': {e}"
        text = str(result)
        log(f"    {'✗' if text.startswith('Error') else '✓'} {text[:200]}{'...' if len(text) > 200 else ''}")
        messages.append(_tool_message(call, text))
    return flags


def fill_dangling_tool_calls(messages: List[Dict[str, Any]], reason: str = "interrupted") -> int:
    """Append an error reply for each unanswered tool_call of the last assistant message."""
    i = len(messages) - 1
    while i >= 0 and messages[i].get("role") == "tool":
        i -= 1
    if i < 0 or messages[i].get("role") != "assistant" or not messages[i].get("tool_calls"):
        return 0
    answered = {m.get("tool_call_id") for m in messages[i + 1:]}
    added = 0
    for tc in messages[i]["tool_calls"]:
        if tc["id"] not in answered:
            messages.append(_tool_message(tc, f"Error: tool call not executed ({reason}). Try again."))
            added += 1
    return added
