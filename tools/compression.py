"""
Context compression tool for managing conversation history.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from .project import get_active_project_folder


SUMMARY_MAX_TOKENS = 16384
FALLBACK_CHARS = 20000
TOOL_RESULT_LIMIT = 1000


def _build_conversation_text(msgs: List[Dict[str, Any]]) -> str:
    """Render messages (dicts) as text for the summarizer."""
    out = ""
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning:
                out += f"\n[Assistant Reasoning]: {reasoning[:500]}...\n"
            infos = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                raw = fn.get("arguments", "")
                if name == "write_file":
                    try:
                        a = json.loads(raw)
                        infos.append(f"write_file(file={a.get('filename')}, mode={a.get('mode')}, chars={len(a.get('content', ''))})")
                        continue
                    except (json.JSONDecodeError, AttributeError):
                        pass
                infos.append(f"{name}({raw[:500]})")
            if infos:
                out += f"\n[Assistant Tool Calls]: {', '.join(infos)}\n"
            if content:
                out += f"\n[Assistant]: {content}\n"
        elif role == "tool":
            name = msg.get("name", "unknown_tool")
            text = str(content)
            if len(text) > TOOL_RESULT_LIMIT and not text.startswith("Error"):
                text = text[:700] + " [...] " + text[-300:]
            out += f"\n[Tool Result - {name}]: {text}\n"
        elif role == "user":
            out += f"\n[User]: {content}\n"
    return out


def compress_context_impl(
    messages: List[Any],
    client,
    model: str,
    keep_recent: int = 10,
    save_only: bool = False
) -> Dict[str, Any]:
    """
    Compresses the conversation context by summarizing older messages.
    
    This function:
    1. Takes all messages except the most recent ones
    2. Calls the kimi API to create a comprehensive summary
    3. Saves the summary to a timestamped file
    4. Returns the compressed messages list and stats
    
    Args:
        messages: The full message history
        client: The OpenAI client instance
        model: The model to use for summarization
        keep_recent: Number of recent messages to keep uncompressed
        save_only: Write the summary file but keep all messages unchanged
        
    Returns:
        Dictionary containing:
        - compressed_messages: New message list with compression applied
        - summary_file: Path to saved summary file
        - tokens_before: Estimated tokens before compression
        - tokens_after: Estimated tokens after compression
    """
    system_message = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if system_message else 0

    if save_only:
        cut = len(messages)
    else:
        cut = len(messages) - keep_recent
        # Never start the recent block with a 'tool' message: snap back to the
        # assistant message that issued the tool calls.
        while cut > start and messages[cut].get("role") == "tool":
            cut -= 1
        if cut <= start:
            return {
                "compressed_messages": messages,
                "summary_file": None,
                "tokens_saved": 0,
                "message": "Not enough messages to compress."
            }

    messages_to_compress = messages[start:cut]
    recent_messages = messages[cut:]
    if not messages_to_compress:
        return {
            "compressed_messages": messages,
            "summary_file": None,
            "tokens_saved": 0,
            "message": "Not enough messages to compress."
        }

    # Create a detailed prompt for summarization
    summary_prompt = """Please provide a comprehensive summary of the conversation history below. Include:
1. The main task or goal discussed
2. Key decisions made
3. Files created and their purposes
4. Progress made so far
5. Any important context for continuing the work

Conversation history to summarize:
"""
    
    conversation_text = _build_conversation_text(messages_to_compress)

    summary = None
    try:
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries of conversations."},
                {"role": "user", "content": summary_prompt + conversation_text}
            ],
            temperature=0.7,
            max_tokens=SUMMARY_MAX_TOKENS
        )
        summary = summary_response.choices[0].message.content
    except Exception as e:
        if not save_only:
            return {
                "compressed_messages": messages,
                "summary_file": None,
                "tokens_saved": 0,
                "message": f"Error during compression: {str(e)}"
            }

    if not summary or not summary.strip():
        # Reasoning can eat max_tokens and leave content empty: fall back to
        # the raw digest so progress is never lost.
        summary = "(Automatic summary unavailable; raw history digest follows.)\n\n" + conversation_text[-FALLBACK_CHARS:]

    # Save summary to file
    project_folder = get_active_project_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if project_folder:
        summary_file = os.path.join(project_folder, f".context_summary_{timestamp}.md")
    else:
        # If no project folder, save in current directory
        summary_file = f".context_summary_{timestamp}.md"
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Context Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Messages Compressed:** {len(messages_to_compress)}\n\n")
            f.write(f"**Messages Retained:** {len(recent_messages)}\n\n")
            f.write(f"---\n\n")
            f.write(summary)
    except Exception as e:
        summary_file = f"Error saving summary: {str(e)}"
    
    if save_only:
        return {
            "compressed_messages": messages,
            "summary_file": summary_file,
            "tokens_saved": 0,
            "message": f"Summary saved to {os.path.basename(summary_file)}."
        }

    # Build the compressed message list
    compressed_messages = []
    
    # Add system message if it exists
    if system_message:
        compressed_messages.append(system_message)
    
    # Add the summary as a user message
    compressed_messages.append({
        "role": "user",
        "content": f"[CONTEXT SUMMARY - Previous conversation compressed]\n\n{summary}\n\n[END CONTEXT SUMMARY - Continuing from here...]"
    })
    
    # Add recent messages
    compressed_messages.extend(recent_messages)
    
    # Calculate token savings (rough estimate)
    original_length = sum(len(str(m)) for m in messages_to_compress)
    compressed_length = len(summary)
    estimated_tokens_saved = (original_length - compressed_length) // 4  # Rough estimate
    
    return {
        "compressed_messages": compressed_messages,
        "summary_file": summary_file,
        "tokens_saved": estimated_tokens_saved,
        "messages_compressed": len(messages_to_compress),
        "messages_retained": len(recent_messages),
        "message": f"Successfully compressed {len(messages_to_compress)} messages. Summary saved to {os.path.basename(summary_file)}."
    }

