#!/usr/bin/env python3
"""
Kimi Writing Agent - An autonomous agent for creative writing tasks.

This agent uses the kimi-k2-thinking model to create novels, books, 
and short story collections based on user prompts.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

from utils import (
    estimate_token_count, 
    get_tool_definitions, 
    get_tool_map,
    get_system_prompt
)
from tools.compression import compress_context_impl
from tools.dispatch import execute_tool_calls, fill_dangling_tool_calls
from tools.project import set_active_project_folder


# Constants
MAX_ITERATIONS = 300
TOKEN_LIMIT = 200000
COMPRESSION_THRESHOLD = 180000  # Trigger compression at 90% of limit
MODEL_NAME = "kimi-k2-thinking"
BACKUP_INTERVAL = 50  # Save backup summary every N iterations


def load_context_from_file(file_path: str) -> str:
    """
    Loads context from a summary file for recovery.
    
    Args:
        file_path: Path to the context summary file
        
    Returns:
        Content of the file as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Loaded context from: {file_path}\n")
        return content
    except Exception as e:
        print(f"✗ Error loading context file: {e}")
        sys.exit(1)


def get_user_input() -> tuple[str, bool, str | None]:
    """
    Gets user input from command line, either as a prompt or recovery file.
    
    Returns:
        Tuple of (prompt/context, is_recovery_mode)
    """
    parser = argparse.ArgumentParser(
        description="Kimi Writing Agent - Create novels, books, and short stories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh start with inline prompt
  python kimi-writer.py "Create a collection of sci-fi short stories"
  
  # Recovery mode from previous context
  python kimi-writer.py --recover my_project/.context_summary_20250107_143022.md
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Your writing request (e.g., "Create a mystery novel")'
    )
    parser.add_argument(
        '--recover',
        type=str,
        help='Path to a context summary file to continue from'
    )
    
    args = parser.parse_args()
    
    # Check if recovery mode
    if args.recover:
        context = load_context_from_file(args.recover)
        return context, True, args.recover
    
    # Check if prompt provided as argument
    if args.prompt:
        return args.prompt, False, None
    
    # Interactive prompt
    print("=" * 60)
    print("Kimi Writing Agent")
    print("=" * 60)
    print("\nEnter your writing request (or 'quit' to exit):")
    print("Example: Create a collection of 15 sci-fi short stories\n")
    
    prompt = input("> ").strip()
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        sys.exit(0)
    
    if not prompt:
        print("Error: Empty prompt. Please provide a writing request.")
        sys.exit(1)
    
    return prompt, False, None


def convert_message_for_api(msg: Any) -> Dict[str, Any]:
    """
    Converts a message object to a dictionary suitable for API calls.
    Preserves reasoning_content if present.
    
    Args:
        msg: Message object (can be OpenAI message object or dict)
        
    Returns:
        Dictionary representation of the message
    """
    if isinstance(msg, dict):
        return msg
    
    # Convert OpenAI message object to dict
    msg_dict = {
        "role": msg.role,
    }
    
    if msg.content:
        msg_dict["content"] = msg.content
    
    # Preserve reasoning_content if present
    if hasattr(msg, "reasoning_content"):
        reasoning = getattr(msg, "reasoning_content")
        if reasoning:
            msg_dict["reasoning_content"] = reasoning
    
    # Preserve tool calls if present
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in msg.tool_calls
        ]
    
    # Preserve tool call id for tool response messages
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        msg_dict["tool_call_id"] = msg.tool_call_id
    
    if hasattr(msg, "name") and msg.name:
        msg_dict["name"] = msg.name
    
    return msg_dict


def main():
    """Main agent loop."""
    
    # Get API key
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("Error: MOONSHOT_API_KEY environment variable not set.")
        print("Please set your API key: export MOONSHOT_API_KEY='your-key-here'")
        sys.exit(1)
    
    base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
    
    # Debug: Show that key is loaded (masked for security)
    if len(api_key) > 8:
        print(f"✓ API Key loaded: {api_key[:4]}...{api_key[-4:]}")
    else:
        print(f"⚠️  Warning: API key seems too short ({len(api_key)} chars)")
    print(f"✓ Base URL: {base_url}\n")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Get user input
    user_prompt, is_recovery, recover_path = get_user_input()
    
    # Initialize message history
    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]
    
    if is_recovery:
        # Summary files live in the project folder: restore it as active
        project_dir = os.path.dirname(os.path.abspath(recover_path))
        if os.path.basename(recover_path).startswith(".context_summary_") and os.path.isdir(project_dir):
            set_active_project_folder(project_dir)
            print(f"✓ Active project folder: {project_dir}")
        messages.append({
            "role": "user",
            "content": f"[RECOVERED CONTEXT]\n\n{user_prompt}\n\n[END RECOVERED CONTEXT]\n\nPlease continue the work from where we left off."
        })
        print("🔄 Recovery mode: Continuing from previous context\n")
    else:
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        print(f"\n📝 Task: {user_prompt}\n")
    
    # Get tool definitions and mapping
    tools = get_tool_definitions()
    tool_map = get_tool_map()
    
    print("=" * 60)
    print("Starting Kimi Writing Agent")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Context limit: {TOKEN_LIMIT:,} tokens")
    print(f"Auto-compression at: {COMPRESSION_THRESHOLD:,} tokens")
    print("=" * 60 + "\n")
    
    force_compress = False
    task_completed = False

    # Main agent loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─' * 60}")
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"{'─' * 60}")
        
        # Check token count before making API call
        try:
            token_count = estimate_token_count(base_url, api_key, MODEL_NAME, messages)
            print(f"📊 Current tokens: {token_count:,}/{TOKEN_LIMIT:,} ({token_count/TOKEN_LIMIT*100:.1f}%)")
            
            # Trigger compression if approaching limit
            if token_count >= COMPRESSION_THRESHOLD or force_compress:
                force_compress = False
                print(f"\n⚠️  Approaching token limit! Compressing context...")
                compression_result = compress_context_impl(
                    messages=messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=10
                )
                
                if "compressed_messages" in compression_result:
                    messages = compression_result["compressed_messages"]
                    print(f"✓ {compression_result['message']}")
                    print(f"✓ Estimated tokens saved: ~{compression_result.get('tokens_saved', 0):,}")
                    
                    # Recalculate token count
                    token_count = estimate_token_count(base_url, api_key, MODEL_NAME, messages)
                    print(f"📊 New token count: {token_count:,}/{TOKEN_LIMIT:,}\n")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not estimate token count: {e}")
            token_count = 0
        
        # Auto-backup every N iterations
        if iteration % BACKUP_INTERVAL == 0:
            print(f"💾 Auto-backup (iteration {iteration})...")
            try:
                compression_result = compress_context_impl(
                    messages=messages,
                    client=client,
                    model=MODEL_NAME,
                    save_only=True
                )
                if compression_result.get("summary_file"):
                    print(f"✓ Backup saved: {os.path.basename(compression_result['summary_file'])}\n")
            except Exception as e:
                print(f"⚠️  Warning: Backup failed: {e}\n")
        
        # Call the model
        try:
            print("🤖 Calling kimi-k2-thinking model...\n")
            
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=65536,  # 64K tokens
                tools=tools,
                temperature=1.0,
                stream=True,  # Enable streaming
            )
            
            # Accumulate the streaming response
            reasoning_content = ""
            content_text = ""
            tool_calls_data = []
            role = None
            finish_reason = None
            
            # Track if we've printed headers
            reasoning_header_printed = False
            content_header_printed = False
            tool_call_header_printed = False
            last_tool_index = -1
            
            # Process the stream
            for chunk in stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                
                # Get role if present (first chunk)
                if hasattr(delta, "role") and delta.role:
                    role = delta.role
                
                # Handle reasoning_content streaming
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not reasoning_header_printed:
                        print("=" * 60)
                        print(f"🧠 Reasoning (Iteration {iteration})")
                        print("=" * 60)
                        reasoning_header_printed = True
                    
                    print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content
                
                # Handle regular content streaming
                if hasattr(delta, "content") and delta.content:
                    # Close reasoning section if it was open
                    if reasoning_header_printed and not content_header_printed:
                        print("\n" + "=" * 60 + "\n")
                    
                    if not content_header_printed:
                        print("💬 Response:")
                        print("-" * 60)
                        content_header_printed = True
                    
                    print(delta.content, end="", flush=True)
                    content_text += delta.content
                
                # Handle tool_calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # Ensure we have enough slots in tool_calls_data
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append({
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "chars_received": 0
                            })
                        
                        tc = tool_calls_data[tc_delta.index]
                        
                        # Print header when we start receiving a tool call
                        if tc_delta.index != last_tool_index:
                            if reasoning_header_printed or content_header_printed:
                                print("\n" + "=" * 60 + "\n")
                            
                            if hasattr(tc_delta, "function") and tc_delta.function.name:
                                print(f"🔧 Preparing tool call: {tc_delta.function.name}")
                                print("─" * 60)
                                tool_call_header_printed = True
                                last_tool_index = tc_delta.index
                        
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if hasattr(tc_delta, "function"):
                            if tc_delta.function.name:
                                tc["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += tc_delta.function.arguments
                                tc["chars_received"] += len(tc_delta.function.arguments)
                                
                                # Show progress indicator every 500 characters
                                if tc["chars_received"] % 500 == 0 or tc["chars_received"] < 100:
                                    # Calculate approximate words (rough estimate: 5 chars per word)
                                    words = tc["chars_received"] // 5
                                    print(f"\r💬 Generating arguments... {tc['chars_received']:,} characters (~{words:,} words)", end="", flush=True)
            
            # Print closing for content if it was printed
            if content_header_printed:
                print("\n" + "-" * 60 + "\n")
            
            # Print completion for tool calls if any were received
            if tool_call_header_printed:
                print("\n✓ Tool call complete")
                print("─" * 60 + "\n")
            
            # Build the assistant message (dict) from accumulated data
            tool_calls = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                for tc in tool_calls_data if tc["id"]
            ]
            assistant_msg = {"role": role or "assistant"}
            if content_text:
                assistant_msg["content"] = content_text
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                if finish_reason == "length":
                    print("⚠️  Response truncated by token limit. Asking the model to continue in smaller parts.")
                    messages.append({"role": "user", "content": "Your last response was cut off by the output limit. Continue, in smaller steps."})
                    continue
                print("=" * 60)
                print("✅ TASK COMPLETED")
                print("=" * 60)
                print(f"Completed in {iteration} iteration(s)")
                print("=" * 60)
                task_completed = True
                break

            print(f"\n🔧 Model decided to call {len(tool_calls)} tool(s):")
            flags = execute_tool_calls(tool_calls, finish_reason, messages, tool_map)
            if flags["compress_requested"]:
                force_compress = True

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving context...")
            # Save current context before exiting
            try:
                compression_result = compress_context_impl(
                    messages=messages,
                    client=client,
                    model=MODEL_NAME,
                    save_only=True
                )
                if compression_result.get("summary_file"):
                    print(f"✓ Context saved to: {compression_result['summary_file']}")
                    print(f"\nTo resume, run:")
                    print(f"  python kimi-writer.py --recover {compression_result['summary_file']}")
            except:
                pass
            sys.exit(0)
        
        except Exception as e:
            print(f"\n✗ Error during iteration {iteration}: {e}")
            print(f"Attempting to continue...\n")
            fill_dangling_tool_calls(messages, f"error: {e}")
            continue
    
    # If we hit max iterations without finishing
    if not task_completed:
        print("\n" + "=" * 60)
        print("⚠️  MAX ITERATIONS REACHED")
        print("=" * 60)
        print(f"\nReached maximum of {MAX_ITERATIONS} iterations.")
        print("Saving final context...")
        
        try:
            compression_result = compress_context_impl(
                messages=messages,
                client=client,
                model=MODEL_NAME,
                save_only=True
            )
            if compression_result.get("summary_file"):
                print(f"✓ Context saved to: {compression_result['summary_file']}")
                print(f"\nTo resume, run:")
                print(f"  python kimi-writer.py --recover {compression_result['summary_file']}")
        except Exception as e:
            print(f"✗ Error saving context: {e}")


if __name__ == "__main__":
    main()

