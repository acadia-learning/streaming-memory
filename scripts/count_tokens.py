#!/usr/bin/env python3
"""Count tokens in Aryan's memory files (JSON or JSONL)."""

import json
from pathlib import Path

import tiktoken


def extract_text_from_message(message_data: dict) -> str:
    """Extract all text content from a message data object."""
    text_parts = []

    # Extract content from different message types
    if 'content' in message_data:
        text_parts.append(str(message_data['content']))

    if 'summary' in message_data:
        summary = message_data['summary']
        if isinstance(summary, list):
            text_parts.extend(str(item) for item in summary)
        else:
            text_parts.append(str(summary))

    if 'arguments' in message_data:
        text_parts.append(str(message_data['arguments']))

    if 'output' in message_data:
        text_parts.append(str(message_data['output']))

    return ' '.join(text_parts)

def count_tokens_in_jsonl(file_path: Path, model: str = "gpt-4") -> dict:
    """Count tokens in a JSONL file."""
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = 0
    total_messages = 0
    message_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    message_data = entry.get('message_data', {})
                    text = extract_text_from_message(message_data)

                    tokens = encoding.encode(text)
                    token_count = len(tokens)
                    total_tokens += token_count
                    total_messages += 1
                    message_lengths.append(token_count)
                except json.JSONDecodeError:
                    continue

    avg_tokens = total_tokens / total_messages if total_messages > 0 else 0
    min_tokens = min(message_lengths) if message_lengths else 0
    max_tokens = max(message_lengths) if message_lengths else 0

    return {
        'total_tokens': total_tokens,
        'total_messages': total_messages,
        'average_tokens_per_message': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'model': model
    }

def count_tokens_in_json(file_path: Path, model: str = "gpt-4") -> dict:
    """Count tokens in a JSON file."""
    encoding = tiktoken.encoding_for_model(model)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_tokens = 0
    total_summaries = len(data)
    summary_lengths = []

    for entry in data:
        summary_text = entry.get('summary', '')
        tokens = encoding.encode(summary_text)
        token_count = len(tokens)
        total_tokens += token_count
        summary_lengths.append(token_count)

    avg_tokens = total_tokens / total_summaries if total_summaries > 0 else 0
    min_tokens = min(summary_lengths) if summary_lengths else 0
    max_tokens = max(summary_lengths) if summary_lengths else 0

    return {
        'total_tokens': total_tokens,
        'total_summaries': total_summaries,
        'average_tokens_per_summary': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'model': model
    }

if __name__ == '__main__':
    import sys

    # Determine which file to process
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Default to messages file
        file_path = Path(__file__).parent.parent / 'aryan_messages.jsonl'

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Determine file type and count tokens
    if file_path.suffix == '.jsonl':
        stats = count_tokens_in_jsonl(file_path, model='gpt-4')
        print(f"\nToken Count Statistics for {file_path.name}")
        print(f"{'='*50}")
        print(f"Model: {stats['model']}")
        print(f"Total Messages: {stats['total_messages']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Average Tokens per Message: {stats['average_tokens_per_message']:.1f}")
        print(f"Min Tokens: {stats['min_tokens']}")
        print(f"Max Tokens: {stats['max_tokens']}")
        print(f"{'='*50}\n")
    else:
        stats = count_tokens_in_json(file_path, model='gpt-4')
        print(f"\nToken Count Statistics for {file_path.name}")
        print(f"{'='*50}")
        print(f"Model: {stats['model']}")
        print(f"Total Summaries: {stats['total_summaries']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Average Tokens per Summary: {stats['average_tokens_per_summary']:.1f}")
        print(f"Min Tokens: {stats['min_tokens']}")
        print(f"Max Tokens: {stats['max_tokens']}")
        print(f"{'='*50}\n")

