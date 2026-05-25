"""Centralized prompt templates for consistent formatting across all modules."""

from src.data.function_pool import format_functions_for_prompt


def build_fc_system_prompt(functions, include_refusal=False):
    """Build the standard function calling system prompt.

    Args:
        functions: list of function definition dicts
        include_refusal: if True, instruct model to refuse unmatched queries
    """
    functions_text = format_functions_for_prompt(functions)

    parts = [
        "You are a helpful assistant with access to the following functions. "
        "When the user asks a question, respond with a JSON function call.",
        "",
        f"Available functions:\n{functions_text}",
        "",
    ]

    if include_refusal:
        parts.append(
            'If the query cannot be answered with the available functions, '
            'respond with: {"name": "none", "arguments": {"reason": "..."}}'
        )
        parts.append("")

    parts.append(
        'Respond with ONLY a JSON object: {"name": "function_name", "arguments": {...}}'
    )

    return "\n".join(parts)


def build_chat_messages(system_prompt, query):
    """Build chat messages in the standard format for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
