import os
import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client for DeepSeek API with token tracking and retry."""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url or "https://api.deepseek.com/v1",
        )
        self.model = model or "deepseek-chat"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def chat(
        self,
        messages,
        temperature=0.7,
        max_tokens=1024,
        json_mode=False,
        max_retries=3,
    ):
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self.total_input_tokens += resp.usage.prompt_tokens
                self.total_output_tokens += resp.usage.completion_tokens
                self.total_requests += 1
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("API call failed (attempt %d/%d): %s. Retrying in %ds",
                               attempt + 1, max_retries, e, wait)
                time.sleep(wait)

    def chat_batch(self, messages_list, delay=0.1, **kwargs):
        results = []
        for i, messages in enumerate(messages_list):
            result = self.chat(messages, **kwargs)
            results.append(result)
            if delay and i < len(messages_list) - 1:
                time.sleep(delay)
        return results

    @property
    def usage_summary(self):
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def parse_json_response(self, text):
        """Extract JSON from LLM response with multi-layer fallback."""
        text = text.strip()

        # Layer 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Layer 2: extract from markdown code block
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Layer 3: find first { ... } or [ ... ]
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = text.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_ch:
                    depth += 1
                elif text[i] == close_ch:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

        raise ValueError(f"Cannot parse JSON from response: {text[:200]}")
