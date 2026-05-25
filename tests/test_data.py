"""Unit tests for data formatting and function pool."""

import json
import pytest
from src.data.function_pool import (
    FUNCTION_POOL, DOMAIN_MAP, get_functions_by_names, format_functions_for_prompt,
)
from src.data.formatter import (
    pairs_to_dpo_format, save_pairs_jsonl, load_pairs_jsonl, dataset_statistics,
)


class TestFunctionPool:
    def test_pool_has_25_functions(self):
        assert len(FUNCTION_POOL) == 25

    def test_all_functions_have_required_fields(self):
        for func in FUNCTION_POOL:
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "properties" in func["parameters"]
            assert "required" in func["parameters"]

    def test_domain_map_covers_all_functions(self):
        all_names_in_map = set()
        for names in DOMAIN_MAP.values():
            all_names_in_map.update(names)
        pool_names = {f["name"] for f in FUNCTION_POOL}
        assert all_names_in_map == pool_names

    def test_get_functions_by_names(self):
        funcs = get_functions_by_names(["get_weather", "search_web"])
        assert len(funcs) == 2
        assert funcs[0]["name"] == "get_weather"

    def test_get_functions_ignores_unknown(self):
        funcs = get_functions_by_names(["get_weather", "nonexistent"])
        assert len(funcs) == 1

    def test_format_functions_for_prompt(self):
        funcs = get_functions_by_names(["get_weather"])
        text = format_functions_for_prompt(funcs)
        assert "get_weather" in text
        assert "city" in text
        assert "[REQUIRED]" in text


class TestFormatter:
    def _make_pairs(self, n=5):
        return [
            {
                "system": "You are a helpful assistant.",
                "query": f"Query {i}",
                "chosen": json.dumps({"name": "get_weather", "arguments": {"city": "Tokyo"}}),
                "rejected": json.dumps({"name": "search_web", "arguments": {"query": "weather"}}),
                "error_type": "wrong_function",
                "expected_function": "get_weather",
            }
            for i in range(n)
        ]

    def test_pairs_to_dpo_format(self):
        pairs = self._make_pairs(10)
        records = pairs_to_dpo_format(pairs)
        assert len(records) == 10
        assert "chosen" in records[0]
        assert "rejected" in records[0]
        assert isinstance(records[0]["chosen"], list)
        assert records[0]["chosen"][-1]["role"] == "assistant"

    def test_dataset_statistics(self):
        pairs = self._make_pairs(5)
        stats = dataset_statistics(pairs)
        assert stats["total_pairs"] == 5
        assert "wrong_function" in stats["error_type_distribution"]

    def test_save_load_jsonl(self, tmp_path):
        pairs = self._make_pairs(3)
        path = str(tmp_path / "test.jsonl")
        save_pairs_jsonl(pairs, path)
        loaded = load_pairs_jsonl(path)
        assert len(loaded) == 3
        assert loaded[0]["query"] == pairs[0]["query"]
