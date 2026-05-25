"""Unit tests for function calling metrics and JSON parsing."""

import pytest
from src.evaluation.metrics import parse_function_call, compute_fc_metrics


class TestParseFunctionCall:
    def test_valid_json(self):
        text = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        result = parse_function_call(text)
        assert result["name"] == "get_weather"
        assert result["arguments"]["city"] == "Tokyo"

    def test_json_in_code_block(self):
        text = '```json\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n```'
        result = parse_function_call(text)
        assert result["name"] == "get_weather"

    def test_json_with_trailing_text(self):
        text = '{"name": "get_weather", "arguments": {"city": "Tokyo"}} some extra text'
        result = parse_function_call(text)
        assert result["name"] == "get_weather"

    def test_redundant_generation(self):
        text = (
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}} '
            '{"name": "search_web", "arguments": {"query": "weather"}}'
        )
        result = parse_function_call(text)
        assert result["name"] == "get_weather"

    def test_nested_arguments(self):
        text = '{"name": "send_email", "arguments": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}}'
        result = parse_function_call(text)
        assert result["name"] == "send_email"
        assert result["arguments"]["to"] == "a@b.com"

    def test_empty_string(self):
        assert parse_function_call("") is None

    def test_invalid_json(self):
        assert parse_function_call("not json at all") is None

    def test_json_without_name(self):
        text = '{"function": "get_weather", "args": {}}'
        assert parse_function_call(text) is None

    def test_leading_text_then_json(self):
        text = 'I will call: {"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        result = parse_function_call(text)
        assert result["name"] == "get_weather"


class TestComputeFCMetrics:
    def test_perfect_predictions(self):
        preds = ['{"name": "get_weather", "arguments": {"city": "Tokyo"}}']
        refs = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["exact_match"] == 1.0
        assert metrics["json_valid_rate"] == 1.0
        assert metrics["name_accuracy"] == 1.0

    def test_wrong_function_name(self):
        preds = ['{"name": "search_web", "arguments": {"city": "Tokyo"}}']
        refs = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["name_accuracy"] == 0.0
        assert metrics["exact_match"] == 0.0

    def test_invalid_json(self):
        preds = ["not valid json"]
        refs = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["json_valid_rate"] == 0.0
        assert metrics["exact_match"] == 0.0

    def test_missing_argument(self):
        preds = ['{"name": "get_weather", "arguments": {}}']
        refs = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["name_accuracy"] == 1.0
        assert metrics["arg_names_accuracy"] == 0.0
        assert metrics["exact_match"] == 0.0

    def test_case_insensitive_values(self):
        preds = ['{"name": "get_weather", "arguments": {"city": "tokyo"}}']
        refs = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["exact_match"] == 1.0

    def test_empty_inputs(self):
        metrics = compute_fc_metrics([], [])
        assert metrics["num_samples"] == 0
        assert metrics["exact_match"] == 0

    def test_multiple_samples(self):
        preds = [
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}',
            '{"name": "wrong_func", "arguments": {}}',
        ]
        refs = [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "search_web", "arguments": {"query": "test"}},
        ]
        metrics = compute_fc_metrics(preds, refs)
        assert metrics["exact_match"] == 0.5
        assert metrics["name_accuracy"] == 0.5
