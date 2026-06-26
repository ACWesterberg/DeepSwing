from __future__ import annotations

import pytest

from src.data.insider_fetcher import _decode_fi_response, _detect_delimiter, _resolve_col


class TestDetectDelimiter:
    def test_semicolon_csv(self):
        text = "Col1;Col2;Col3\nA;B;C"
        assert _detect_delimiter(text) == ";"

    def test_comma_csv(self):
        text = "Col1,Col2,Col3\nA,B,C"
        assert _detect_delimiter(text) == ","

    def test_mixed_prefers_semicolons(self):
        # More semicolons → semicolon delimiter
        text = "A;B;C;D,E"
        assert _detect_delimiter(text) == ";"

    def test_empty_string_defaults_to_semicolon(self):
        assert _detect_delimiter("") == ";"


class TestDecodeFiResponse:
    def test_utf8_bom(self):
        text = "hello"
        encoded = b"\xef\xbb\xbf" + text.encode("utf-8")
        result = _decode_fi_response(encoded)
        assert result == text

    def test_latin1_encoding(self):
        text = "Ericsson AB \xe5\xe4\xf6"
        encoded = text.encode("latin-1")
        result = _decode_fi_response(encoded)
        assert "Ericsson" in result

    def test_plain_utf8(self):
        text = "normal ascii"
        result = _decode_fi_response(text.encode("utf-8"))
        assert result == text


class TestResolveCol:
    def test_first_alias_found(self):
        row = {"Emittent": "Ericsson"}
        assert _resolve_col(row, "issuer") == "Ericsson"

    def test_fallback_alias(self):
        row = {"Issuer": "Volvo"}
        assert _resolve_col(row, "issuer") == "Volvo"

    def test_swedish_alias(self):
        row = {"Handelsdatum": "2026-01-15"}
        assert _resolve_col(row, "date") == "2026-01-15"

    def test_english_alias(self):
        row = {"TransactionDate": "2026-01-15"}
        assert _resolve_col(row, "date") == "2026-01-15"

    def test_strips_whitespace(self):
        row = {"Emittent": "  Ericsson  "}
        assert _resolve_col(row, "issuer") == "Ericsson"

    def test_unknown_canonical_returns_question_mark(self):
        row = {"anything": "value"}
        assert _resolve_col(row, "nonexistent_key") == "?"

    def test_empty_value_falls_through_to_next_alias(self):
        row = {"Emittent": "", "Issuer": "Volvo"}
        assert _resolve_col(row, "issuer") == "Volvo"

    def test_no_matching_alias_returns_question_mark(self):
        row = {"UnrelatedColumn": "data"}
        assert _resolve_col(row, "issuer") == "?"
