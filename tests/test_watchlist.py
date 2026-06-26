from __future__ import annotations

import pytest

from src.data.watchlist import _parse_nasdaq_nordic_html, _parse_wikipedia_html


class TestParseNasdaqNordicHtml:
    def test_extracts_ticker_from_td(self):
        html = "<table><tr><td>ERIC-B</td><td>Ericsson</td></tr></table>"
        result = _parse_nasdaq_nordic_html(html)
        assert "ERIC-B.STO" in result

    def test_skips_html_tag_names(self):
        html = "<table><tr><td>TD</td><td>TR</td><td>SAND</td></tr></table>"
        result = _parse_nasdaq_nordic_html(html)
        assert "TD.STO" not in result
        assert "TR.STO" not in result

    def test_deduplicates_tickers(self):
        html = "<table><tr><td>ERIC-B</td><td>ERIC-B</td><td>VOLV-B</td></tr></table>"
        result = _parse_nasdaq_nordic_html(html)
        assert result.count("ERIC-B.STO") == 1

    def test_empty_html_returns_empty(self):
        assert _parse_nasdaq_nordic_html("") == []

    def test_all_results_have_sto_suffix(self):
        html = "<table><tr><td>SAND</td><td>SEB-A</td></tr></table>"
        result = _parse_nasdaq_nordic_html(html)
        assert all(r.endswith(".STO") for r in result)

    def test_caps_at_35(self):
        tickers = " ".join(f"<td>T{i:02d}</td>" for i in range(50))
        html = f"<table><tr>{tickers}</tr></table>"
        result = _parse_nasdaq_nordic_html(html)
        assert len(result) <= 35


class TestParseWikipediaHtml:
    def _make_wiki_table(self, rows: list[str]) -> str:
        rows_html = "\n".join(f"<tr>{r}</tr>" for r in rows)
        return f'<table class="wikitable sortable">{rows_html}</table>'

    def test_extracts_linked_ticker(self):
        html = self._make_wiki_table([
            '<td><a href="/wiki/Ericsson">ERIC-B</a></td>'
        ])
        result = _parse_wikipedia_html(html)
        assert "ERIC-B.STO" in result

    def test_extracts_plain_ticker_cell(self):
        html = self._make_wiki_table(["<td>VOLV-B</td>"])
        result = _parse_wikipedia_html(html)
        assert "VOLV-B.STO" in result

    def test_rejects_non_ticker_content(self):
        html = self._make_wiki_table(["<td>This is a description with words</td>"])
        result = _parse_wikipedia_html(html)
        assert result == []

    def test_no_table_returns_empty(self):
        assert _parse_wikipedia_html("<html><body>no table</body></html>") == []

    def test_deduplicates(self):
        html = self._make_wiki_table([
            '<td><a href="#">SAND</a></td>',
            '<td><a href="#">SAND</a></td>',
        ])
        result = _parse_wikipedia_html(html)
        assert result.count("SAND.STO") == 1

    def test_all_results_have_sto_suffix(self):
        html = self._make_wiki_table([
            '<td><a href="#">HEXA-B</a></td>',
            '<td><a href="#">SEB-A</a></td>',
        ])
        result = _parse_wikipedia_html(html)
        assert all(r.endswith(".STO") for r in result)
