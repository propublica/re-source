"""The keyword strategy searches complete parent rows without embedding them."""

from pathlib import Path

from re_source.vespa_strategies import BM25VespaStrategy


class _ParentClient:
    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "root": {
                "children": [
                    {
                        "relevance": 0.4,
                        "fields": {
                            "parent_id": "tips__r1",
                            "source_file": "tips",
                            "full_row": {
                                "Email address": "reporter@example.com",
                                "Response": "Call me after five.",
                            },
                            "matchfeatures": {"bm25(keyword_text)": 3.5},
                        },
                    }
                ]
            }
        }

    def get(self, *_args, **_kwargs):
        raise AssertionError("keyword search must not fetch chunk documents")


def test_keyword_search_queries_parent_rows_with_web_syntax():
    client = _ParentClient()
    BM25VespaStrategy(client).execute("reporter@example.com", top_k=25)

    assert "from parent" in client.kwargs["yql"]
    # "web" requires every term like "all", and adds `a OR b`.
    assert 'grammar: "web"' in client.kwargs["yql"]
    assert 'defaultIndex: "keyword_text"' in client.kwargs["yql"]
    assert client.kwargs["ranking"] == "keyword"
    assert client.kwargs["hits"] == 25
    assert "q_dense" not in client.kwargs
    assert "q_colbert" not in client.kwargs


def test_keyword_search_sends_operator_queries_untouched():
    for query in ('"the home depot"', "heb OR lowes", "walmart -the", "+running", "(kroger ralphs)"):
        client = _ParentClient()
        BM25VespaStrategy(client).execute(query)

        assert client.kwargs["query"] == query


def test_keyword_search_still_strips_stopwords_from_plain_queries():
    client = _ParentClient()
    BM25VespaStrategy(client).execute("the home depot or lowes h-e-b")

    assert client.kwargs["query"] == "home depot lowes h-e-b"


def test_keyword_search_returns_the_matching_complete_row_field():
    result = BM25VespaStrategy(_ParentClient()).execute("reporter@example.com").results[0]

    assert result.doc_id == "tips__r1"
    assert result.fields["Email address"] == "reporter@example.com"
    assert result.text == "reporter@example.com"
    assert result.score == 3.5
    assert result.payload["matching_chunks"][0]["column_name"] == "Email address"
    assert result.payload["highlights"][0] == "reporter@example.com"


def test_parent_schema_has_lexical_search_but_no_embedding_pipeline():
    schema = (
        Path(__file__).resolve().parents[1] / "vespa" / "schemas" / "parent.sd"
    ).read_text()

    assert "field keyword_text type string" in schema
    assert "index: enable-bm25" in schema
    assert "rank-profile keyword" in schema
    assert "| embed " not in schema


def test_operator_highlights_keep_phrases_whole_and_skip_excluded_terms():
    from re_source.vespa_strategies import _keyword_phrases

    assert _keyword_phrases('"home depot" lowes') == ["home depot", "lowes"]
    assert _keyword_phrases('walmart -"sam\'s club" -costco') == ["walmart"]
    assert _keyword_phrases('heb OR "h-e-b"') == ["heb", "h-e-b"]
    assert _keyword_phrases("+running (kroger ralphs) -(foo bar)") == ["running", "kroger", "ralphs"]
    assert _keyword_phrases('"home depot') == ["home depot"]


def test_plain_highlights_are_unchanged():
    from re_source.vespa_strategies import _keyword_phrases

    assert _keyword_phrases("reporter@example.com") == ["reporter@example.com", "reporter", "example", "com"]
    assert _keyword_phrases("home depot") == ["home depot", "home", "depot"]


def test_an_excluded_term_does_not_mark_a_field_as_matched():
    from re_source.vespa_strategies import _matching_keyword_chunks

    fields = {"Employer": "Walmart", "Other job": "Sam's Club", "Note": "costco"}
    chunks = _matching_keyword_chunks("p1", fields, 'walmart -"sam\'s club" -costco', 1.0)

    assert [c["column_name"] for c in chunks] == ["Employer"]
