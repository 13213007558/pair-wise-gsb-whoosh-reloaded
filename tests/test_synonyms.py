import pytest

from whoosh import analysis, fields, highlight, qparser, query
from whoosh.analysis.synonyms import (
    SynonymGraphFilter,
    SynonymPathLimitError,
    path_position_gap,
    token_paths,
)
from whoosh.filedb.filestore import RamStorage


def analyze(text, synonyms, keep_orig=True, stoplist=None, renumber=True):
    ana = analysis.RegexTokenizer() | analysis.LowercaseFilter()
    if stoplist is not None:
        ana = ana | analysis.StopFilter(stoplist=stoplist, renumber=renumber)
    ana = ana | SynonymGraphFilter(synonyms, keep_orig=keep_orig)
    return list(ana(text, positions=True))


def texts(tokens):
    return [t.text for t in tokens]


def paths(tokens, **kwargs):
    return [texts(p) for p in token_paths(tokens, **kwargs)]


def make_index(synonyms, docs, stoplist=None, renumber=True):
    ana = analysis.RegexTokenizer() | analysis.LowercaseFilter()
    if stoplist is not None:
        ana = ana | analysis.StopFilter(stoplist=stoplist, renumber=renumber)
    ana = ana | analysis.MultiFilter(query=SynonymGraphFilter(synonyms))
    schema = fields.Schema(content=fields.TEXT(stored=True, analyzer=ana))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    for doc in docs:
        w.add_document(content=doc)
    w.commit()
    return ix


def test_single_to_multi():
    tokens = analyze("visit nyc office", {"nyc": "new york city"})
    assert paths(tokens) == [
        ["visit", "new", "york", "city", "office"],
        ["visit", "nyc", "office"],
    ]
    # The kept original spans the whole expansion region
    nyc = [t for t in tokens if t.text == "nyc"][0]
    assert (nyc.pos, nyc.poslen) == (1, 3)


def test_multi_to_single():
    tokens = analyze("new york city office", {"new york city": "nyc"})
    assert paths(tokens) == [
        ["nyc", "office"],
        ["new", "york", "city", "office"],
    ]


def test_keep_orig_false():
    tokens = analyze("nyc office", {"nyc": "new york city"}, keep_orig=False)
    assert paths(tokens) == [["new", "york", "city", "office"]]


def test_longest_match_wins():
    syn = {"new york": "big apple", "new york city": "nyc"}
    tokens = analyze("new york city", syn)
    assert paths(tokens) == [["nyc"], ["new", "york", "city"]]


def test_overlap_is_deterministic():
    syn = {"a b": "x", "b c": "y"}
    tokens = analyze("a b c", syn)
    # Leftmost-longest: 'a b' matches first, so 'b c' never fires
    assert paths(tokens) == [["x", "c"], ["a", "b", "c"]]


def test_no_recursive_expansion():
    syn = {"nyc": "new york city", "new york city": "big apple"}
    # The expansion of 'nyc' must not trigger the second rule
    tokens = analyze("nyc", syn)
    assert paths(tokens) == [["new", "york", "city"], ["nyc"]]
    # But the second rule fires on its own input
    tokens = analyze("new york city", syn)
    assert paths(tokens) == [["big", "apple"], ["new", "york", "city"]]


def test_multiple_expansions():
    syn = {"usa": ["united states", "america"]}
    tokens = analyze("usa", syn)
    assert paths(tokens) == [["united", "states"], ["america"], ["usa"]]


def test_path_limit():
    tokens = analyze("nyc nyc", {"nyc": "new york city"})
    # Two synonym positions with two alternatives each -> 4 paths
    assert len(token_paths(tokens, max_paths=4)) == 4
    with pytest.raises(SynonymPathLimitError):
        token_paths(tokens, max_paths=3)


def test_position_gap():
    tokens = analyze("visit the office", {}, stoplist={"the"}, renumber=False)
    [path] = token_paths(tokens)
    assert texts(path) == ["visit", "office"]
    assert path_position_gap(path) == 1


def test_phrase_full_paths_match():
    ix = make_index(
        {"nyc": "new york city"},
        [
            "visit new york city office",
            "visit nyc office",
            "new city office",
        ],
    )
    qp = qparser.QueryParser("content", schema=ix.schema)
    q = qp.parse('"visit nyc office"')
    with ix.searcher() as s:
        hits = sorted(h["content"] for h in s.search(q))
    # Both complete legal paths match, the mixed path does not
    assert hits == [
        "visit new york city office",
        "visit nyc office",
    ]


def test_term_query_expands_to_paths():
    ix = make_index(
        {"nyc": "new york city"},
        [
            "visit new york city office",
            "visit nyc office",
            "new city office",
        ],
    )
    qp = qparser.QueryParser("content", schema=ix.schema)
    q = qp.parse("nyc")
    assert isinstance(q, query.Or)
    with ix.searcher() as s:
        hits = sorted(h["content"] for h in s.search(q))
    assert hits == [
        "visit new york city office",
        "visit nyc office",
    ]


def test_index_keeps_original_words():
    ix = make_index({"nyc": "new york city"}, ["visit nyc office"])
    with ix.reader() as r:
        terms = {t for f, t in r.all_terms() if f == "content"}
    # The index is not rewritten by the query-time synonym rules
    assert b"nyc" in terms
    assert b"york" not in terms


def test_two_synonym_positions():
    syn = {"nyc": "new york city", "sf": "san francisco"}
    ix = make_index(
        syn,
        [
            "new york city san francisco office",
            "nyc sf office",
            "new york san francisco office",
        ],
    )
    qp = qparser.QueryParser("content", schema=ix.schema)
    q = qp.parse('"nyc sf office"')
    with ix.searcher() as s:
        hits = sorted(h["content"] for h in s.search(q))
    # The doc mixing halves of two expansions must not match
    assert hits == [
        "new york city san francisco office",
        "nyc sf office",
    ]


def test_words_adjacent_to_expansion():
    ix = make_index(
        {"nyc": "new york city"},
        [
            "visit new york city office",
            "visit nyc office",
            "new city office",
        ],
    )
    qp = qparser.QueryParser("content", schema=ix.schema)
    q = qp.parse('"nyc" office')
    with ix.searcher() as s:
        hits = sorted(h["content"] for h in s.search(q))
    assert hits == [
        "visit new york city office",
        "visit nyc office",
    ]


def test_stopword_position_holes():
    syn = {"nyc": "new york city"}
    docs = [
        "visit the new york city office",
        "visit the nyc office",
    ]
    ix = make_index(syn, docs, stoplist={"the"}, renumber=False)
    qp = qparser.QueryParser("content", schema=ix.schema)
    with ix.searcher() as s:
        # The hole left by the removed stop word is preserved, so the
        # phrase containing the gap matches both documents
        q = qp.parse('"visit the nyc office"')
        hits = sorted(h["content"] for h in s.search(q))
        assert hits == docs
        # A phrase without the gap does not match the gapped positions
        assert s.search(qp.parse('"visit nyc office"')).is_empty()


def test_boost_is_preserved():
    ix = make_index({"nyc": "new york city"}, ["visit nyc office"])
    qp = qparser.QueryParser("content", schema=ix.schema)
    q = qp.parse("nyc^2")
    assert isinstance(q, query.Or)
    assert q.boost == 2.0
    q = qp.parse('"nyc office"^2')
    assert isinstance(q, query.Or)
    assert q.boost == 2.0
    # A single remaining path carries the boost directly
    qp2 = qparser.QueryParser("content", schema=ix.schema)
    q = qp2.parse('"nyc"^2')
    assert q.boost == 2.0


def test_highlighting_matched_words():
    ix = make_index(
        {"nyc": "new york city"},
        [
            "visit new york city office",
            "visit nyc office",
        ],
    )
    qp = qparser.QueryParser("content", schema=ix.schema)
    with ix.searcher() as s:
        r = s.search(qp.parse("nyc"), terms=True)
        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        outputs = {}
        for hit in r:
            outputs[hit["content"]] = hit.highlights("content")
    assert outputs["visit new york city office"] == "visit NEW YORK CITY office"
    assert outputs["visit nyc office"] == "visit NYC office"


def test_path_limit_returns_error():
    syn = {"nyc": "new york city", "sf": "san francisco"}
    ix = make_index(syn, ["nyc sf office"])
    qp = qparser.QueryParser("content", schema=ix.schema)
    qp.max_synonym_paths = 3
    # Four possible paths (2 synonym positions x 2 alternatives)
    q = qp.parse('"nyc sf office"')
    assert q.error is not None
    assert "token paths" in q.error
    # Term queries report the limit the same way
    qp.max_synonym_paths = 1
    q = qp.parse("nyc")
    assert q.error is not None
    # Within the limit, no error
    qp.max_synonym_paths = 4
    q = qp.parse('"nyc sf office"')
    assert q.error is None
