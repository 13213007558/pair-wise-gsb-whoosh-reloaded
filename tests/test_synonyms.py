import re

import pytest

from whoosh import analysis, fields, qparser, query
from whoosh.analysis.synonyms import (
    SynonymExpansionError,
    SynonymGraphFilter,
    SynonymMap,
    graph_paths,
)
from whoosh.filedb.filestore import RamStorage


def make_analyzer(synmap, **kwargs):
    return (
        analysis.RegexTokenizer()
        | analysis.LowercaseFilter()
        | SynonymGraphFilter(synmap, **kwargs)
    )


def make_index(analyzer, docs):
    schema = fields.Schema(content=fields.TEXT(analyzer=analyzer, stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as writer:
        for doc in docs:
            writer.add_document(content=doc)
    return ix, schema


def search_texts(ix, schema, qstring):
    with ix.searcher() as searcher:
        q = qparser.QueryParser("content", schema).parse(qstring)
        return sorted(hit["content"] for hit in searcher.search(q, limit=None))


def test_synmap_matching():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city")
    synmap.add("new york", "ny")
    synmap.add("new york city", "the big apple")

    # Longest left-hand side wins
    assert synmap.match(["new", "york", "city"], 0) == (
        3,
        [(("the", "big", "apple"), False)],
    )
    assert synmap.match(["new", "york"], 0) == (2, [(("ny",), False)])
    assert synmap.match(["nyc"], 0) == (1, [(("new", "york", "city"), False)])
    assert synmap.match(["boston"], 0) is None

    # Rules with the same left-hand side are returned in insertion order
    synmap.add("nyc", "the big apple")
    _, rules = synmap.match(["nyc"], 0)
    assert rules == [
        (("new", "york", "city"), False),
        (("the", "big", "apple"), False),
    ]

    # Empty sides are rejected
    with pytest.raises(ValueError):
        synmap.add("", "foo")
    with pytest.raises(ValueError):
        synmap.add("foo", "")


def test_filter_token_graph():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ana = make_analyzer(synmap)

    tokens = list(ana("visit nyc office", mode="query", positions=True))
    graph = [(t.text, t.pos, t.poslength) for t in tokens]
    # The kept original spans the whole expansion so paths cannot mix
    assert graph == [
        ("visit", 0, 1),
        ("nyc", 1, 3),
        ("new", 1, 1),
        ("york", 2, 1),
        ("city", 3, 1),
        ("office", 4, 1),
    ]

    paths = [[t.text for t in path] for path in graph_paths(tokens)]
    assert paths == [
        ["visit", "nyc", "office"],
        ["visit", "new", "york", "city", "office"],
    ]


def test_filter_no_recursive_expansion():
    synmap = SynonymMap()
    synmap.add("alfa", "bravo")
    synmap.add("bravo", "charlie")
    ana = make_analyzer(synmap)

    tokens = list(ana("alfa", mode="query", positions=True))
    paths = [[t.text for t in path] for path in graph_paths(tokens)]
    # "bravo" is produced by the first rule but must not trigger the second
    assert paths == [["bravo"]]


def test_filter_overlapping_rules_fixed_order():
    synmap = SynonymMap()
    synmap.add("alfa bravo", "x")
    synmap.add("bravo charlie", "y")
    ana = make_analyzer(synmap)

    tokens = list(ana("alfa bravo charlie", mode="query", positions=True))
    paths = [[t.text for t in path] for path in graph_paths(tokens)]
    # The rule starting at the first token wins; the overlapping rule is
    # never considered
    assert paths == [["x", "charlie"]]


def test_filter_passthrough_when_indexing():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ana = make_analyzer(synmap)

    tokens = list(ana("visit nyc office", mode="index", positions=True))
    assert [(t.text, t.pos) for t in tokens] == [
        ("visit", 0),
        ("nyc", 1),
        ("office", 2),
    ]
    assert not any(hasattr(t, "poslength") for t in tokens)


def test_filter_expansion_limit():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    synmap.add("usa", "united states", keep_orig=True)
    # Two synonym sites with two alternatives each = 4 paths > limit of 3
    ana = make_analyzer(synmap, max_paths=3)

    with pytest.raises(SynonymExpansionError):
        list(ana("nyc office usa", mode="query", positions=True))


def test_graph_paths_limit():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ana = make_analyzer(synmap)
    tokens = list(ana("nyc", mode="query", positions=True))
    with pytest.raises(SynonymExpansionError):
        graph_paths(tokens, max_paths=1)


def test_phrase_full_replacement_matches():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "we should visit the new york city office today",
            "we should visit new york city office today",
            "we should visit new city office today",
            "we should visit nyc office today",
        ],
    )

    hits = search_texts(ix, schema, '"visit nyc office"')
    # The complete original path and the complete replacement path match...
    assert "we should visit nyc office today" in hits
    assert "we should visit new york city office today" in hits
    # ...but a path mixing the two alternatives must not match
    assert "we should visit new city office today" not in hits
    assert "we should visit the new york city office today" not in hits


def test_phrase_multiword_to_word():
    synmap = SynonymMap()
    synmap.add("new york city", "nyc")
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "we should visit nyc office today",
            "we should visit new york city office today",
        ],
    )

    hits = search_texts(ix, schema, '"visit new york city office"')
    assert hits == ["we should visit nyc office today"]


def test_phrase_two_synonym_positions():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    synmap.add("usa", "united states", keep_orig=True)
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "the nyc office usa",
            "the new york city office united states",
            "the nyc office united states",
            "the new york city office usa",
            "the new city office united states",
        ],
    )

    hits = search_texts(ix, schema, '"nyc office usa"')
    assert hits == [
        "the new york city office united states",
        "the new york city office usa",
        "the nyc office united states",
        "the nyc office usa",
    ]


def test_term_query_expansion():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "office in nyc",
            "office in new york city",
            "office in new city",
        ],
    )

    hits = search_texts(ix, schema, "nyc")
    assert hits == ["office in new york city", "office in nyc"]


def test_phrase_followed_by_adjacent_word():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "visit new york city office tomorrow morning",
            "visit nyc office tomorrow morning",
            "visit new york city office yesterday",
        ],
    )

    # The word following the phrase still restricts the results
    hits = search_texts(ix, schema, '"visit nyc office" tomorrow')
    assert hits == [
        "visit new york city office tomorrow morning",
        "visit nyc office tomorrow morning",
    ]


def test_index_keeps_original_words():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ix, schema = make_index(make_analyzer(synmap), ["visit nyc office"])

    with ix.reader() as reader:
        lexicon = list(reader.lexicon("content"))
    # The synonym filter must not inject expansion terms into the index
    assert b"nyc" in lexicon
    assert b"york" not in lexicon
    assert b"city" not in lexicon


def test_phrase_stopword_hole():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    stopper = analysis.StopFilter(
        stoplist=frozenset(["the"]), minsize=1, renumber=False
    )
    ana = (
        analysis.RegexTokenizer()
        | analysis.LowercaseFilter()
        | stopper
        | SynonymGraphFilter(synmap)
    )
    ix, schema = make_index(
        ana,
        [
            "visit the new york city office",
            "visit the nyc office",
            "visit the new city office",
        ],
    )

    hits = search_texts(ix, schema, '"visit the nyc office"')
    assert hits == [
        "visit the new york city office",
        "visit the nyc office",
    ]


def test_phrase_boost_preserved():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    schema = fields.Schema(
        content=fields.TEXT(analyzer=make_analyzer(synmap), stored=True)
    )
    qp = qparser.QueryParser("content", schema)
    q = qp.parse('"visit nyc office"^2')

    assert isinstance(q, query.Or)
    assert len(q.subqueries) == 2
    for subq in q.subqueries:
        assert isinstance(subq, query.Phrase)
        assert subq.boost == 2.0


def test_highlight_matched_words():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    ix, schema = make_index(
        make_analyzer(synmap),
        [
            "we should visit new york city office today",
            "we should visit nyc office today",
        ],
    )

    with ix.searcher() as searcher:
        qp = qparser.QueryParser("content", schema)
        results = searcher.search(qp.parse('"visit nyc office"'), limit=None)
        highlights = {}
        for hit in results:
            highlights[hit["content"]] = hit.highlights("content")

    def matched_words(fragment):
        return re.findall(r'<b class="[^"]*">([^<]+)</b>', fragment)

    expanded = highlights["we should visit new york city office today"]
    for word in ("visit", "new", "york", "city", "office"):
        assert word in matched_words(expanded)
    original = highlights["we should visit nyc office today"]
    for word in ("visit", "nyc", "office"):
        assert word in matched_words(original)


def test_phrase_char_ranges():
    synmap = SynonymMap()
    synmap.add("nyc", "new york city", keep_orig=True)
    schema = fields.Schema(
        content=fields.TEXT(analyzer=make_analyzer(synmap), stored=True)
    )
    qp = qparser.QueryParser("content", schema)
    q = qp.parse('"visit nyc office"')

    assert isinstance(q, query.Or)
    by_first_word = {}
    for subq in q.subqueries:
        by_first_word[subq.words[1]] = subq
    # The expansion tokens map back to the character range of "nyc"
    expanded = by_first_word["new"]
    assert expanded.char_ranges == [(1, 6), (7, 10), (7, 10), (7, 10), (11, 17)]
    original = by_first_word["nyc"]
    assert original.char_ranges == [(1, 6), (7, 10), (11, 17)]
