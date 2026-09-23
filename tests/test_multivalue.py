import pytest

from whoosh import analysis, fields, formats, highlight, qparser, query
from whoosh.filedb.filestore import RamStorage
from whoosh.query import spans


def make_ix(**kwargs):
    stored = kwargs.pop("stored", True)
    schema = fields.Schema(
        ident=fields.ID(stored=True),
        tags=fields.MULTITEXT(stored=stored, **kwargs),
    )
    return RamStorage().create_index(schema)


def add_docs(ix, docs):
    w = ix.writer()
    for ident, tags in docs:
        w.add_document(ident=ident, tags=tags)
    w.commit()


def idents(ix, q):
    with ix.searcher() as s:
        return sorted(hit["ident"] for hit in s.search(q, limit=None))


def test_phrase_does_not_cross_values():
    ix = make_ix()
    add_docs(
        ix,
        [
            ("cross", ["bright red", "car repair"]),
            ("same", ["bright red car"]),
        ],
    )

    q = query.Phrase("tags", ["red", "car"])
    assert idents(ix, q) == ["same"]

    # Term queries are unaffected by value boundaries
    assert idents(ix, query.Term("tags", "red")) == ["cross", "same"]
    assert idents(ix, query.Term("tags", "car")) == ["cross", "same"]

    # Same thing through the query parser
    qp = qparser.QueryParser("tags", ix.schema)
    assert idents(ix, qp.parse('"red car"')) == ["same"]


def test_strict_mode_beats_slop():
    ix = make_ix()
    add_docs(ix, [("cross", ["red", "car"]), ("same", ["red car"])])

    # Even a huge slop cannot cross a value boundary in strict mode
    q = query.Phrase("tags", ["red", "car"], slop=100000)
    assert idents(ix, q) == ["same"]


def test_gap_mode_allows_crossing_with_enough_slop():
    ix = make_ix(position_gap=2, cross_values=True)
    add_docs(ix, [("cross", ["bright red", "car repair"])])

    # red is at position 1, car is at position 1 + 1 + 2 = 4
    assert idents(ix, query.Phrase("tags", ["red", "car"], slop=1)) == []
    assert idents(ix, query.Phrase("tags", ["red", "car"], slop=2)) == []
    assert idents(ix, query.Phrase("tags", ["red", "car"], slop=3)) == ["cross"]


def test_positions_and_value_indexes():
    ix = make_ix(position_gap=10)
    add_docs(ix, [("a", ["bright red", "", "car"])])

    r = ix.reader()
    m = r.postings("tags", b"red")
    assert m.value_as("positions") == [1]
    assert m.value_as("value_positions") == [(1, 0)]

    # The empty value still consumes a position gap: car is at
    # (1 + 1 + 10) + (0 + 10) = 22, in value 2
    m = r.postings("tags", b"car")
    assert m.value_as("positions") == [22]
    assert m.value_as("value_positions") == [(22, 2)]
    r.close()


def test_empty_and_stopword_values_dont_glue():
    ix = make_ix()
    add_docs(
        ix,
        [
            ("empty", ["red", "", "car"]),
            ("stops", ["red", "the a an", "car"]),
            ("intact", ["bright red", "the", "car repair"]),
        ],
    )

    # Nothing can match across the empty/stopword-only values
    q = query.Phrase("tags", ["red", "car"], slop=100000)
    assert idents(ix, q) == []

    # The value after a stopword-only value is still indexed normally
    assert idents(ix, query.Phrase("tags", ["car", "repair"])) == ["intact"]
    assert idents(ix, query.Term("tags", "car")) == ["empty", "intact", "stops"]


def test_stopword_positions_stay_within_value():
    # Use an analyzer that keeps the positions of removed stop words
    ana = (
        analysis.RegexTokenizer()
        | analysis.LowercaseFilter()
        | analysis.StopFilter(renumber=False)
    )
    ix = make_ix(analyzer=ana)
    add_docs(ix, [("a", ["red the car"])])

    # "the" is removed but its position is preserved inside the value
    assert idents(ix, query.Phrase("tags", ["red", "car"], slop=1)) == []
    assert idents(ix, query.Phrase("tags", ["red", "car"], slop=2)) == ["a"]


def test_stored_value_is_the_original_list():
    ix = make_ix()
    values = ["bright red", "car repair"]
    add_docs(ix, [("a", values)])

    with ix.searcher() as s:
        hit = s.search(query.Term("tags", "red"))[0]
        assert hit["tags"] == values
        assert hit.fields()["tags"] == values


def test_highlights_are_per_value():
    ix = make_ix()
    add_docs(
        ix,
        [("a", ["bright red car", "unrelated text", "red automobile"])],
    )

    with ix.searcher() as s:
        r = s.search(query.Phrase("tags", ["red", "car"]), terms=True)
        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        hit = r[0]
        # Fragments come back keyed by value index, not joined together
        assert hit.highlights("tags") == [
            (0, "bright RED CAR"),
            (2, "RED automobile"),
        ]


def test_highlight_single_value_field_still_returns_string():
    schema = fields.Schema(ident=fields.ID(stored=True), title=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(ident="a", title="bright red car")
    w.commit()

    with ix.searcher() as s:
        r = s.search(query.Term("title", "red"), terms=True)
        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        assert r[0].highlights("title") == "bright RED car"


def test_span_queries_respect_boundaries():
    ix = make_ix()
    add_docs(
        ix,
        [
            ("cross", ["alfa red", "car omega"]),
            ("same", ["alfa red car omega"]),
        ],
    )

    red = query.Term("tags", "red")
    car = query.Term("tags", "car")

    q = spans.SpanNear(red, car, slop=100000)
    assert idents(ix, q) == ["same"]

    q = spans.SpanNear2([red, car], slop=100000)
    assert idents(ix, q) == ["same"]

    q = spans.SpanBefore(red, car)
    assert idents(ix, q) == ["same"]

    q = query.Sequence([red, car], slop=100000)
    assert idents(ix, q) == ["same"]


def test_span_or_does_not_merge_across_values():
    ix = make_ix(position_gap=1)
    add_docs(ix, [("a", ["red", "car"])])

    with ix.searcher() as s:
        q = spans.SpanOr([query.Term("tags", "red"), query.Term("tags", "car")])
        m = q.matcher(s)
        m.next()
        # red is at position 0 (value 0), car at position 2 (value 1): the
        # spans must not be merged into a single 0-2 span
        assert [(span.start, span.end) for span in m.spans()] == [(0, 0), (2, 2)]


def test_segments_deletes_and_optimize():
    ix = make_ix()

    # First segment
    w = ix.writer()
    w.add_document(ident="cross", tags=["bright red", "car repair"])
    w.add_document(ident="same", tags=["bright red car"])
    w.commit()

    # Second segment
    w = ix.writer()
    w.add_document(ident="same2", tags=["unrelated", "red car"])
    w.add_document(ident="reversed", tags=["car", "red"])
    w.commit()

    # Delete one of the matching docs
    w = ix.writer()
    w.delete_by_term("ident", "same")
    w.commit()

    def check():
        phrase = query.Phrase("tags", ["red", "car"])
        assert idents(ix, phrase) == ["same2"]
        with ix.searcher() as s:
            r = s.search(phrase, terms=True)
            r.fragmenter = highlight.WholeFragmenter()
            r.formatter = highlight.UppercaseFormatter()
            assert r[0].highlights("tags") == [(1, "RED CAR")]

    # Multiple segments with a pending deletion
    check()

    # Everything must be identical after optimization
    ix.optimize()
    check()


def test_characters_format():
    ix = make_ix(chars=True)
    add_docs(ix, [("a", ["bright red", "red car"])])

    assert idents(ix, query.Phrase("tags", ["red", "car"])) == ["a"]

    r = ix.reader()
    m = r.postings("tags", b"red")
    # Character offsets keep their per-value meaning
    assert m.value_as("value_characters") == [(1, 0, 7, 10), (102, 1, 0, 3)]
    assert m.value_as("characters") == [(1, 7, 10), (102, 0, 3)]
    r.close()


def test_frequency_only_multitext():
    ix = make_ix(phrase=False)
    add_docs(ix, [("a", ["bright red", "car repair"])])

    assert idents(ix, query.Term("tags", "car")) == ["a"]
    with ix.searcher() as s:
        assert s.search(query.Term("tags", "red"))[0]["tags"] == [
            "bright red",
            "car repair",
        ]

    # Phrase queries on a position-less multi-valued field get a useful error
    with ix.searcher() as s:
        with pytest.raises(query.QueryError) as excinfo:
            query.Phrase("tags", ["red", "car"]).matcher(s)
        assert "MULTITEXT" in str(excinfo.value)


def test_capability_flags():
    # The new field type and formats are recognizable
    field = fields.MULTITEXT()
    assert field.multivalue
    assert field.format.multivalue
    assert field.format.supports("value_positions")
    assert field.format.position_gap == 100
    assert field.format.strict

    field = fields.MULTITEXT(position_gap=5, cross_values=True)
    assert field.format.position_gap == 5
    assert not field.format.strict

    # Old single-value formats don't claim the capability
    old = fields.TEXT()
    assert not old.multivalue
    assert not old.format.multivalue
    assert not old.format.supports("value_positions")


def test_old_single_value_index_reads_normally():
    schema = fields.Schema(ident=fields.ID(stored=True), title=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(ident="a", title="bright red car")
    w.commit()

    assert idents(ix, query.Phrase("title", ["red", "car"])) == ["a"]
    with ix.searcher() as s:
        r = s.search(query.Term("title", "red"), terms=True)
        assert isinstance(r[0].highlights("title"), str)


def test_strict_field_without_value_positions_errors():
    ix = make_ix()
    add_docs(ix, [("a", ["bright red car"])])

    with ix.searcher() as s:
        # Simulate a field whose index lacks multi-value position
        # information (for example an index built before the feature
        # existed): the query must fail with a rebuild hint
        s.schema["tags"].format = formats.Positions()
        with pytest.raises(query.QueryError) as excinfo:
            query.Phrase("tags", ["red", "car"]).matcher(s)
        message = str(excinfo.value)
        assert "MULTITEXT" in message
        assert "Rebuild" in message


def test_schema_roundtrip():
    storage = RamStorage()
    ix = storage.create_index(
        fields.Schema(tags=fields.MULTITEXT(position_gap=7, stored=True)),
        indexname="test",
    )
    ix.close()

    ix = storage.open_index("test")
    field = ix.schema["tags"]
    assert isinstance(field, fields.MULTITEXT)
    assert field.format.position_gap == 7
    assert field.format.strict
    assert field == fields.MULTITEXT(position_gap=7, stored=True)


def test_single_string_value_still_works():
    ix = make_ix()
    add_docs(ix, [("a", "bright red car")])

    assert idents(ix, query.Phrase("tags", ["red", "car"])) == ["a"]
    with ix.searcher() as s:
        assert s.search(query.Term("tags", "red"))[0]["tags"] == "bright red car"


def test_vector():
    ix = make_ix(vector=True)
    add_docs(ix, [("a", ["bright red", "car repair"])])

    with ix.searcher() as s:
        docnum = s.document_number(ident="a")
        v = s.vector(docnum, "tags")
        items = dict(v.items_as("value_positions"))
        assert items["car"] == [(102, 1)]
