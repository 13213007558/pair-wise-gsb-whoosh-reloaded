import pytest

from whoosh import analysis, fields, formats, highlight, qparser, query
from whoosh.filedb.filestore import RamStorage
from whoosh.query import spans
from whoosh.util.testing import TempIndex


def make_schema(**kwargs):
    kwargs.setdefault("stored", True)
    return fields.Schema(
        id=fields.ID(stored=True, unique=True),
        body=fields.MULTITEXT(**kwargs),
    )


def test_field_capabilities():
    field = fields.MULTITEXT()
    assert field.multivalue
    assert field.format.supports_multivalue
    assert field.format.supports("positions")
    assert field.format.supports("value_boundaries")

    # Single-value TEXT is not multi-value capable
    text = fields.TEXT()
    assert not text.multivalue
    assert not text.format.supports_multivalue
    assert not text.format.supports("value_boundaries")


def test_format_positions_and_boundaries():
    fmt = formats.MultiPositions(position_gap=100)
    sa = analysis.StandardAnalyzer()
    wvs = {w: vb for w, _, _, vb in fmt.word_values(["bright red", "car repair"], sa)}
    assert fmt.decode_positions(wvs["bright"]) == [0]
    assert fmt.decode_positions(wvs["red"]) == [1]
    # Second value starts past the position gap
    assert fmt.decode_positions(wvs["car"]) == [102]
    assert fmt.decode_value_boundaries(wvs["car"]) == [102]
    # Frequency/weight decoding still works
    assert fmt.decode_frequency(wvs["car"]) == 1


def test_empty_and_stopword_values_dont_glue():
    # Empty values and values containing only stop words must still keep
    # neighboring values separated
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "", "car repair"])
            w.add_document(id="2", body=["bright red", "the", "car repair"])
            w.add_document(id="3", body=["bright red car"])

        with ix.searcher() as s:
            q = query.Phrase("body", ["red", "car"])
            assert [hit["id"] for hit in s.search(q)] == ["3"]


def test_phrase_does_not_cross_values():
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
            w.add_document(id="2", body=["bright red car"])

        with ix.searcher() as s:
            q = query.Phrase("body", ["red", "car"])
            assert [hit["id"] for hit in s.search(q)] == ["2"]

            # Both documents contain both terms
            q = query.And([query.Term("body", "red"), query.Term("body", "car")])
            assert sorted(hit["id"] for hit in s.search(q)) == ["1", "2"]


def test_position_gap_allows_configured_crossing():
    # With a small gap and no strict boundaries, a slop large enough to span
    # the gap can match across values
    schema = make_schema(position_gap=5)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])

        with ix.searcher() as s:
            assert len(s.search(query.Phrase("body", ["red", "car"]))) == 0
            q = query.Phrase("body", ["red", "car"], slop=6)
            assert len(s.search(q)) == 1


def test_strict_boundaries_absolute():
    # Strict mode forbids cross-value matching no matter how large the slop
    schema = make_schema(strict_boundaries=True, position_gap=1)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
            w.add_document(id="2", body=["bright red car"])

        with ix.searcher() as s:
            q = query.Phrase("body", ["red", "car"], slop=1000)
            assert [hit["id"] for hit in s.search(q)] == ["2"]

            # SpanNear and Sequence enforce boundaries too
            t1 = query.Term("body", "red")
            t2 = query.Term("body", "car")
            q = spans.SpanNear(t1, t2, slop=1000)
            assert [hit["id"] for hit in s.search(q)] == ["2"]
            q = query.Sequence([t1, t2], slop=1000)
            assert [hit["id"] for hit in s.search(q)] == ["2"]


def test_value_phrase_query():
    # ValuePhrase forces boundary enforcement even on non-strict fields
    schema = make_schema(position_gap=1)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
            w.add_document(id="2", body=["bright red car"])

        with ix.searcher() as s:
            q = query.ValuePhrase("body", ["red", "car"], slop=1000)
            assert [hit["id"] for hit in s.search(q)] == ["2"]


def test_value_phrase_unsupported_field():
    # ValuePhrase on a field without value boundary information raises a
    # QueryError with a rebuild hint
    schema = fields.Schema(body=fields.TEXT(stored=True))
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(body="bright red car")

        with ix.searcher() as s:
            # Old single-value index reads/queries normally
            assert len(s.search(query.Phrase("body", ["red", "car"]))) == 1

            with pytest.raises(query.QueryError) as excinfo:
                s.search(query.ValuePhrase("body", ["red", "car"]))
            assert "rebuild" in str(excinfo.value)

            with pytest.raises(query.QueryError):
                s.search(
                    spans.SpanNear2(
                        [query.Term("body", "red"), query.Term("body", "car")],
                        respect_boundaries=True,
                    )
                )


def test_stored_value_is_list():
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])

        with ix.searcher() as s:
            hit = s.search(query.Term("body", "red"))[0]
            assert hit["body"] == ["bright red", "car repair"]


def test_highlights_per_value():
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(
                id="1",
                body=["the red car is fast", "nothing to see", "a car wash"],
            )

        with ix.searcher() as s:
            r = s.search(query.Term("body", "car"), terms=True)
            r.formatter = highlight.UppercaseFormatter()
            output = r[0].highlights("body")
            # One result per value, matched terms highlighted within each
            # value, values not joined together
            assert output == [
                "the red CAR is fast",
                "",
                "a CAR wash",
            ]


def test_highlights_phrase_per_value():
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red car", "red", "car"])

        with ix.searcher() as s:
            q = query.Phrase("body", ["red", "car"])
            r = s.search(q, terms=True)
            r.formatter = highlight.UppercaseFormatter()
            output = r[0].highlights("body", strict_phrase=True)
            assert output[0] == "bright RED CAR"
            # The phrase does not span values, so the single-word values are
            # not highlighted as a phrase
            assert output[1] == ""
            assert output[2] == ""


def test_multiple_segments_delete_optimize():
    schema = make_schema(strict_boundaries=True)
    with TempIndex(schema) as ix:
        # Write several segments
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
            w.add_document(id="2", body=["bright red car"])
        with ix.writer() as w:
            w.add_document(id="3", body=["red", "bright", "car"])
            w.add_document(id="4", body=["a bright red car today"])
        with ix.writer() as w:
            w.add_document(id="5", body=["bright red", "red car"])

        phrase = query.Phrase("body", ["red", "car"])

        def check(s, expected):
            hits = {hit["id"]: hit for hit in s.search(phrase, terms=True)}
            assert sorted(hits) == expected
            # Highlight boundaries are consistent too: value 0 contains
            # "red" but not "car", value 1 contains the phrase
            hl = hits["5"].highlights("body")
            assert len(hl) == 2
            assert "red" in hl[0] and "car" not in hl[0]
            assert "<b" in hl[1] and "red" in hl[1] and "car" in hl[1]
            return hits

        with ix.searcher() as s:
            check(s, ["2", "4", "5"])

        # Delete a document and re-check
        with ix.writer() as w:
            w.delete_by_term("id", "4")

        with ix.searcher() as s:
            check(s, ["2", "5"])

        # Optimize (merge all segments) and re-check
        w = ix.writer()
        w.commit(optimize=True)

        with ix.searcher() as s:
            check(s, ["2", "5"])


def test_optimize_merges_postings():
    # The same term in the same field across segments keeps correct
    # positions and boundaries after merging
    schema = make_schema(strict_boundaries=True)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
        with ix.writer() as w:
            w.add_document(id="2", body=["bright red car"])
        w = ix.writer()
        w.commit(optimize=True)

        with ix.searcher() as s:
            assert [
                h["id"] for h in s.search(query.Phrase("body", ["red", "car"]))
            ] == ["2"]
            m = s.postings("body", "red")
            positions = {}
            while m.is_active():
                positions[m.id()] = m.value_as("positions")
                m.next()
            # Positions keep the gap between values after optimize
            assert max(positions[0]) < 10
            assert max(positions[1]) < 10


def test_characters_format_per_value_offsets():
    schema = make_schema(chars=True)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "red car"])

        with ix.searcher() as s:
            m = s.postings("body", "red")
            chars = m.value_as("characters")
            # Character offsets are per-value: the second "red" starts at
            # character 0 of the second value
            assert chars == [(1, 7, 10), (102, 0, 3)]
            assert m.value_as("value_boundaries") == [102]


def test_single_string_value():
    # A plain string value still works like a single-value field
    schema = make_schema()
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body="bright red car")

        with ix.searcher() as s:
            assert len(s.search(query.Phrase("body", ["red", "car"]))) == 1
            hit = s.search(query.Term("body", "red"))[0]
            assert hit["body"] == "bright red car"
            assert hit.highlights("body") == "bright <b>red</b> car".replace(
                "<b>", '<b class="match term0">'
            )


def test_query_parser_phrase():
    schema = make_schema(strict_boundaries=True)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(id="1", body=["bright red", "car repair"])
            w.add_document(id="2", body=["bright red car"])

        with ix.searcher() as s:
            qp = qparser.QueryParser("body", ix.schema)
            q = qp.parse('"red car"')
            assert [hit["id"] for hit in s.search(q)] == ["2"]
