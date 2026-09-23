import pytest

from whoosh import fields, query, sorting
from whoosh.disjunctive import DisjunctiveResults, FacetFilters, search_disjunctive
from whoosh.filedb.filestore import RamStorage


def get_schema():
    return fields.Schema(
        id=fields.ID(stored=True, unique=True),
        name=fields.TEXT(stored=True),
        brand=fields.ID(stored=True),
        color=fields.ID(stored=True),
        tags=fields.KEYWORD(stored=True, commas=True),
        instock=fields.BOOLEAN(stored=True),
        price=fields.NUMERIC(int, stored=True, sortable=True),
    )


DOCS = [
    dict(id="1", name="red apple", brand="a", color="red",
         tags="sale,new", instock=True, price=10),
    dict(id="2", name="red cherry", brand="a", color="red",
         tags="sale,sale", instock=True, price=20),
    dict(id="3", name="blue berry", brand="a", color="blue",
         tags="new", instock=False, price=30),
    dict(id="4", name="red pepper", brand="b", color="red",
         tags="sale,clearance", instock=True, price=10),
    dict(id="5", name="red tomato", brand="b", color="red",
         instock=True, price=50),  # no tags field
    dict(id="6", name="green lime", brand="b", color="green",
         tags="clearance", instock=True, price=60),
    dict(id="7", name="yellow lemon", brand="a",
         tags="new", instock=True, price=70),  # no color field
    # This document is deleted below and must never show up anywhere
    dict(id="8", name="red deleted", brand="b", color="red",
         tags="sale", instock=True, price=80),
]


@pytest.fixture(scope="module")
def ix():
    storage = RamStorage()
    ix = storage.create_index(get_schema())
    with ix.writer() as w:
        for doc in DOCS:
            w.add_document(**doc)
    with ix.writer() as w:
        w.delete_by_query(query.Term("id", "8"))
    return ix


def hit_ids(results):
    return sorted(hit["id"] for hit in results)


def manual_counts(s, q, fieldname):
    # Cross-check: count the buckets by hand with a plain grouped search
    facet = sorting.FieldFacet(fieldname, allow_overlap=True)
    r = s.search(q, limit=None, groupedby=facet, maptype=sorting.Count)
    return r.groups(fieldname)


def test_hits_apply_all_filters(ix):
    with ix.searcher() as s:
        filters = FacetFilters()
        filters.add("brand", query.Term("brand", "a"))
        filters.add("color", query.Term("color", "red"))

        dr = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=None
        )
        assert isinstance(dr, DisjunctiveResults)
        # The hit list is NOT relaxed: only docs matching every filter
        assert hit_ids(dr.results) == ["1", "2"]


def test_counts_match_manual_queries(ix):
    with ix.searcher() as s:
        filters = FacetFilters()
        filters.add("brand", query.Term("brand", "a"))
        filters.add("color", query.Term("color", "red"))

        dr = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=None
        )

        # Brand counts drop the brand filter but keep color=red, so we can
        # still see how many red items brand "b" has
        assert dr.counts("brand") == {"a": 2, "b": 2}
        assert dr.counts("brand") == manual_counts(
            s, query.Term("color", "red"), "brand"
        )

        # Color counts drop the color filter but keep brand=a
        assert dr.counts("color") == {"red": 2, "blue": 1, None: 1}
        assert dr.counts("color") == manual_counts(
            s, query.Term("brand", "a"), "color"
        )


def test_counts_respect_main_query(ix):
    with ix.searcher() as s:
        filters = FacetFilters()
        filters.add("color", query.Term("color", "red"))

        dr = s.search_disjunctive(
            query.Term("name", "cherry"), ["brand", "color"],
            filters=filters, limit=None,
        )
        assert hit_ids(dr.results) == ["2"]
        # The main query still applies to the counts
        assert dr.counts("brand") == {"a": 1}
        assert dr.counts("brand") == manual_counts(
            s, query.Term("name", "cherry"), "brand"
        )
        assert dr.counts("color") == {"red": 1}


def test_or_within_facet_and_across_facets(ix):
    with ix.searcher() as s:
        filters = FacetFilters()
        # Multiple options in the same facet are ORed
        filters.add("color", query.Term("color", "red"))
        filters.add("color", query.Term("color", "blue"))
        filters.add("brand", query.Term("brand", "a"))

        dr = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=None
        )
        # (red OR blue) AND brand=a
        assert hit_ids(dr.results) == ["1", "2", "3"]

        # Color counts drop the whole color group (both options)
        assert dr.counts("color") == manual_counts(
            s, query.Term("brand", "a"), "color"
        )
        # Brand counts drop the brand group but keep (red OR blue)
        expected = manual_counts(
            s,
            query.Or([query.Term("color", "red"), query.Term("color", "blue")]),
            "brand",
        )
        assert dr.counts("brand") == expected == {"a": 3, "b": 2}


def test_mandatory_filters_never_dropped(ix):
    with ix.searcher() as s:
        filters = FacetFilters(mandatory=[query.Term("instock", True)])
        filters.add("brand", query.Term("brand", "a"))

        dr = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=None
        )
        # Document 3 is out of stock: it must not appear in any count
        assert dr.counts("brand") == {"a": 3, "b": 3}
        assert dr.counts("color") == {"red": 2, None: 1}
        assert dr.counts("brand") == manual_counts(
            s, query.Term("instock", True), "brand"
        )
        assert dr.counts("color") == manual_counts(
            s,
            query.And([query.Term("instock", True), query.Term("brand", "a")]),
            "color",
        )


def test_multivalued_doc_counts_once_per_bucket(ix):
    with ix.searcher() as s:
        dr = s.search_disjunctive(query.Every(), ["tags"], limit=None)
        counts = dr.counts("tags")
        # Doc 2 has "sale" twice but is only counted once; multi-valued docs
        # count towards each of their values
        assert counts == {"sale": 3, "new": 3, "clearance": 2, None: 1}
        assert counts == manual_counts(s, query.Every(), "tags")


def test_missing_value_bucket(ix):
    with ix.searcher() as s:
        dr = s.search_disjunctive(query.Every(), ["color"], limit=None)
        # Doc 7 has no color field
        assert dr.counts("color")[None] == 1

        # The missing bucket key is configurable
        dr = s.search_disjunctive(query.Every(), ["color"], missing="(none)",
                                  limit=None)
        counts = dr.counts("color")
        assert counts["(none)"] == 1
        assert None not in counts


def test_deleted_documents_are_excluded(ix):
    with ix.searcher() as s:
        dr = s.search_disjunctive(
            query.Every(), ["brand", "color", "tags"], limit=None
        )
        # Document 8 was deleted: brand "b" has 3 docs, not 4
        assert dr.counts("brand") == {"a": 4, "b": 3}
        assert dr.counts("color")["red"] == 4
        assert dr.counts("tags")["sale"] == 3
        assert "8" not in hit_ids(dr.results)


def test_page_size_does_not_affect_counts(ix):
    filters = FacetFilters()
    filters.add("brand", query.Term("brand", "a"))

    with ix.searcher() as s:
        dr1 = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters,
            pagenum=1, pagelen=1, sortedby="price",
        )
        dr2 = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters,
            pagenum=2, pagelen=2, sortedby="price",
        )
        dr3 = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=1,
        )
        dr4 = s.search_disjunctive(
            query.Every(), ["brand", "color"], filters=filters, limit=None,
        )

        assert dr1.counts() == dr2.counts() == dr3.counts() == dr4.counts()
        # The page contents still reflect the requested page size
        assert dr1.results.pagelen == 1
        assert [h["id"] for h in dr2.results] == ["3", "7"]
        assert len(dr3.results.top_n) == 1
        assert len(dr4.results.top_n) == 4


def test_bucket_ordering_is_stable_and_configurable(ix):
    with ix.searcher() as s:
        dr = s.search_disjunctive(query.Every(), ["brand", "tags"],
                                  limit=None)

        # Default: count descending, ties broken by bucket name, missing last
        assert dr.buckets("brand") == [("a", 4), ("b", 3)]
        assert dr.buckets("tags") == [
            ("new", 3),
            ("sale", 3),
            ("clearance", 2),
            (None, 1),
        ]

        # Sort by name, missing bucket still last
        assert dr.buckets("tags", sortby="name") == [
            ("clearance", 2),
            ("new", 3),
            ("sale", 3),
            (None, 1),
        ]

        # Reversed
        assert dr.buckets("brand", reverse=True) == [("b", 3), ("a", 4)]

        # Custom key
        buckets = dr.buckets("tags", sortby=lambda bucket, count: count)
        assert buckets[0] == (None, 1)
        assert buckets[-1] in (("new", 3), ("sale", 3))


def test_sorted_hits_have_stable_ties(ix):
    with ix.searcher() as s:
        dr = s.search_disjunctive(
            query.Every(), ["brand"], sortedby="price", limit=None
        )
        ids = [hit["id"] for hit in dr.results]
        # Docs 1 and 4 both have price=10; ties break by document number
        assert ids == ["1", "4", "2", "3", "5", "6", "7"]

        dr = s.search_disjunctive(
            query.Every(), ["brand"], sortedby="price", reverse=True,
            limit=None,
        )
        assert [hit["id"] for hit in dr.results] == list(reversed(ids))


def test_filters_argument_forms(ix):
    with ix.searcher() as s:
        ff = FacetFilters()
        ff.add("brand", query.Term("brand", "a"))
        ff.add_mandatory(query.Term("instock", True))
        expected = s.search_disjunctive(
            query.Every(), ["brand"], filters=ff, limit=None
        ).counts()

        # A dict maps facet names to queries (or lists of queries)
        as_dict = s.search_disjunctive(
            query.Every(), ["brand"],
            filters={"brand": query.Term("brand", "a")}, limit=None,
        ).counts()

        # A bare query is treated as a mandatory filter
        bare = s.search_disjunctive(
            query.Every(), ["brand"],
            filters=query.Term("instock", True), limit=None,
        ).counts()

        assert as_dict["brand"] == {"a": 4, "b": 3}
        assert bare["brand"] == {"a": 3, "b": 3}
        assert expected["brand"] == {"a": 3, "b": 3}


def test_facet_filters_structure():
    ff = FacetFilters()
    assert not ff
    ff.add("color", query.Term("color", "red"))
    ff.add("color", query.Term("color", "blue"))
    ff.add("brand", query.Term("brand", "a"))
    ff.add_mandatory(query.Term("instock", True))
    assert ff

    # Options within a facet are ORed
    assert ff.group_query("color") == query.Or(
        [query.Term("color", "red"), query.Term("color", "blue")]
    )
    assert ff.group_query("brand") == query.Term("brand", "a")
    assert ff.group_query("unknown") is None

    # Groups are ANDed together with the mandatory filters
    assert ff.combined_query() == query.And(
        [
            query.Term("instock", True),
            query.Or([query.Term("color", "red"), query.Term("color", "blue")]),
            query.Term("brand", "a"),
        ]
    )
    # Excluding a facet drops only its own group
    assert ff.combined_query(exclude="color") == query.And(
        [query.Term("instock", True), query.Term("brand", "a")]
    )
    assert ff.combined_query(exclude="brand") == query.And(
        [
            query.Term("instock", True),
            query.Or([query.Term("color", "red"), query.Term("color", "blue")]),
        ]
    )

    with pytest.raises(TypeError):
        ff.add("color", "red")
