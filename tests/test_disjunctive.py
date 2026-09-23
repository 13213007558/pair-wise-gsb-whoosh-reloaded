import pytest

from whoosh import fields, query, sorting
from whoosh.filedb.filestore import RamStorage


def make_index():
    schema = fields.Schema(
        id=fields.ID(stored=True),
        content=fields.TEXT,
        brand=fields.KEYWORD(stored=True),
        color=fields.KEYWORD(stored=True),
        tags=fields.KEYWORD(stored=True),
        status=fields.ID,
        rank=fields.NUMERIC(sortable=True),
    )
    ix = RamStorage().create_index(schema)

    docs = [
        ("a-red-1", "a", "red", "sale sale new", "active"),
        ("a-blue", "a", "blue", "sale", "active"),
        ("b-red-1", "b", "red", "new", "active"),
        ("b-red-2", "b", "red", "sale", "active"),
        ("b-blue", "b", "blue", None, "active"),
        ("c-red-inactive", "c", "red", "sale", "inactive"),
        ("b-red-deleted", "b", "red", "new", "active"),
        ("missing-brand", None, "red", "new", "active"),
        ("a-red-untagged", "a", "red", None, "active"),
        ("a-red-2", "a", "red", "sale", "active"),
    ]

    for segment_docs, merge in ((docs[:5], True), (docs[5:], False)):
        with ix.writer() as writer:
            writer.merge = merge
            for docid, brand, color, tags, status in segment_docs:
                doc = {
                    "id": docid,
                    "content": "product",
                    "status": status,
                    "rank": 1,
                }
                if brand is not None:
                    doc["brand"] = brand
                if color is not None:
                    doc["color"] = color
                if tags is not None:
                    doc["tags"] = tags
                writer.add_document(**doc)

    with ix.writer() as writer:
        writer.delete_by_term("id", "b-red-deleted")

    return ix


def facet_filters():
    return sorting.FacetFilters(
        {
            "brand": query.Term("brand", "a"),
            "color": query.Term("color", "red"),
            "tags": [query.Term("tags", "sale"), query.Term("tags", "new")],
        }
    )


def groupedby():
    return {
        "brand": sorting.FieldFacet("brand"),
        "color": sorting.FieldFacet("color"),
        "tags": sorting.FieldFacet("tags", allow_overlap=True),
    }


def manual_groups(searcher, fieldname, count_query):
    groups = {}
    for docnum in searcher.docs_for_query(count_query):
        value = searcher.stored_fields(docnum).get(fieldname)
        if fieldname == "tags" and value is not None:
            values = value.split()
        else:
            values = [value]

        for key in set(values):
            groups[key] = groups.get(key, 0) + 1
    return groups


def manual_count_query(main_query, mandatory_query, filters, omit):
    clauses = [main_query, mandatory_query]
    for facetname, filter_query in filters.items():
        if facetname != omit:
            clauses.append(filter_query)
    return query.And(clauses).normalize()


def test_disjunctive_facets_match_manual_queries():
    ix = make_index()
    main_query = query.Term("content", "product")
    mandatory_query = query.Term("status", "active")
    filters = {
        "brand": query.Term("brand", "a"),
        "color": query.Term("color", "red"),
        "tags": query.Or([query.Term("tags", "sale"), query.Term("tags", "new")]),
    }

    with ix.searcher() as searcher:
        results = searcher.search(
            main_query,
            groupedby=groupedby(),
            facet_filters=facet_filters(),
            filter=mandatory_query,
            sortedby=["rank", "id"],
            limit=1,
        )

        assert [hit["id"] for hit in results] == ["a-red-1"]
        assert len(results) == 2

        for facetname in ("brand", "color", "tags"):
            count_query = manual_count_query(
                main_query, mandatory_query, filters, facetname
            )
            expected = manual_groups(searcher, facetname, count_query)
            assert results.groups(facetname) == expected

        assert results.groups("brand") == {"a": 2, "b": 2, None: 1}
        assert results.groups("color") == {"red": 2, "blue": 1}
        assert results.groups("tags") == {"sale": 2, "new": 1, None: 1}


def test_disjunctive_or_within_facet_and_and_between_facets():
    ix = make_index()
    filters = sorting.FacetFilters()
    filters.add_filter(
        "brand", [query.Term("brand", "a"), query.Term("brand", "b")]
    )
    filters.add_filter("color", query.Term("color", "red"))
    filters.add_required(query.Term("status", "active"))

    with ix.searcher() as searcher:
        results = searcher.search(
            query.Term("content", "product"),
            groupedby={"brand": sorting.FieldFacet("brand")},
            facet_filters=filters,
            sortedby="id",
            limit=None,
        )

        assert [hit["id"] for hit in results] == [
            "a-red-1",
            "a-red-2",
            "a-red-untagged",
            "b-red-1",
            "b-red-2",
        ]
        assert results.groups("brand") == {"a": 3, "b": 2, None: 1}


def test_disjunctive_mandatory_filter_and_mask_are_never_omitted():
    ix = make_index()

    with ix.searcher() as searcher:
        results = searcher.search(
            query.Term("content", "product"),
            groupedby=groupedby(),
            facet_filters={
                "brand": query.Term("brand", "a"),
                "color": query.Term("color", "red"),
            },
            required_filters=query.Term("status", "active"),
            mask=query.Term("tags", "new"),
            limit=None,
        )

        assert [hit["id"] for hit in results] == ["a-red-untagged", "a-red-2"]
        assert results.groups("brand") == {"a": 2, "b": 1}
        assert results.groups("color") == {"red": 2, "blue": 1}
        assert results.groups("tags") == {"sale": 1, None: 1}

        empty = searcher.search(
            query.Term("content", "product"),
            groupedby=groupedby(),
            facet_filters={"brand": query.Term("brand", "a")},
            filter=set(),
        )
        assert empty.scored_length() == 0
        assert len(empty) == 0
        assert empty.groups("brand") == {}
        assert empty.groups("color") == {}
        assert empty.groups("tags") == {}


def test_disjunctive_page_size_does_not_change_counts():
    ix = make_index()

    with ix.searcher() as searcher:
        kwargs = {
            "groupedby": groupedby(),
            "facet_filters": facet_filters(),
            "filter": query.Term("status", "active"),
            "sortedby": ["rank", "id"],
        }
        first = searcher.search_page(
            query.Term("content", "product"), 1, pagelen=1, **kwargs
        )
        second = searcher.search_page(
            query.Term("content", "product"), 1, pagelen=2, **kwargs
        )

        assert [hit["id"] for hit in first] == ["a-red-1"]
        assert [hit["id"] for hit in second] == ["a-red-1", "a-red-2"]
        for facetname in ("brand", "color", "tags"):
            assert first.groups(facetname) == second.groups(facetname)


def test_disjunctive_uses_fixed_searcher_snapshot():
    ix = make_index()

    with ix.searcher() as searcher:
        with ix.writer() as writer:
            writer.add_document(
                id="a-red-late",
                content="product",
                brand="a",
                color="red",
                tags="sale",
                status="active",
                rank=1,
            )

        results = searcher.search(
            query.Term("content", "product"),
            groupedby=groupedby(),
            facet_filters=facet_filters(),
            filter=query.Term("status", "active"),
            limit=None,
        )

        assert "a-red-late" not in [hit["id"] for hit in results]
        assert results.groups("brand")["a"] == 2
        assert results.groups("tags")["sale"] == 2


def test_disjunctive_rejects_non_field_facets():
    ix = make_index()
    facet = sorting.QueryFacet({"red": query.Term("color", "red")})

    with ix.searcher() as searcher:
        with pytest.raises(ValueError, match="only supports FieldFacet"):
            searcher.search(
                query.Term("content", "product"),
                groupedby={"color": facet},
                facet_filters={"color": query.Term("color", "red")},
            )
