'''Disjunctive (a.k.a. "multi-select") faceting.

In a faceted search interface, when the user has already selected a value in
one facet (for example brand=A), you usually still want to display the
counts for the *other* values of that facet (how many red items does brand B
have?), computed as if the filter on that facet were not applied, while still
respecting the filters on all other facets and the main query.

The entry point is :func:``search_disjunctive`` (also available as
:meth:``whoosh.searching.Searcher.search_disjunctive``). The caller explicitly
groups filter queries by the facet they belong to using :class:``FacetFilters``;
this module never tries to guess facet membership from an arbitrary query
tree.
'''

from whoosh import collectors, query, sorting


class FacetFilters:
    '''Groups of filter queries associated with facet names.

    Filters added under the same facet name are combined with OR. Different
    facet groups are combined with AND. Filters added with
    :meth:``add_mandatory`` do not belong to any facet: they are always applied,
    both to the hits and to every facet's counts::

        filters = FacetFilters()
        filters.add("brand", query.Term("brand", "a"))
        filters.add("color", query.Term("color", "red"))
        filters.add("color", query.Term("color", "blue"))  # red OR blue
        filters.add_mandatory(query.Term("instock", True))
    '''

    def __init__(self, mandatory=None):
        '''
        :param mandatory: an optional query or list of queries that must
            always be applied (to the hits and to all facet counts).
        '''

        self._groups = {}
        self._mandatory = []
        if mandatory is not None:
            if isinstance(mandatory, query.Query):
                mandatory = [mandatory]
            for q in mandatory:
                self.add_mandatory(q)

    def __repr__(self):
        return "{}({!r}, mandatory={!r})".format(
            self.__class__.__name__, self._groups, self._mandatory
        )

    def __bool__(self):
        return bool(self._groups) or bool(self._mandatory)

    def add(self, facetname, q):
        '''Adds a filter query belonging to the given facet. Multiple filters
        under the same facet name are ORed together.
        '''

        if not isinstance(q, query.Query):
            raise TypeError(f"{q!r} is not a query object")
        self._groups.setdefault(facetname, []).append(q)
        return self

    def add_mandatory(self, q):
        '''Adds a filter query that does not belong to any facet. Mandatory
        filters are never excluded when computing facet counts.
        '''

        if not isinstance(q, query.Query):
            raise TypeError(f"{q!r} is not a query object")
        self._mandatory.append(q)
        return self

    def names(self):
        '''Returns an iterator of the facet names that have filters.'''

        return iter(self._groups)

    def items(self):
        '''Returns a list of (facetname, [queries]) tuples.'''

        return [(name, list(qs)) for name, qs in self._groups.items()]

    def mandatory(self):
        '''Returns a list of the mandatory (facet-less) filter queries.'''

        return list(self._mandatory)

    def group_query(self, facetname):
        '''Returns the queries for one facet combined with OR, or None if the
        facet has no filters.
        '''

        qs = self._groups.get(facetname)
        if not qs:
            return None
        if len(qs) == 1:
            return qs[0]
        return query.Or(qs)

    def combined_query(self, exclude=None):
        '''Returns the mandatory filters ANDed with each facet's ORed group,
        optionally leaving out one facet's group. Returns None if there is
        nothing to combine.
        '''

        qs = list(self._mandatory)
        for name in self._groups:
            if name == exclude:
                continue
            group_q = self.group_query(name)
            if group_q is not None:
                qs.append(group_q)
        return _and_all(qs)


def _and_all(qs):
    qs = [q for q in qs if q is not None]
    if not qs:
        return None
    if len(qs) == 1:
        return qs[0]
    return query.And(qs)


def _to_facet_filters(filters):
    if filters is None:
        return FacetFilters()
    if isinstance(filters, FacetFilters):
        return filters
    if isinstance(filters, query.Query):
        # A bare query is treated as a mandatory filter
        return FacetFilters(mandatory=[filters])
    if isinstance(filters, dict):
        ff = FacetFilters()
        for name, f in filters.items():
            if isinstance(f, (list, tuple)):
                for q in f:
                    ff.add(name, q)
            else:
                ff.add(name, f)
        return ff
    raise TypeError(f"Don't know what to do with filters={filters!r}")


class _BucketCategorizer(sorting.Categorizer):
    '''Wraps another categorizer to guarantee overlap semantics for counting:
    each document contributes at most once to any bucket, and documents with
    no value fall into a ``missing`` bucket.
    '''

    allow_overlap = True

    def __init__(self, catter, missing):
        self._catter = catter
        self._missing = missing
        self._sentinel = object()

    @property
    def needs_current(self):
        return self._catter.needs_current

    def set_searcher(self, segment_searcher, docoffset):
        self._catter.set_searcher(segment_searcher, docoffset)

    def keys_for(self, matcher, segment_docnum):
        catter = self._catter
        if catter.allow_overlap:
            keys = catter.keys_for(matcher, segment_docnum)
        else:
            keys = (catter.key_for(matcher, segment_docnum),)

        # De-duplicate the keys so a document is only counted once per bucket
        unique = []
        for key in keys:
            if key is None:
                continue
            if key not in unique:
                unique.append(key)
        if not unique:
            unique.append(self._sentinel)
        return unique

    def key_to_name(self, key):
        if key is self._sentinel:
            return self._missing
        return self._catter.key_to_name(key)


class _CategorizerFacet(sorting.FacetType):
    '''Facet that simply returns a pre-built categorizer.'''

    def __init__(self, catter):
        self._catter = catter

    def categorizer(self, global_searcher):
        return self._catter


def _count_buckets(searcher, q, facet, missing):
    # For counting we always want overlap semantics: a document with multiple
    # values in the field counts towards each value's bucket (but only once
    # per bucket)
    if isinstance(facet, sorting.FieldFacet):
        facet = sorting.FieldFacet(facet.fieldname, allow_overlap=True)

    catter = _BucketCategorizer(facet.categorizer(searcher), missing)
    child = collectors.UnsortedCollector()
    fc = collectors.FacetCollector(
        child, {"facet": _CategorizerFacet(catter)}, maptype=sorting.Count
    )
    searcher.search_with_collector(q, fc)
    return fc.facetmaps["facet"].as_dict()


def _name_key(bucket):
    # Sort the ``missing`` bucket (None) after everything else
    if bucket is None:
        return (1, "")
    return (0, bucket)


class DisjunctiveResults:
    '''The result of a disjunctive faceted search: the hits (with all filters
    applied) plus per-facet bucket counts (each computed without that facet's
    own filters).

    ``results`` is the :class:``whoosh.searching.Results`` (or
    :class:``whoosh.searching.ResultsPage``) object for the hits.
    '''

    def __init__(self, results, counts, missing=None):
        self.results = results
        self._counts = counts
        self.missing = missing

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.results!r} {self._counts!r}>"

    def facet_names(self):
        '''Returns an iterator of the faceted field names.'''

        return iter(self._counts)

    def counts(self, name=None):
        '''Returns the bucket counts for one facet, or a dictionary of all
        facets if no name is given.
        '''

        if name is None:
            return {facetname: dict(c) for facetname, c in self._counts.items()}
        return dict(self._counts[name])

    def buckets(self, name, sortby="count", reverse=False):
        '''Returns a list of (bucket, count) pairs for the given facet.

        :param name: the facet name.
        :param sortby: ``count`` (the default) sorts by descending count,
            breaking ties by bucket name; ``name`` sorts by bucket name;
            None leaves the buckets in their natural order; or pass a
            callable taking (bucket, count) and returning a sort key.
            The ``missing`` bucket always sorts last for the built-in orderings.
        :param reverse: reverse the final ordering.
        '''

        items = list(self._counts[name].items())
        if sortby is None:
            pass
        elif sortby == "count":
            items.sort(key=lambda kv: (-kv[1], _name_key(kv[0])))
        elif sortby == "name":
            items.sort(key=lambda kv: _name_key(kv[0]))
        elif callable(sortby):
            items.sort(key=lambda kv: sortby(kv[0], kv[1]))
        else:
            raise ValueError(f"Don't know how to sort by {sortby!r}")
        if reverse:
            items.reverse()
        return items


def search_disjunctive(
    searcher,
    q,
    facets,
    filters=None,
    limit=10,
    sortedby=None,
    reverse=False,
    pagenum=None,
    pagelen=10,
    missing=None,
    optimize=True,
    scored=True,
    terms=False,
    **kwargs
):
    '''Runs a search with disjunctive faceting on a single searcher.

    The hits are computed with the main query ANDed with *all* filters. The
    bucket counts for each facet are computed with the main query ANDed with
    all filters *except* the filters belonging to that facet, so the user can
    still see (and select) the other values of facets they have already
    filtered on. Mandatory filters (see :class:``FacetFilters``) are applied to
    the hits and to every facet's counts.

    All queries run on the given searcher, so the hits and the counts come
    from the same fixed index snapshot. Facet counts are always computed over
    the full set of matching documents; ``limit``, ``pagenum``, and ``pagelen``
    only affect the returned hits, never the counts.

    :param searcher: the :class:``whoosh.searching.Searcher`` to search on.
    :param q: the main :class:``whoosh.query.Query`` object (or None to match
        every document).
    :param facets: the facets to count, for example a dictionary mapping facet
        names to :class:``whoosh.sorting.FieldFacet`` objects, a list of field
        names, or a :class:``whoosh.sorting.Facets`` object.
    :param filters: a :class:``FacetFilters`` object, a dictionary mapping facet
        names to queries (or lists of queries), or a single query object to
        use as a mandatory filter.
    :param limit: the maximum number of hits to return (as in
        :meth:``whoosh.searching.Searcher.search``).
    :param sortedby: a facet to sort the hits by; ties are broken by document
        number, so the ordering is stable.
    :param reverse: reverse the order of the hits.
    :param pagenum: if given, return this page of hits (numbered from 1) as a
        :class:``whoosh.searching.ResultsPage`` instead of a
        :class:``whoosh.searching.Results`` object.
    :param pagelen: the page length when ``pagenum`` is given.
    :param missing: the bucket key to use for documents that have no value in
        a faceted field. The default is None.
    :rtype: :class:``DisjunctiveResults``
    '''

    facets = sorting.Facets.from_groupedby(facets)
    ffilters = _to_facet_filters(filters)
    base_q = q if q is not None else query.Every()

    # The hits always see every filter
    full_q = _and_all([base_q, ffilters.combined_query()])
    search_kwargs = dict(
        sortedby=sortedby,
        reverse=reverse,
        optimize=optimize,
        scored=scored,
        terms=terms,
    )
    search_kwargs.update(kwargs)
    if pagenum is not None:
        hits = searcher.search_page(
            full_q, pagenum, pagelen=pagelen, **search_kwargs
        )
    else:
        hits = searcher.search(full_q, limit=limit, **search_kwargs)

    # Each facet's counts ignore that facet's own filters, but still apply
    # the main query, the mandatory filters, and the other facets' filters
    counts = {}
    for name, facet in facets.items():
        facet_q = _and_all([base_q, ffilters.combined_query(exclude=name)])
        counts[name] = _count_buckets(searcher, facet_q, facet, missing)

    return DisjunctiveResults(hits, counts, missing=missing)

