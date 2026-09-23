import gzip

from whoosh import analysis, fields, highlight, query, spelling
from whoosh.qparser import QueryParser
from whoosh.support.levenshtein import levenshtein
from whoosh.util.testing import TempIndex


def u(s):
    return s.decode("ascii") if isinstance(s, bytes) else s


_wordlist = sorted(
    """
    render animation animate shader shading zebra koala
    ready kismet reaction page delete quick fox jumped
    over lazy dog wicked erase red team yellow under interest
    open print acrid sear deaf feed grow heal jolly kilt
    low zone xylophone crown vale brown neat meat reduction
    blunder preaction lamppost
    """.split()
)


def test_list_corrector():
    corr = spelling.ListCorrector(_wordlist)
    typo = "reoction"
    sugs = list(corr.suggest(typo, maxdist=2))
    target = []
    for lev_dist in range(1, 3):
        # sugs will return suggest first ordered by levenshtein distance
        # then second order by dictionary order
        target += [
            w for w in _wordlist if levenshtein(typo, w) <= lev_dist and w not in target
        ]
    assert sugs == target


def test_automaton():
    schema = fields.Schema(text=fields.TEXT)
    with TempIndex(schema, "automatonspell") as ix:
        with ix.writer() as w:
            w.add_document(text=" ".join(_wordlist))

        with ix.reader() as r:
            bterms = list(r.lexicon("text"))
            words = [bterm.decode("utf8") for bterm in bterms]
            assert words == _wordlist

            typo = "reoction"
            sugs = list(r.terms_within("text", typo, maxdist=2))
            target = [w for w in _wordlist if levenshtein(typo, w) <= 2]
            assert sugs == target


def test_reader_corrector():
    schema = fields.Schema(text=fields.TEXT())
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(text="render zorro kaori postal")
            w.add_document(text="reader zebra koala pastry")
            w.add_document(text="leader libra oola paster")
            w.add_document(text="feeder lorry zoala baster")

        with ix.reader() as r:
            sp = spelling.ReaderCorrector(r, "text", schema["text"])
            assert sp.suggest("koala", maxdist=1) == ["koala", "zoala"]

            target = ["kaori", "koala", "oola"]
            sugs = sp.suggest("kaola", maxdist=2)
            assert sugs == target


def test_unicode_spelling():
    schema = fields.Schema(text=fields.ID())

    domain = [
        "\u0924\u092a\u093e\u0907\u0939\u0930\u0941",
        "\u65e5\u672c",
        "\uc774\uc124\ud76c",
    ]

    with TempIndex(schema) as ix:
        with ix.writer() as w:
            for word in domain:
                w.add_document(text=word)

        with ix.reader() as r:
            rc = spelling.ReaderCorrector(r, "text", schema["text"])
            assert rc.suggest("\u65e5\u672e\u672c") == ["\u65e5\u672c"]


def test_wordfile():
    import os.path

    path = os.path.join(os.path.dirname(__file__), "english-words.10.gz")
    wordfile = gzip.open(path, "rb")
    words = sorted(line.decode("latin1").strip().lower() for line in wordfile)

    cor = spelling.ListCorrector(words)
    assert cor.suggest("specail") == ["special"]


def test_query_highlight():
    qp = QueryParser("a", None)
    hf = highlight.HtmlFormatter()

    def do(text, terms):
        q = qp.parse(text)
        tks = [tk for tk in q.all_tokens() if tk.text in terms]
        for tk in tks:
            if tk.startchar is None or tk.endchar is None:
                assert False, tk
        fragment = highlight.Fragment(text, tks)
        return hf.format_fragment(fragment)

    assert do("a b c d", ["b"]) == 'a <strong class="match term0">b</strong> c d'
    assert (
        do('a (x:b OR y:"c d") e', ("b", "c"))
        == 'a (x:<strong class="match term0">b</strong> OR y:"<strong class="match term1">c</strong> d") e'
    )


def test_query_terms():
    qp = QueryParser("a", None)

    q = qp.parse("alfa b:(bravo OR c:charlie) delta")
    assert sorted(q.iter_all_terms()) == [
        ("a", "alfa"),
        ("a", "delta"),
        ("b", "bravo"),
        ("c", "charlie"),
    ]

    q = qp.parse("alfa brav*")
    assert sorted(q.iter_all_terms()) == [("a", "alfa")]

    q = qp.parse('a b:("b c" d)^2 e')
    tokens = [(t.fieldname, t.text, t.boost) for t in q.all_tokens()]
    assert tokens == [
        ("a", "a", 1.0),
        ("b", "b", 2.0),
        ("b", "c", 2.0),
        ("b", "d", 2.0),
        ("a", "e", 1.0),
    ]


def test_correct_query():
    schema = fields.Schema(a=fields.TEXT(), b=fields.TEXT)
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(a="alfa bravo charlie delta")
            w.add_document(a="delta echo foxtrot golf")
            w.add_document(a="golf hotel india juliet")
            w.add_document(a="juliet kilo lima mike")

        with ix.searcher() as s:
            qp = QueryParser("a", ix.schema)
            qtext = 'alpha ("brovo november" OR b:dolta) detail'
            q = qp.parse(qtext, ix.schema)

            c = s.correct_query(q, qtext)
            cq = c.query
            assert isinstance(cq, query.And)
            assert cq[0].text == "alfa"
            assert isinstance(cq[1], query.Or)
            assert isinstance(cq[1][0], query.Phrase)
            assert cq[1][0].words == ["bravo", "november"]

            qtext = 'alpha b:("brovo november" a:delta) detail'
            q = qp.parse(qtext, ix.schema)
            c = s.correct_query(q, qtext)
            assert (
                str(c.query)
                == '(a:alfa AND b:"brovo november" AND a:delta AND a:detail)'
            )
            assert c.string == 'alfa b:("brovo november" a:delta) detail'

            hf = highlight.HtmlFormatter(classname="c")
            assert (
                c.format_string(hf)
                == '<strong class="c term0">alfa</strong> b:("brovo november" a:delta) detail'
            )


def test_spelling_field():
    text = "rendering shading modeling reactions"
    ana = analysis.StemmingAnalyzer()
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, spelling=True))

    assert schema["text"].spelling
    assert schema["text"].separate_spelling()

    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(text=text)

        with ix.searcher() as s:
            r = s.reader()
            fieldobj = schema["text"]
            words = [fieldobj.from_bytes(t) for t in r.lexicon("text")]
            assert words == ["model", "reaction", "render", "shade"]

            words = [fieldobj.from_bytes(t) for t in r.lexicon("spell_text")]
            assert words == ["modeling", "reactions", "rendering", "shading"]

            # suggest() automatically looks in the spell_text field because
            # it calls fieldobj.spelling_fieldname() first
            assert s.suggest("text", "renderink") == ["rendering"]

        with ix.writer() as w:
            w.delete_document(0)


def test_correct_spell_field():
    ana = analysis.StemmingAnalyzer()
    schema = fields.Schema(text=fields.TEXT(analyzer=ana, spelling=True))
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(text="rendering shading modeling reactions")

        with ix.searcher() as s:
            text = s.schema["text"]
            spell_text = s.schema["spell_text"]

            r = s.reader()
            words = [text.from_bytes(t) for t in r.lexicon("text")]
            assert words == ["model", "reaction", "render", "shade"]

            words = [spell_text.from_bytes(t) for t in r.lexicon("spell_text")]
            assert words == ["modeling", "reactions", "rendering", "shading"]

            qp = QueryParser("text", s.schema)
            qtext = "renderink"
            q = qp.parse(qtext, s.schema)

            r = s.search(q)
            assert len(r) == 0

            c = s.correct_query(q, qtext)
            assert c.string == "rendering"
            assert c.query == query.Term("text", "rendering")

            hf = highlight.HtmlFormatter(classname="c")
            assert c.format_string(hf) == '<strong class="c term0">rendering</strong>'


def test_suggest_prefix():
    domain = (
        "Shoot To Kill",
        "Bloom, Split and Deviate",
        "Rankle the Seas and the Skies",
        "Lightning Flash Flame Shell",
        "Flower Wind Rage and Flower God Roar, Heavenly Wind Rage and "
        "Heavenly Demon Sneer",
        "All Waves, Rise now and Become my Shield, Lightning, Strike "
        "now and Become my Blade",
        "Cry, Raise Your Head, Rain Without end",
        "Sting All Enemies To Death",
        "Reduce All Creation to Ash",
        "Sit Upon the Frozen Heavens",
        "Call forth the Twilight",
    )

    schema = fields.Schema(
        content=fields.TEXT(
            stored=True,
        ),
        quick=fields.NGRAM(maxsize=10, stored=True),
    )
    with TempIndex(schema, "sugprefix") as ix:
        with ix.writer() as w:
            for item in domain:
                content = u(item)
                w.add_document(content=content, quick=content)

        with ix.searcher() as s:
            sugs = s.suggest("content", "ra", maxdist=2, prefix=2)
            assert sugs == ["rage", "rain"]

            sugs = s.suggest("content", "ra", maxdist=2, prefix=1)
            assert sugs == ["rage", "rain", "roar"]


def test_prefix_address():
    fieldtype = fields.TEXT()
    schema = fields.Schema(f1=fieldtype, f2=fieldtype)
    with TempIndex(schema, "prefixaddr") as ix:
        with ix.writer() as w:
            w.add_document(f1="aabc aawx aaqr aade", f2="aa12 aa34 aa56 aa78")

        with ix.searcher() as s:
            sugs = s.suggest("f1", "aa", maxdist=2, prefix=2)
            assert sorted(sugs) == ["aabc", "aade", "aaqr", "aawx"]

            sugs = s.suggest("f2", "aa", maxdist=2, prefix=2)
            assert sorted(sugs) == ["aa12", "aa34", "aa56", "aa78"]


def test_correct_correct():
    from whoosh import qparser

    schema = fields.Schema(a=fields.TEXT())
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            w.add_document(a="dworska")
            w.add_document(a="swojska")

        with ix.searcher() as s:
            s = ix.searcher()
            qtext = "dworska"

            qp = qparser.QueryParser("a", ix.schema)
            q = qp.parse(qtext, ix.schema)
            c = s.correct_query(q, qtext)

            assert c.string == "dworska"
            string = c.format_string(highlight.UppercaseFormatter())
            assert string == "dworska"


def test_very_long_words():
    import sys

    length = int(sys.getrecursionlimit() * 1.5)

    strings1 = [u(chr(i) * length) for i in range(65, 70)]
    strings2 = [u(chr(i) * length) for i in range(71, 75)]

    ana = analysis.StemmingAnalyzer()
    schema = fields.Schema(
        text=fields.TEXT(
            analyzer=ana,
        )
    )
    with TempIndex(schema) as ix:
        with ix.writer() as w:
            for string in strings1:
                w.add_document(text=string)

        with ix.writer() as w:
            for string in strings2:
                w.add_document(text=string)
            w.optimize = True


_place_words = sorted(
    "new york hotel hotels grand central plaza station inn suites "
    "manhattan brooklyn".split()
)
_place_freqs = {
    "hotel": 20,
    "new": 15,
    "york": 15,
    "grand": 10,
    "central": 10,
    "plaza": 8,
    "manhattan": 7,
    "inn": 6,
    "station": 5,
    "brooklyn": 5,
    "hotels": 4,
    "suites": 3,
}
_place_bigrams = {
    ("new", "york"): 1000,
    ("grand", "central"): 800,
    ("central", "station"): 200,
    ("york", "hotel"): 100,
    ("york", "plaza"): 30,
}


def _place_corrector(**kwargs):
    args = dict(
        wordlist=_place_words, word_freq=_place_freqs,
        bigram_freq=_place_bigrams,
    )
    args.update(kwargs)
    return spelling.ContextualQueryCorrector(**args)


def test_contextual_split_and_spell():
    c = _place_corrector()
    cor = c.correct("newyork hotle")
    assert cor.text == "new york hotel"
    assert cor.exhaustive

    # Every change carries offsets into the original string
    assert len(cor.changes) == 2
    first, second = cor.changes
    assert (first.start, first.end) == (0, 7)
    assert first.original == "newyork"
    assert first.replacement == "new york"
    assert "split" in first.reason
    assert (second.start, second.end) == (8, 13)
    assert second.original == "hotle"
    assert second.replacement == "hotel"
    assert "spelling" in second.reason
    for change in cor.changes:
        assert "newyork hotle"[change.start:change.end] == change.original

    # Alternatives are ranked and include the un-split fallback
    assert cor.alternatives[0].text == "new york hotel"
    alt_texts = [a.text for a in cor.alternatives]
    assert "newyork hotel" in alt_texts


def test_contextual_merge():
    c = _place_corrector()
    cor = c.correct("hot el")
    assert cor.text == "hotel"
    assert len(cor.changes) == 1
    change = cor.changes[0]
    assert change.original == "hot el"
    assert change.replacement == "hotel"
    assert "merged" in change.reason


def test_contextual_quoted_phrase_preserved():
    c = _place_corrector()
    cor = c.correct('"newyork hotle" hotle')
    assert cor.text == '"newyork hotle" hotel'
    assert len(cor.changes) == 1
    assert cor.changes[0].original == "hotle"


def test_contextual_field_and_wildcard_preserved():
    c = _place_corrector()
    cor = c.correct("sku:A* hotle")
    assert cor.text == "sku:A* hotel"

    cor = c.correct("sku:12345 hotle")
    assert cor.text == "sku:12345 hotel"

    cor = c.correct("hotle AND (newyork OR plaza)")
    assert cor.text == "hotel AND (new york OR plaza)"


def test_contextual_reader_dictionary():
    schema = fields.Schema(text=fields.TEXT())
    with TempIndex(schema, "ctxspell") as ix:
        with ix.writer() as w:
            w.add_document(text="new york hotel")
            w.add_document(text="new york plaza")
            w.add_document(text="grand central station")
            w.add_document(text="hotel inn suites")
            w.add_document(text="hotel brooklyn manhattan")

        with ix.reader() as r:
            c = spelling.ContextualQueryCorrector(
                reader=r, fieldname="text", bigram_freq=_place_bigrams
            )
            cor = c.correct("newyork hotle")
            assert cor.text == "new york hotel"
            assert cor.exhaustive

            cor = c.correct("hot el")
            assert cor.text == "hotel"


def test_contextual_limits_mark_not_exhaustive():
    c = _place_corrector()
    assert c.correct("newyork hotle").exhaustive

    c = _place_corrector(max_states=1)
    assert not c.correct("newyork hotle").exhaustive

    c = _place_corrector(max_candidates=1)
    assert not c.correct("newyork hotle").exhaustive


def test_contextual_configurable_costs():
    # A very expensive split effectively disables splitting
    costs = spelling.CorrectionCosts(split=100.0)
    c = _place_corrector(costs=costs)
    assert c.correct("newyork").text == "newyork"

    # A very expensive merge effectively disables merging
    costs = spelling.CorrectionCosts(merge=100.0)
    c = _place_corrector(costs=costs, context_weight=0.0)
    assert c.correct("hot el").text == "hot el"

    # Per-character costs are configurable too
    costs = spelling.CorrectionCosts(replace=0.5)
    c = _place_corrector(costs=costs)
    best_cost, best_word = c.word_candidates("hotle")[0]
    assert best_word == "hotel"
    assert best_cost == 1.0


def test_contextual_tie_break_is_deterministic():
    words = ["hotem", "hotel"]
    c = spelling.ContextualQueryCorrector(
        wordlist=words, oov_cost=3.0,
    )
    cor = c.correct("hotle")
    # Both candidates have the same cost and context; ties break by text
    assert cor.alternatives[0].text == "hotel"
    assert cor.alternatives[1].text == "hotem"
    assert cor.alternatives[0].score == cor.alternatives[1].score


def test_contextual_no_split_merge_matches_word_corrector():
    # With split and merge disabled, the new interface preserves the
    # existing per-word suggestion behavior
    c = _place_corrector(allow_split=False, allow_merge=False)

    cor = c.correct("newyork hotle")
    assert cor.text == "newyork hotel"

    # Per-word candidates match the classic ListCorrector suggestions
    classic = spelling.ListCorrector(_place_words)
    assert [w for _, w in c.word_candidates("hotle")] == classic.suggest(
        "hotle", maxdist=2
    )

    schema = fields.Schema(text=fields.TEXT())
    with TempIndex(schema, "ctxcompat") as ix:
        with ix.writer() as w:
            w.add_document(text="new york hotel")
            w.add_document(text="grand central station")
            w.add_document(text="hotel inn suites")

        with ix.reader() as r:
            rc = spelling.ReaderCorrector(r, "text", schema["text"])
            c = spelling.ContextualQueryCorrector(
                reader=r, fieldname="text",
                allow_split=False, allow_merge=False,
            )
            assert [w for _, w in c.word_candidates("hotle")] == rc.suggest(
                "hotle", maxdist=2
            )


# def test_add_spelling():
#     schema = fields.Schema(text1=fields.TEXT, text2=fields.TEXT)
#     ix = RamStorage().create_index(schema)
#     w = ix.writer()
#     w.add_document(text1=u"render zorro kaori postal", text2=u"alfa")
#     w.add_document(text1=u"reader zebra koala pastry", text2=u"alpa")
#     w.add_document(text1=u"leader libra ooala paster", text2=u"alpha")
#     w.add_document(text1=u"feeder lorry zoala baster", text2=u"olfo")
#     w.commit()
#
#     with ix.reader() as r:
#         assert not r.has_word_graph("text1")
#         assert not r.has_word_graph("text2")
#
#     from whoosh.writing import add_spelling
#     add_spelling(ix, ["text1", "text2"])
#
#     with ix.reader() as r:
#         assert r.has_word_graph("text1")
#         assert r.has_word_graph("text2")
#
#         sp = spelling.ReaderCorrector(r, "text1")
#         assert sp.suggest(u"kaola", maxdist=1) == [u'koala']
#         assert sp.suggest(u"kaola", maxdist=2) == [u'koala', u'kaori', u'ooala', u'zoala']
#
#         sp = spelling.ReaderCorrector(r, "text2")
#         assert sp.suggest(u"alfo", maxdist=1) == [u"alfa", u"olfo"]


# def test_multicorrector():
#     schema = fields.Schema(text=fields.TEXT())
#     ix = RamStorage().create_index(schema)
#     domain = u"special specious spectacular spongy spring specials".split()
#     for word in domain:
#         w = ix.writer()
#         w.add_document(text=word)
#         w.commit(merge=False)
#
#     c1 = ix.reader().corrector("text")
#
#     wordlist = sorted(u"bear bare beer sprung".split())
#     c2 = words_to_corrector(wordlist)
#
#     mc = spelling.MultiCorrector([c1, c2])
#     assert mc.suggest("specail") == ["special", "specials"]
#     assert mc.suggest("beur") == ["bear", "beer"]
#     assert mc.suggest("sprang") == ["sprung", "spring"]
