import pytest

# from jieba.analyse import ChineseAnalyzer
from whoosh import analysis, fields, highlight, qparser, query
from whoosh.analysis import Token
from whoosh.filedb.filestore import RamStorage
from whoosh.util.testing import TempIndex

_doc = "alfa bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"


def u(s):
    return s.decode("ascii") if isinstance(s, bytes) else s


def test_null_fragment():
    terms = frozenset(("bravo", "india"))
    sa = analysis.StandardAnalyzer()
    nf = highlight.WholeFragmenter()
    uc = highlight.UppercaseFormatter()
    htext = highlight.highlight(_doc, terms, sa, nf, uc)
    assert (
        htext
        == "alfa BRAVO charlie delta echo foxtrot golf hotel INDIA juliet kilo lima"
    )


def test_phrase_strict():
    def search(searcher, query_string):
        parser = qparser.QueryParser("title", schema=ix.schema)
        q = parser.parse(u(query_string))
        result = searcher.search(q, terms=True)
        result.fragmenter = highlight.ContextFragmenter()
        result.formatter = highlight.UppercaseFormatter()
        return result

    schema = fields.Schema(id=fields.ID(stored=True), title=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(
        id="1",
        title="strict phrase highlights phrase terms but not individual terms",
    )
    w.commit()

    with ix.searcher() as s:
        # Phrase
        r = search(s, '"phrase terms"')

        # Non-strict
        outputs = [hit.highlights("title", strict_phrase=False) for hit in r]
        assert outputs == [
            "strict PHRASE highlights PHRASE TERMS but not individual...TERMS"
        ]

        # Strict
        outputs = [hit.highlights("title", strict_phrase=True) for hit in r]
        assert outputs == ["phrase highlights PHRASE TERMS but not individual"]

        # Phrase with slop
        r = search(s, '"strict highlights terms"~2')

        # Non-strict
        outputs = [hit.highlights("title", strict_phrase=False) for hit in r]
        assert outputs == [
            "STRICT phrase HIGHLIGHTS phrase TERMS but not individual...TERMS"
        ]

        # Strict
        outputs = [hit.highlights("title", strict_phrase=True) for hit in r]
        assert outputs == ["STRICT phrase HIGHLIGHTS phrase TERMS but not individual"]

        # Phrase with individual terms
        r = search(s, 'individual AND "phrase terms"')

        # Non-strict
        outputs = [hit.highlights("title", strict_phrase=False) for hit in r]
        assert outputs == [
            "strict PHRASE highlights PHRASE TERMS but not INDIVIDUAL TERMS"
        ]

        # Strict
        outputs = [hit.highlights("title", strict_phrase=True) for hit in r]
        assert outputs == ["phrase highlights PHRASE TERMS but not INDIVIDUAL terms"]


def test_sentence_fragment():
    text = (
        "This is the first sentence. This one doesn't have the word. "
        + "This sentence is the second. Third sentence here."
    )
    terms = ("sentence",)
    sa = analysis.StandardAnalyzer(stoplist=None)
    sf = highlight.SentenceFragmenter()
    uc = highlight.UppercaseFormatter()
    htext = highlight.highlight(text, terms, sa, sf, uc)
    assert (
        htext
        == "This is the first SENTENCE...This SENTENCE is the second...Third SENTENCE here"
    )


def test_context_fragment():
    terms = frozenset(("bravo", "india"))
    sa = analysis.StandardAnalyzer()
    cf = highlight.ContextFragmenter(surround=6)
    uc = highlight.UppercaseFormatter()
    htext = highlight.highlight(_doc, terms, sa, cf, uc)
    assert htext == "alfa BRAVO charlie...hotel INDIA juliet"


def test_context_at_start():
    terms = frozenset(["alfa"])
    sa = analysis.StandardAnalyzer()
    cf = highlight.ContextFragmenter(surround=15)
    uc = highlight.UppercaseFormatter()
    htext = highlight.highlight(_doc, terms, sa, cf, uc)
    assert htext == "ALFA bravo charlie delta echo foxtrot"


def test_html_format():
    terms = frozenset(("bravo", "india"))
    sa = analysis.StandardAnalyzer()
    cf = highlight.ContextFragmenter(surround=6)
    hf = highlight.HtmlFormatter()
    htext = highlight.highlight(_doc, terms, sa, cf, hf)
    assert (
        htext
        == 'alfa <strong class="match term0">bravo</strong> charlie...hotel <strong class="match term1">india</strong> juliet'
    )


def test_html_escape():
    terms = frozenset(["bravo"])
    sa = analysis.StandardAnalyzer()
    wf = highlight.WholeFragmenter()
    hf = highlight.HtmlFormatter()
    htext = highlight.highlight('alfa <bravo "charlie"> delta', terms, sa, wf, hf)
    assert (
        htext
        == 'alfa &lt;<strong class="match term0">bravo</strong> "charlie"&gt; delta'
    )


def test_maxclasses():
    terms = frozenset(("alfa", "bravo", "charlie", "delta", "echo"))
    sa = analysis.StandardAnalyzer()
    cf = highlight.ContextFragmenter(surround=6)
    hf = highlight.HtmlFormatter(tagname="b", termclass="t", maxclasses=2)
    htext = highlight.highlight(_doc, terms, sa, cf, hf)
    assert (
        htext
        == '<b class="match t0">alfa</b> <b class="match t1">bravo</b> <b class="match t0">charlie</b>...<b class="match t1">delta</b> <b class="match t0">echo</b> foxtrot'
    )


def test_workflow_easy():
    schema = fields.Schema(id=fields.ID(stored=True), title=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)

    w = ix.writer()
    w.add_document(id="1", title="The man who wasn't there")
    w.add_document(id="2", title="The dog who barked at midnight")
    w.add_document(id="3", title="The invisible man")
    w.add_document(id="4", title="The girl with the dragon tattoo")
    w.add_document(id="5", title="The woman who disappeared")
    w.commit()

    with ix.searcher() as s:
        # Parse the user query
        parser = qparser.QueryParser("title", schema=ix.schema)
        q = parser.parse("man")
        r = s.search(q, terms=True)
        assert len(r) == 2

        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        outputs = [hit.highlights("title") for hit in r]
        assert outputs == ["The invisible MAN", "The MAN who wasn't there"]


def test_workflow_manual():
    schema = fields.Schema(id=fields.ID(stored=True), title=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)

    w = ix.writer()
    w.add_document(id="1", title="The man who wasn't there")
    w.add_document(id="2", title="The dog who barked at midnight")
    w.add_document(id="3", title="The invisible man")
    w.add_document(id="4", title="The girl with the dragon tattoo")
    w.add_document(id="5", title="The woman who disappeared")
    w.commit()

    with ix.searcher() as s:
        # Parse the user query
        parser = qparser.QueryParser("title", schema=ix.schema)
        q = parser.parse("man")

        # Extract the terms the user used in the field we're interested in
        terms = [text for fieldname, text in q.all_terms() if fieldname == "title"]

        # Perform the search
        r = s.search(q)
        assert len(r) == 2

        # Use the same analyzer as the field uses. To be sure, you can
        # do schema[fieldname].analyzer. Be careful not to do this
        # on non-text field types such as DATETIME.
        analyzer = schema["title"].analyzer

        # Since we want to highlight the full title, not extract fragments,
        # we'll use WholeFragmenter.
        nf = highlight.WholeFragmenter()

        # In this example we'll simply uppercase the matched terms
        fmt = highlight.UppercaseFormatter()

        outputs = []
        for d in r:
            text = d["title"]
            outputs.append(highlight.highlight(text, terms, analyzer, nf, fmt))

        assert outputs == ["The invisible MAN", "The MAN who wasn't there"]


def test_unstored():
    schema = fields.Schema(text=fields.TEXT, tags=fields.KEYWORD)
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(text="alfa bravo charlie", tags="delta echo")
    w.commit()

    hit = ix.searcher().search(query.Term("text", "bravo"))[0]
    with pytest.raises(KeyError):
        hit.highlights("tags")


def test_multifilter():
    iwf_for_index = analysis.IntraWordFilter(mergewords=True, mergenums=False)
    iwf_for_query = analysis.IntraWordFilter(mergewords=False, mergenums=False)
    mf = analysis.MultiFilter(index=iwf_for_index, query=iwf_for_query)
    ana = analysis.RegexTokenizer() | mf | analysis.LowercaseFilter()

    schema = fields.Schema(text=fields.TEXT(analyzer=ana, stored=True))
    with TempIndex(schema) as ix:
        w = ix.writer()
        w.add_document(text="Our BabbleTron5000 is great")
        w.commit()

        with ix.searcher() as s:
            assert ("text", "5000") in s.reader()
            hit = s.search(query.Term("text", "5000"))[0]
            assert (
                hit.highlights("text")
                == 'Our BabbleTron<b class="match term0">5000</b> is great'
            )


def test_pinpoint():
    domain = (
        "alfa bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oskar papa quebec romeo sierra tango"
    )
    schema = fields.Schema(text=fields.TEXT(stored=True, chars=True))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(text=domain)
    w.commit()

    assert ix.schema["text"].supports("characters")
    with ix.searcher() as s:
        r = s.search(query.Term("text", "juliet"), terms=True)
        hit = r[0]
        hi = highlight.Highlighter()
        hi.formatter = highlight.UppercaseFormatter()

        assert not hi.can_load_chars(r, "text")
        assert (
            hi.highlight_hit(hit, "text")
            == "golf hotel india JULIET kilo lima mike november"
        )

        hi.fragmenter = highlight.PinpointFragmenter()
        assert hi.can_load_chars(r, "text")
        assert (
            hi.highlight_hit(hit, "text")
            == "ot golf hotel india JULIET kilo lima mike nove"
        )

        hi.fragmenter.autotrim = True
        assert hi.highlight_hit(hit, "text") == "golf hotel india JULIET kilo lima mike"


def test_highlight_wildcards():
    schema = fields.Schema(text=fields.TEXT(stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(text="alfa bravo charlie delta cookie echo")

    with ix.searcher() as s:
        qp = qparser.QueryParser("text", ix.schema)
        q = qp.parse("c*")
        r = s.search(q)
        assert r.scored_length() == 1
        r.formatter = highlight.UppercaseFormatter()
        hit = r[0]
        assert hit.highlights("text") == "alfa bravo CHARLIE delta COOKIE echo"


def test_highlight_ngrams():
    schema = fields.Schema(text=fields.NGRAMWORDS(stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(text="Multiplication and subtraction are good")

    with ix.searcher() as s:
        qp = qparser.QueryParser("text", ix.schema)
        q = qp.parse("multiplication")
        r = s.search(q)
        assert r.scored_length() == 1

        r.fragmenter = highlight.SentenceFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        snippet = r[0].highlights("text")
        assert snippet == "MULTIPLICATIon and subtracTION are good"


def test_issue324():
    sa = analysis.StemmingAnalyzer()
    result = highlight.highlight(
        "Indexed!\n1",
        ["index"],
        sa,
        fragmenter=highlight.ContextFragmenter(),
        formatter=highlight.UppercaseFormatter(),
    )
    assert result == "INDEXED!\n1"


def test_whole_noterms():
    schema = fields.Schema(text=fields.TEXT(stored=True), tag=fields.KEYWORD)
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(text="alfa bravo charlie delta echo foxtrot golf", tag="foo")

    with ix.searcher() as s:
        r = s.search(query.Term("text", "delta"))
        assert len(r) == 1

        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        hi = r[0].highlights("text")
        assert hi == "alfa bravo charlie DELTA echo foxtrot golf"

        r = s.search(query.Term("tag", "foo"))
        assert len(r) == 1
        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        hi = r[0].highlights("text")
        assert hi == ""

        hi = r[0].highlights("text", minscore=0)
        assert hi == "alfa bravo charlie delta echo foxtrot golf"


def test_html_fragment_inline_tags():
    # A term may span adjacent inline tags; the hit is split into one
    # element per text run and the original tags are preserved
    sa = analysis.StandardAnalyzer()
    r = highlight.highlight_html("co<strong>de</strong> &amp; data", ["code"], sa)
    assert r.text == "code & data"
    assert r.matches == [(0, 4)]
    assert r.html == (
        '<mark class="match">co</mark><strong><mark class="match">de</mark>'
        "</strong> &amp; data"
    )


def test_html_fragment_result_str():
    sa = analysis.StandardAnalyzer()
    r = highlight.highlight_html("alfa bravo", ["bravo"], sa)
    assert r.text == "alfa bravo"
    assert r.matches == [(5, 10)]
    assert str(r) == r.html
    assert r == 'alfa <mark class="match">bravo</mark>'


def test_html_fragment_malicious_attributes():
    sa = analysis.StandardAnalyzer()
    html = (
        '<a href="javascript:alert(1)" onclick="steal()">alfa</a> '
        '<b onmouseover="x()">bravo</b>'
    )
    r = highlight.highlight_html(html, ["alfa", "bravo"], sa)
    # Unsafe URL scheme and non-whitelisted attributes are dropped
    assert r.html == (
        '<a><mark class="match">alfa</mark></a> <b><mark class="match">bravo</mark></b>'
    )


def test_html_fragment_attribute_escaping():
    sa = analysis.StandardAnalyzer()
    html = '<a href="https://example.com/?a=1&amp;b=2" title=\'x" y\'>alfa</a>'
    r = highlight.highlight_html(html, ["alfa"], sa)
    # Safe href is kept, values are re-escaped
    assert r.html == (
        '<a href="https://example.com/?a=1&amp;b=2" title="x&quot; y">'
        '<mark class="match">alfa</mark></a>'
    )


def test_html_fragment_double_entity():
    sa = analysis.StandardAnalyzer()
    r = highlight.highlight_html("alfa &amp;amp; bravo", ["bravo"], sa)
    # One decoding pass: the visible text contains the literal "&amp;"
    assert r.text == "alfa &amp; bravo"
    assert r.matches == [(11, 16)]
    assert r.html == 'alfa &amp;amp; <mark class="match">bravo</mark>'


def test_html_fragment_combining_chars():
    sa = analysis.StandardAnalyzer()
    r = highlight.highlight_html("cafe\u0301 au lait", ["cafe"], sa)
    assert r.text == "cafe\u0301 au lait"
    # The match covers the base letters; the combining mark stays outside
    assert r.matches == [(0, 4)]
    assert r.html == '<mark class="match">cafe</mark>\u0301 au lait'


def test_html_fragment_script_style():
    sa = analysis.StandardAnalyzer()
    html = (
        "<script>var alfa = 1;</script>alfa "
        "<style>.alfa { color: red }</style><b>bravo</b>"
    )
    r = highlight.highlight_html(html, ["alfa", "bravo"], sa)
    # script/style contents are excluded from the visible text and output
    assert r.text == "alfa bravo"
    assert r.matches == [(0, 4), (5, 10)]
    assert r.html == (
        '<mark class="match">alfa</mark> <b><mark class="match">bravo</mark></b>'
    )


def test_html_fragment_block_boundary():
    sa = analysis.StandardAnalyzer()
    r = highlight.highlight_html("<p>alfa</p><p>bravo</p>", ["alfa", "bravo"], sa)
    # A newline separates the paragraphs so matches cannot span them
    assert r.text == "alfa\nbravo"
    assert r.matches == [(0, 4), (5, 10)]
    assert r.html == (
        '<p><mark class="match">alfa</mark></p>\n'
        '<p><mark class="match">bravo</mark></p>'
    )
    # "alfabravo" is not a token because of the paragraph boundary
    r = highlight.highlight_html("<p>alfa</p><p>bravo</p>", ["alfabravo"], sa)
    assert r.matches == []
    assert r.html == ""


def test_html_fragment_overlap_merge():
    # Overlapping matches are merged deterministically into one element
    frag = highlight.Fragment(
        "alfa bravo",
        [
            Token(text="alfa", startchar=0, endchar=7),
            Token(text="bravo", startchar=5, endchar=10),
        ],
        0,
        11,
    )
    fmt = highlight.HtmlFragmentFormatter()
    assert fmt.format([frag]) == '<mark class="match">alfa bravo</mark>'


def test_html_fragment_whitelist():
    sa = analysis.StandardAnalyzer()
    # Tags not on the whitelist are dropped but their text is kept
    r = highlight.highlight_html(
        "<marquee>alfa</marquee> <i>bravo</i>",
        ["alfa", "bravo"],
        sa,
        formatter=highlight.HtmlFragmentFormatter(allowed_tags=["i"]),
    )
    assert r.html == (
        '<mark class="match">alfa</mark> <i><mark class="match">bravo</mark></i>'
    )


def test_html_fragment_unclosed_and_escaped():
    sa = analysis.StandardAnalyzer()
    # Unclosed tags are closed at the end of the fragment; stray markup
    # characters in the text are escaped
    r = highlight.highlight_html("<b>1 < 2 & alfa", ["alfa"], sa)
    assert r.text == "1 < 2 & alfa"
    assert r.html == '<b>1 &lt; 2 &amp; <mark class="match">alfa</mark></b>'


def test_html_fragment_context_window():
    sa = analysis.StandardAnalyzer()
    html = "zero one <em>two three four five six</em> seven eight nine ten"
    r = highlight.highlight_html(
        html, ["six"], sa, fragmenter=highlight.ContextFragmenter(surround=10)
    )
    # Tags enclosing the start of the window are re-opened and re-closed
    assert r.html == '<em>four five <mark class="match">six</mark></em> seven eight'


def test_html_fragment_plain_highlight_unaffected():
    # The plain-text highlighting pipeline is unchanged
    terms = frozenset(("bravo", "india"))
    sa = analysis.StandardAnalyzer()
    cf = highlight.ContextFragmenter(surround=6)
    hf = highlight.HtmlFormatter()
    htext = highlight.highlight(_doc, terms, sa, cf, hf)
    assert (
        htext == 'alfa <strong class="match term0">bravo</strong> charlie...'
        'hotel <strong class="match term1">india</strong> juliet'
    )
