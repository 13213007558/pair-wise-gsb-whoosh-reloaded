# Copyright 2008 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""The highlight module contains classes and functions for displaying short
excerpts from hit documents in the search results you present to the user, with
query terms highlighted.

The highlighting system has four main elements.

* **Fragmenters** chop up the original text into __fragments__, based on the
  locations of matched terms in the text.

* **Scorers** assign a score to each fragment, allowing the system to rank the
  best fragments by whatever criterion.

* **Order functions** control in what order the top-scoring fragments are
  presented to the user. For example, you can show the fragments in the order
  they appear in the document (FIRST) or show higher-scoring fragments first
  (SCORE)

* **Formatters** turn the fragment objects into human-readable output, such as
  an HTML string.

See :doc:`/highlight` for more information.
"""

import re
from collections import deque
from heapq import nlargest
from html import escape as htmlescape
from html import unescape as htmlunescape
from html.parser import HTMLParser
from itertools import groupby

from whoosh.analysis import StandardAnalyzer, Token

# The default value for the maximum chars to examine when fragmenting
DEFAULT_CHARLIMIT = 2**15


# Fragment object


def mkfrag(text, tokens, startchar=None, endchar=None, charsbefore=0, charsafter=0):
    """Returns a :class:`Fragment` object based on the :class:`analysis.Token`
    objects in ``tokens`.
    """

    if startchar is None:
        startchar = tokens[0].startchar if tokens else 0
    if endchar is None:
        endchar = tokens[-1].endchar if tokens else len(text)

    startchar = max(0, startchar - charsbefore)
    endchar = min(len(text), endchar + charsafter)

    return Fragment(text, tokens, startchar, endchar)


class Fragment:
    """Represents a fragment (extract) from a hit document. This object is
    mainly used to keep track of the start and end points of the fragment and
    the "matched" character ranges inside; it does not contain the text of the
    fragment or do much else.

    The useful attributes are:

    ``Fragment.text``
        The entire original text from which this fragment is taken.

    ``Fragment.matches``
        An ordered list of objects representing the matched terms in the
        fragment. These objects have ``startchar`` and ``endchar`` attributes.

    ``Fragment.startchar``
        The index of the first character in the fragment.

    ``Fragment.endchar``
        The index of the last character in the fragment.

    ``Fragment.matched_terms``
        A ``set`` of the ``text`` of the matched terms in the fragment (if
        available).
    """

    def __init__(self, text, matches, startchar=0, endchar=-1):
        """
        :param text: the source text of the fragment.
        :param matches: a list of objects which have ``startchar`` and
            ``endchar`` attributes, and optionally a ``text`` attribute.
        :param startchar: the index into ``text`` at which the fragment starts.
            The default is 0.
        :param endchar: the index into ``text`` at which the fragment ends.
            The default is -1, which is interpreted as the length of ``text``.
        """

        self.text = text
        self.matches = matches

        if endchar == -1:
            endchar = len(text)
        self.startchar = startchar
        self.endchar = endchar

        self.matched_terms = set()
        for t in matches:
            if hasattr(t, "text"):
                self.matched_terms.add(t.text)

    def __repr__(self):
        return "<Fragment %d:%d has %d matches>" % (
            self.startchar,
            self.endchar,
            len(self.matches),
        )

    def __len__(self):
        return self.endchar - self.startchar

    def overlaps(self, fragment):
        sc = self.startchar
        ec = self.endchar
        fsc = fragment.startchar
        fec = fragment.endchar
        return (sc < fsc < ec) or (sc < fec < ec)

    def overlapped_length(self, fragment):
        sc = self.startchar
        ec = self.endchar
        fsc = fragment.startchar
        fec = fragment.endchar
        return max(ec, fec) - min(sc, fsc)

    def __lt__(self, other):
        return self.startchar < other.startchar


# Tokenizing


def set_matched_filter(tokens, termset):
    """
    Mark tokens to be highlighted as matched.
    Phrase agnostic: highlights all matching tokens individually,
                     even if the terms are part of a phrase

    :param tokens: Result tokens to scan for matched terms to highlight
    :param termset: Query terms
    :return: yield each token with t.matched = True / False, indicating if the
             token should be highlighted
    """
    for t in tokens:
        t.matched = t.text in termset
        yield t


def set_matched_filter_phrases(tokens, text, terms, phrases):
    """
    Mark tokens to be highlighted as matched. Used for Strict Phrase highlighting.
    Phrase-aware: highlights only individual matches for individual query terms
                  and phrase matches for phrase terms.

    :param tokens:  Result tokens
    :param text:    Result text to scan for matched terms to highlight
    :param terms:   Individual query terms
    :param phrases: Query Phrases
    :return: yield each token with t.matched = True / False, indicating if the
             token should be highlighted
    """

    """
    Implementation note: Because the Token object follows a Singleton pattern,
    we can only read each one once. Because phrase matching requires rescanning,
    we require a rendered token list (the text parameter) instead. The function must
    still yield Token objects at the end, so the text list is used as a way to build a list
    of Token indices (the matches set). The yield loop at the end uses this
    to properly set .matched on the yielded Token objects.
    """
    text = text.split()
    matches = set()

    # Match phrases
    for phrase in phrases:
        i = 0
        n_phrase_words = len(phrase.words)
        slop = phrase.slop
        while i < len(text):
            if phrase.words[0] == text[i]:  # If first word matched
                if slop == 1:
                    # Simple substring match
                    if (
                        text[i + 1 : i + n_phrase_words] == phrase.words[1:]
                    ):  # If rest of phrase matches
                        any(
                            map(matches.add, range(i, i + n_phrase_words))
                        )  # Collect matching indices
                        # Advance past match area.
                        # Choosing to ignore possible overlapping matches for efficiency due to low probability.
                        i += n_phrase_words
                    else:
                        i += 1
                else:
                    # Slop match
                    current_word_index = first_slop_match = last_slop_match = i
                    slop_matches = [first_slop_match]
                    for word in phrase.words[1:]:
                        try:
                            """
                            Find the *last* occurrence of word in the slop substring by reversing it and mapping the index back.
                            If multiple tokens match in the substring, picking the first one can overlook valid matches.
                            For example, phrase is: 'one two three'~2
                            Target substring is:    'one two two six three', which is a valid match.
                                                     [0] [1] [2] [3] [4]

                            Looking for the first match will find [0], then [1] then fail since [3] is more than ~2 words away
                            Looking for the last match will find [0], then, given a choice between [1] or [2], will pick [2],
                            making [4] visible from there
                            """
                            text_sub = text[
                                current_word_index + 1 : current_word_index + 1 + slop
                            ][::-1]  # Substring to scan (reversed)
                            len_sub = len(text_sub)
                            next_word_index = (
                                len_sub - text_sub.index(word) - 1
                            )  # Map index back to unreversed list
                            last_slop_match = current_word_index + next_word_index + 1
                            slop_matches.append(last_slop_match)
                            current_word_index = last_slop_match
                        except ValueError:
                            # word not found in substring
                            i += 1
                            break
                    else:
                        i = last_slop_match
                        any(map(matches.add, slop_matches))  # Collect matching indices
            else:
                i += 1

    # Match individual terms
    for i, word in enumerate(text):
        for term in terms:
            if term.text == word:
                matches.add(i)
                break

    for i, t in enumerate(tokens):
        t.matched = i in matches
        yield t


# Fragmenters


class Fragmenter:
    def must_retokenize(self):
        """Returns True if this fragmenter requires retokenized text.

        If this method returns True, the fragmenter's ``fragment_tokens``
        method  will be called with an iterator of ALL tokens from the text,
        with the tokens for matched terms having the ``matched`` attribute set
        to True.

        If this method returns False, the fragmenter's ``fragment_matches``
        method will be called with a LIST of matching tokens.
        """

        return True

    def fragment_tokens(self, text, all_tokens):
        """Yields :class:`Fragment` objects based on the tokenized text.

        :param text: the string being highlighted.
        :param all_tokens: an iterator of :class:`analysis.Token`
            objects from the string.
        """

        raise NotImplementedError

    def fragment_matches(self, text, matched_tokens):
        """Yields :class:`Fragment` objects based on the text and the matched
        terms.

        :param text: the string being highlighted.
        :param matched_tokens: a list of :class:`analysis.Token` objects
            representing the term matches in the string.
        """

        raise NotImplementedError


class WholeFragmenter(Fragmenter):
    """Doesn't fragment the token stream. This object just returns the entire
    entire stream as one "fragment". This is useful if you want to highlight
    the entire text.

    Note that even if you use the `WholeFragmenter`, the highlight code will
    return no fragment if no terms matched in the given field. To return the
    whole fragment even in that case, call `highlights()` with `minscore=0`::

        # Query where no terms match in the "text" field
        q = query.Term("tag", "new")

        r = mysearcher.search(q)
        r.fragmenter = highlight.WholeFragmenter()
        r.formatter = highlight.UppercaseFormatter()
        # Since no terms in the "text" field matched, we get no fragments back
        assert r[0].highlights("text") == ""

        # If we lower the minimum score to 0, we get a fragment even though it
        # has no matching terms
        assert r[0].highlights("text", minscore=0) == "This is the text field."

    """

    def __init__(self, charlimit=DEFAULT_CHARLIMIT):
        self.charlimit = charlimit

    def fragment_tokens(self, text, tokens):
        charlimit = self.charlimit
        matches = []
        for t in tokens:
            if charlimit and t.endchar > charlimit:
                break
            if t.matched:
                matches.append(t.copy())
        return [Fragment(text, matches)]


# Backwards compatiblity
NullFragmeter = WholeFragmenter


class SentenceFragmenter(Fragmenter):
    """Breaks the text up on sentence end punctuation characters
    (".", "!", or "?"). This object works by looking in the original text for a
    sentence end as the next character after each token's 'endchar'.

    When highlighting with this fragmenter, you should use an analyzer that
    does NOT remove stop words, for example::

        sa = StandardAnalyzer(stoplist=None)
    """

    def __init__(self, maxchars=200, sentencechars=".!?", charlimit=DEFAULT_CHARLIMIT):
        """
        :param maxchars: The maximum number of characters allowed in a
            fragment.
        """

        self.maxchars = maxchars
        self.sentencechars = frozenset(sentencechars)
        self.charlimit = charlimit

    def fragment_tokens(self, text, tokens):
        maxchars = self.maxchars
        sentencechars = self.sentencechars
        charlimit = self.charlimit

        textlen = len(text)
        # startchar of first token in the current sentence
        first = None
        # Buffer for matched tokens in the current sentence
        tks = []
        endchar = None
        # Number of chars in the current sentence
        currentlen = 0

        for t in tokens:
            startchar = t.startchar
            endchar = t.endchar
            if charlimit and endchar > charlimit:
                break

            if first is None:
                # Remember the startchar of the first token in a sentence
                first = startchar
                currentlen = 0

            tlength = endchar - startchar
            currentlen += tlength

            if t.matched:
                tks.append(t.copy())

            # If the character after the current token is end-of-sentence
            # punctuation, finish the sentence and reset
            if endchar < textlen and text[endchar] in sentencechars:
                # Don't break for two periods in a row (e.g. ignore "...")
                if endchar + 1 < textlen and text[endchar + 1] in sentencechars:
                    continue

                # If the sentence had matches and it's not too long, yield it
                # as a token
                if tks and currentlen <= maxchars:
                    yield mkfrag(text, tks, startchar=first, endchar=endchar)
                # Reset the counts
                tks = []
                first = None
                currentlen = 0

        # If we get to the end of the text and there's still a sentence
        # in the buffer, yield it
        if tks:
            yield mkfrag(text, tks, startchar=first, endchar=endchar)


class ContextFragmenter(Fragmenter):
    """Looks for matched terms and aggregates them with their surrounding
    context.
    """

    def __init__(self, maxchars=200, surround=20, charlimit=DEFAULT_CHARLIMIT):
        """
        :param maxchars: The maximum number of characters allowed in a
            fragment.
        :param surround: The number of extra characters of context to add both
            before the first matched term and after the last matched term.
        """

        self.maxchars = maxchars
        self.surround = surround
        self.charlimit = charlimit

    def fragment_tokens(self, text, tokens):
        maxchars = self.maxchars
        surround = self.surround
        charlimit = self.charlimit

        # startchar of the first token in the fragment
        first = None
        # Stack of startchars
        firsts = deque()
        # Each time we see a matched token, we reset the countdown to finishing
        # the fragment. This also indicates whether we're currently inside a
        # fragment (< 0 not in fragment, >= 0 in fragment)
        countdown = -1
        # Tokens in current fragment
        tks = []
        endchar = None
        # Number of chars in the current fragment
        currentlen = 0

        for t in tokens:
            startchar = t.startchar
            endchar = t.endchar
            tlength = endchar - startchar
            if charlimit and endchar > charlimit:
                break

            if countdown < 0 and not t.matched:
                # We're not in a fragment currently, so just maintain the
                # "charsbefore" buffer
                firsts.append(startchar)
                while firsts and endchar - firsts[0] > surround:
                    firsts.popleft()
            elif currentlen + tlength > maxchars:
                # We're in a fragment, but adding this token would put us past
                # the maximum size. Zero the countdown so the code below will
                # cause the fragment to be emitted
                countdown = 0
            elif t.matched:
                # Start/restart the countdown
                countdown = surround
                # Remember the first char of this fragment
                if first is None:
                    if firsts:
                        first = firsts[0]
                    else:
                        first = startchar
                        # Add on unused front context
                        countdown += surround
                tks.append(t.copy())

            # If we're in a fragment...
            if countdown >= 0:
                # Update the counts
                currentlen += tlength
                countdown -= tlength

                # If the countdown is expired
                if countdown <= 0:
                    # Finish the fragment
                    yield mkfrag(text, tks, startchar=first, endchar=endchar)
                    # Reset the counts
                    tks = []
                    firsts = deque()
                    first = None
                    currentlen = 0

        # If there's a fragment left over at the end, yield it
        if tks:
            yield mkfrag(text, tks, startchar=first, endchar=endchar)


class PinpointFragmenter(Fragmenter):
    """This is a NON-RETOKENIZING fragmenter. It builds fragments from the
    positions of the matched terms.
    """

    def __init__(
        self, maxchars=200, surround=20, autotrim=False, charlimit=DEFAULT_CHARLIMIT
    ):
        """
        :param maxchars: The maximum number of characters allowed in a
            fragment.
        :param surround: The number of extra characters of context to add both
            before the first matched term and after the last matched term.
        :param autotrim: automatically trims text before the first space and
            after the last space in the fragments, to try to avoid truncated
            words at the start and end. For short fragments or fragments with
            long runs between spaces this may give strange results.
        """

        self.maxchars = maxchars
        self.surround = surround
        self.autotrim = autotrim
        self.charlimit = charlimit

    def must_retokenize(self):
        return False

    def fragment_tokens(self, text, tokens):
        matched = [t for t in tokens if t.matched]
        return self.fragment_matches(text, matched)

    @staticmethod
    def _autotrim(fragment):
        text = fragment.text
        startchar = fragment.startchar
        endchar = fragment.endchar

        firstspace = text.find(" ", startchar, endchar)
        if firstspace > 0:
            startchar = firstspace + 1
        lastspace = text.rfind(" ", startchar, endchar)
        if lastspace > 0:
            endchar = lastspace

        if fragment.matches:
            startchar = min(startchar, fragment.matches[0].startchar)
            endchar = max(endchar, fragment.matches[-1].endchar)

        fragment.startchar = startchar
        fragment.endchar = endchar

    def fragment_matches(self, text, tokens):
        maxchars = self.maxchars
        surround = self.surround
        autotrim = self.autotrim
        charlimit = self.charlimit

        j = -1

        for i, t in enumerate(tokens):
            if j >= i:
                continue
            j = i
            left = t.startchar
            right = t.endchar
            if charlimit and right > charlimit:
                break

            currentlen = right - left
            while j < len(tokens) - 1 and currentlen < maxchars:
                next = tokens[j + 1]
                ec = next.endchar
                if ec - right <= surround and ec - left <= maxchars:
                    j += 1
                    right = ec
                    currentlen += ec - next.startchar
                else:
                    break

            left = max(0, left - surround)
            right = min(len(text), right + surround)
            fragment = Fragment(text, tokens[i : j + 1], left, right)
            if autotrim:
                self._autotrim(fragment)
            yield fragment


# Fragment scorers


class FragmentScorer:
    pass


class BasicFragmentScorer(FragmentScorer):
    def __call__(self, f):
        # Add up the boosts for the matched terms in this passage
        score = sum(t.boost for t in f.matches)

        # Favor diversity: multiply score by the number of separate
        # terms matched
        score *= (len(f.matched_terms) * 100) or 1

        return score


# Fragment sorters


def SCORE(fragment):
    "Sorts higher scored passages first."
    return 1


def FIRST(fragment):
    "Sorts passages from earlier in the document first."
    return fragment.startchar


def LONGER(fragment):
    "Sorts longer passages first."
    return 0 - len(fragment)


def SHORTER(fragment):
    "Sort shorter passages first."
    return len(fragment)


# Formatters


def get_text(original, token, replace):
    """Convenience function for getting the text to use for a match when
    formatting.

    If ``replace`` is False, returns the part of ``original`` between
    ``token.startchar`` and ``token.endchar``. If ``replace`` is True, returns
    ``token.text``.
    """

    if replace:
        return token.text
    else:
        return original[token.startchar : token.endchar]


class Formatter:
    """Base class for formatters.

    For highlighters that return strings, it is usually only necessary to
    override :meth:`Formatter.format_token`.

    Use the :func:`get_text` function as a convenience to get the token text::

        class MyFormatter(Formatter):
            def format_token(text, token, replace=False):
                ttext = get_text(text, token, replace)
                return "[%s]" % ttext
    """

    between = "..."

    def _text(self, text):
        return text

    def format_token(self, text, token, replace=False):
        """Returns a formatted version of the given "token" object, which
        should have at least ``startchar`` and ``endchar`` attributes, and
        a ``text`` attribute if ``replace`` is True.

        :param text: the original fragment text being highlighted.
        :param token: an object having ``startchar`` and ``endchar`` attributes
            and optionally a ``text`` attribute (if ``replace`` is True).
        :param replace: if True, the original text between the token's
            ``startchar`` and ``endchar`` indices will be replaced with the
            value of the token's ``text`` attribute.
        """

        raise NotImplementedError

    def format_fragment(self, fragment, replace=False):
        """Returns a formatted version of the given text, using the "token"
        objects in the given :class:`Fragment`.

        :param fragment: a :class:`Fragment` object representing a list of
            matches in the text.
        :param replace: if True, the original text corresponding to each
            match will be replaced with the value of the token object's
            ``text`` attribute.
        """

        output = []
        index = fragment.startchar
        text = fragment.text

        # For overlapping tokens (such as in Chinese), sort by position,
        # then by inverse of length.
        # Because the formatter is sequential, it will only pick the first
        # token for a given position to highlight. This makes sure it picks
        # the longest overlapping token.
        for t in sorted(
            fragment.matches,
            key=lambda token: (token.startchar, -(token.endchar - token.startchar)),
        ):
            if t.startchar is None:
                continue
            if t.startchar < index:
                continue
            if t.startchar > index:
                output.append(self._text(text[index : t.startchar]))
            output.append(self.format_token(text, t, replace))
            index = t.endchar
        output.append(self._text(text[index : fragment.endchar]))

        out_string = "".join(output)
        return out_string

    def format(self, fragments, replace=False):
        """Returns a formatted version of the given text, using a list of
        :class:`Fragment` objects.
        """

        formatted = [self.format_fragment(f, replace=replace) for f in fragments]
        return self.between.join(formatted)

    def __call__(self, text, fragments):
        # For backwards compatibility
        return self.format(fragments)


class NullFormatter(Formatter):
    """Formatter that does not modify the string."""

    def format_token(self, text, token, replace=False):
        return get_text(text, token, replace)


class UppercaseFormatter(Formatter):
    """Returns a string in which the matched terms are in UPPERCASE."""

    def __init__(self, between="..."):
        """
        :param between: the text to add between fragments.
        """

        self.between = between

    def format_token(self, text, token, replace=False):
        ttxt = get_text(text, token, replace)
        return ttxt.upper()


class HtmlFormatter(Formatter):
    """Returns a string containing HTML formatting around the matched terms.

    This formatter wraps matched terms in an HTML element with two class names.
    The first class name (set with the constructor argument ``classname``) is
    the same for each match. The second class name (set with the constructor
    argument ``termclass`` is different depending on which term matched. This
    allows you to give different formatting (for example, different background
    colors) to the different terms in the excerpt.

    >>> hf = HtmlFormatter(tagname="span", classname="match", termclass="term")
    >>> hf(mytext, myfragments)
    "The <span class="match term0">template</span> <span class="match term1">geometry</span> is..."

    This object maintains a dictionary mapping terms to HTML class names (e.g.
    ``term0`` and ``term1`` above), so that multiple excerpts will use the same
    class for the same term. If you want to re-use the same HtmlFormatter
    object with different searches, you should call HtmlFormatter.clear()
    between searches to clear the mapping.
    """

    template = "<%(tag)s class=%(q)s%(cls)s%(tn)s%(q)s>%(t)s</%(tag)s>"

    def __init__(
        self,
        tagname="strong",
        between="...",
        classname="match",
        termclass="term",
        maxclasses=5,
        attrquote='"',
    ):
        """
        :param tagname: the tag to wrap around matching terms.
        :param between: the text to add between fragments.
        :param classname: the class name to add to the elements wrapped around
            matching terms.
        :param termclass: the class name prefix for the second class which is
            different for each matched term.
        :param maxclasses: the maximum number of term classes to produce. This
            limits the number of classes you have to define in CSS by recycling
            term class names. For example, if you set maxclasses to 3 and have
            5 terms, the 5 terms will use the CSS classes ``term0``, ``term1``,
            ``term2``, ``term0``, ``term1``.
        """

        self.between = between
        self.tagname = tagname
        self.classname = classname
        self.termclass = termclass
        self.attrquote = attrquote
        self.maxclasses = maxclasses
        self.seen = {}
        self.htmlclass = " ".join((self.classname, self.termclass))

    def _text(self, text):
        return htmlescape(text, quote=False)

    def format_token(self, text, token, replace=False):
        seen = self.seen
        ttext = self._text(get_text(text, token, replace))
        if ttext in seen:
            termnum = seen[ttext]
        else:
            termnum = len(seen) % self.maxclasses
            seen[ttext] = termnum

        return self.template % {
            "tag": self.tagname,
            "q": self.attrquote,
            "cls": self.htmlclass,
            "t": ttext,
            "tn": termnum,
        }

    def clean(self):
        """Clears the dictionary mapping terms to HTML classnames."""
        self.seen = {}


class GenshiFormatter(Formatter):
    """Returns a Genshi event stream containing HTML formatting around the
    matched terms.
    """

    def __init__(self, qname="strong", between="..."):
        """
        :param qname: the QName for the tag to wrap around matched terms.
        :param between: the text to add between fragments.
        """

        self.qname = qname
        self.between = between

        from genshi.core import (  # type: ignore @UnresolvedImport  # type: ignore @UnresolvedImport
            END,
            START,
            TEXT,
            Attrs,
            Stream,
        )

        self.START, self.END, self.TEXT = START, END, TEXT
        self.Attrs, self.Stream = Attrs, Stream

    def _add_text(self, text, output):
        if output and output[-1][0] == self.TEXT:
            output[-1] = (self.TEXT, output[-1][1] + text, output[-1][2])
        else:
            output.append((self.TEXT, text, (None, -1, -1)))

    def format_token(self, text, token, replace=False):
        qn = self.qname
        txt = get_text(text, token, replace)
        return self.Stream(
            [
                (self.START, (qn, self.Attrs()), (None, -1, -1)),
                (self.TEXT, txt, (None, -1, -1)),
                (self.END, qn, (None, -1, -1)),
            ]
        )

    def format_fragment(self, fragment, replace=False):
        output = []
        index = fragment.startchar
        text = fragment.text

        for t in fragment.matches:
            if t.startchar > index:
                self._add_text(text[index : t.startchar], output)
            output.append((text, t, replace))
            index = t.endchar
        if index < len(text):
            self._add_text(text[index:], output)
        return self.Stream(output)

    def format(self, fragments, replace=False):
        output = []
        first = True
        for fragment in fragments:
            if not first:
                self._add_text(self.between, output)
            output += self.format_fragment(fragment, replace=replace)
            first = False
        return self.Stream(output)


# Highlighting


def top_fragments(fragments, count, scorer, order, minscore=1):
    scored_fragments = ((scorer(f), f) for f in fragments)
    scored_fragments = nlargest(count, scored_fragments)
    best_fragments = [sf for score, sf in scored_fragments if score >= minscore]
    best_fragments.sort(key=order)
    return best_fragments


def highlight(
    text,
    terms,
    analyzer,
    fragmenter,
    formatter,
    top=3,
    scorer=None,
    minscore=1,
    order=FIRST,
    mode="query",
):
    if scorer is None:
        scorer = BasicFragmentScorer()

    if type(fragmenter) is type:
        fragmenter = fragmenter()
    if type(formatter) is type:
        formatter = formatter()
    if type(scorer) is type:
        scorer = scorer()

    if scorer is None:
        scorer = BasicFragmentScorer()

    termset = frozenset(terms)
    tokens = analyzer(text, chars=True, mode=mode, removestops=False)
    tokens = set_matched_filter(tokens, termset)
    fragments = fragmenter.fragment_tokens(text, tokens)
    fragments = top_fragments(fragments, top, scorer, order, minscore)
    return formatter(text, fragments)


class Highlighter:
    def __init__(
        self,
        fragmenter=None,
        scorer=None,
        formatter=None,
        always_retokenize=False,
        order=FIRST,
    ):
        self.fragmenter = fragmenter or ContextFragmenter()
        self.scorer = scorer or BasicFragmentScorer()
        self.formatter = formatter or HtmlFormatter(tagname="b")
        self.order = order
        self.always_retokenize = always_retokenize

    def can_load_chars(self, results, fieldname):
        # Is it possible to build a mapping between the matched terms/docs and
        # their start and end chars for "pinpoint" highlighting (ie not require
        # re-tokenizing text)?

        if self.always_retokenize:
            # No, we've been configured to always retokenize some text
            return False
        if not results.has_matched_terms():
            # No, we don't know what the matched terms are yet
            return False
        if self.fragmenter.must_retokenize():
            # No, the configured fragmenter doesn't support it
            return False

        # Maybe, if the field was configured to store characters
        field = results.searcher.schema[fieldname]
        return field.supports("characters")

    @staticmethod
    def _load_chars(results, fieldname, texts, to_bytes):
        # For each docnum, create a mapping of text -> [(startchar, endchar)]
        # for the matched terms

        results._char_cache[fieldname] = cache = {}
        sorted_ids = sorted(docnum for _, docnum in results.top_n)

        for docnum in sorted_ids:
            cache[docnum] = {}

        for text in texts:
            btext = to_bytes(text)
            m = results.searcher.postings(fieldname, btext)
            docset = set(results.termdocs[(fieldname, btext)])
            for docnum in sorted_ids:
                if docnum in docset:
                    m.skip_to(docnum)
                    assert m.id() == docnum
                    cache[docnum][text] = m.value_as("characters")

    @staticmethod
    def _merge_matched_tokens(tokens):
        # Merges consecutive matched tokens together, so they are highlighted
        # as one

        token = None

        for t in tokens:
            if not t.matched:
                if token is not None:
                    yield token
                    token = None
                yield t
                continue

            if token is None:
                token = t.copy()
            elif t.startchar <= token.endchar:
                if t.endchar > token.endchar:
                    token.text += t.text[token.endchar - t.endchar :]
                    token.endchar = t.endchar
            else:
                yield token
                token = None
                # t was not merged, also has to be yielded
                yield t

        if token is not None:
            yield token

    def highlight_hit(
        self, hitobj, fieldname, text=None, top=3, minscore=1, strict_phrase=False
    ):
        results = hitobj.results
        schema = results.searcher.schema
        field = schema[fieldname]
        to_bytes = field.to_bytes
        from_bytes = field.from_bytes

        if text is None:
            if fieldname not in hitobj:
                raise KeyError(f"Field {fieldname!r} is not stored.")
            text = hitobj[fieldname]

        # Get the terms searched for/matched in this field
        if results.has_matched_terms():
            bterms = (term for term in results.matched_terms() if term[0] == fieldname)
        else:
            bterms = results.query_terms(expand=True, fieldname=fieldname)
        # Convert bytes to unicode
        words = frozenset(from_bytes(term[1]) for term in bterms)

        # If we can do "pinpoint" highlighting...
        if self.can_load_chars(results, fieldname):
            # Build the docnum->[(startchar, endchar),] map
            if fieldname not in results._char_cache:
                self._load_chars(results, fieldname, words, to_bytes)

            hitterms = (
                from_bytes(term[1])
                for term in hitobj.matched_terms()
                if term[0] == fieldname
            )

            # Grab the word->[(startchar, endchar)] map for this docnum
            cmap = results._char_cache[fieldname][hitobj.docnum]
            # A list of Token objects for matched words
            tokens = []
            charlimit = self.fragmenter.charlimit
            for word in hitterms:
                chars = cmap[word]
                for pos, startchar, endchar in chars:
                    if charlimit and endchar > charlimit:
                        break
                    tokens.append(
                        Token(text=word, pos=pos, startchar=startchar, endchar=endchar)
                    )
            tokens.sort(key=lambda t: t.startchar)
            tokens = [
                max(group, key=lambda t: t.endchar - t.startchar)
                for key, group in groupby(tokens, lambda t: t.startchar)
            ]
            fragments = self.fragmenter.fragment_matches(text, tokens)
        else:
            # Retokenize the text
            analyzer = results.searcher.schema[fieldname].analyzer
            tokens = analyzer(
                text, positions=True, chars=True, mode="index", removestops=False
            )

            # Set Token.matched attribute for tokens that match a query term
            if strict_phrase:
                terms, phrases = results.q.phrases()
                tokens = set_matched_filter_phrases(tokens, text, terms, phrases)
            else:
                tokens = set_matched_filter(tokens, words)
            tokens = self._merge_matched_tokens(tokens)
            fragments = self.fragmenter.fragment_tokens(text, tokens)

        fragments = top_fragments(
            fragments, top, self.scorer, self.order, minscore=minscore
        )
        output = self.formatter.format(fragments)
        return output


# HTML fragment highlighting
#
# The functions and classes below implement an "HTML fragment" mode for
# highlighting. Instead of treating the stored value as plain text (which
# would break tags and entities such as "&amp;" when inserting highlight
# markup at raw character offsets), the fragment is parsed into events. Only
# the visible text nodes are analyzed; hit positions computed on the visible
# text are mapped back onto the original event stream, and the output is
# rebuilt from scratch using a caller-supplied whitelist of allowed tags and
# attributes. The input is never trusted or echoed back verbatim.

# Tags whose content is not visible text and is excluded entirely (neither
# analyzed nor emitted)
HTML_SKIP_TAGS = frozenset(["script", "style"])

# Tags that do not have closing tags
HTML_VOID_TAGS = frozenset(
    [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    ]
)

# Tags that break the flow of text: a match may span adjacent inline tags
# (e.g. "co<strong>de</strong>") but never one of these boundaries
HTML_BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ]
)

# Default whitelist of tags preserved in the output. Anything not listed here
# is dropped (its text content is kept).
DEFAULT_HTML_FRAGMENT_TAGS = frozenset(
    [
        "a",
        "abbr",
        "b",
        "blockquote",
        "br",
        "cite",
        "code",
        "dd",
        "del",
        "dfn",
        "div",
        "dl",
        "dt",
        "em",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "i",
        "ins",
        "kbd",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
        "var",
    ]
)

# Default whitelist of attributes, mapped by tag. The special key "*" applies
# to every allowed tag.
DEFAULT_HTML_FRAGMENT_ATTRIBUTES = {
    "*": frozenset(["title"]),
    "a": frozenset(["href", "title"]),
}

# Attributes whose values are URLs and must pass the scheme check
HTML_URL_ATTRIBUTES = frozenset(["href", "src", "cite", "action"])

# URL schemes considered safe in URL attributes
HTML_SAFE_URL_SCHEMES = frozenset(["http", "https", "mailto"])

_url_scheme_pat = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")
_tag_name_pat = re.compile(r"^[a-zA-Z][a-zA-Z0-9-]*$")


def _is_safe_url(value):
    # Strip whitespace and control characters browsers would ignore
    cleaned = "".join(value.split())
    match = _url_scheme_pat.match(cleaned)
    if match:
        return match.group(1).lower() in HTML_SAFE_URL_SCHEMES
    # No scheme: relative URL, fragment, or protocol-relative URL
    return True


class _HtmlFragmentParser(HTMLParser):
    """Parses an HTML fragment into a flat list of events while building the
    visible text (entities decoded, script/style content removed, block
    boundaries inserted as newlines).

    Events are lists of the form ["start", tag, attrs], ["end", tag],
    ["void", tag, attrs], or ["text", decoded_text].
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.events = []
        self.visible = []
        self._skip_tag = None
        self._skip_depth = 0

    def _boundary(self):
        if self.visible and self.visible[-1] != "\n":
            self.visible.append("\n")

    def _append_text(self, text):
        if not text:
            return
        if self.events and self.events[-1][0] == "text":
            self.events[-1][1] += text
        else:
            self.events.append(["text", text])
        self.visible.extend(text)

    def handle_starttag(self, tag, attrs):
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in HTML_SKIP_TAGS:
            self._skip_tag = tag
            self._skip_depth = 1
            return
        if tag in HTML_BLOCK_TAGS:
            self._boundary()
        if tag in HTML_VOID_TAGS:
            self.events.append(["void", tag, attrs])
            if tag in HTML_BLOCK_TAGS:
                self._boundary()
        else:
            self.events.append(["start", tag, attrs])

    def handle_startendtag(self, tag, attrs):
        if self._skip_depth or tag in HTML_SKIP_TAGS:
            return
        if tag in HTML_BLOCK_TAGS:
            self._boundary()
        self.events.append(["void", tag, attrs])
        if tag in HTML_BLOCK_TAGS:
            self._boundary()

    def handle_endtag(self, tag):
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if not self._skip_depth:
                    self._skip_tag = None
            return
        if tag in HTML_SKIP_TAGS or tag in HTML_VOID_TAGS:
            return
        if tag in HTML_BLOCK_TAGS:
            self._boundary()
        self.events.append(["end", tag])

    def handle_data(self, data):
        if not self._skip_depth:
            self._append_text(data)

    def handle_entityref(self, name):
        if not self._skip_depth:
            self._append_text(htmlunescape("&%s;" % name))

    def handle_charref(self, name):
        if not self._skip_depth:
            self._append_text(htmlunescape("&#%s;" % name))


def merge_ranges(ranges):
    """Deterministically merges overlapping or directly adjacent (start, end)
    ranges: ranges are sorted by (start, end) and any range that starts at or
    before the end of the previous one is folded into it. Returns a list of
    (start, end) tuples.
    """

    merged = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


class HtmlHighlightResult:
    """The result of :func:highlight_html.

    visible_text
        The text extracted from the HTML fragment (entities decoded,
        script/style content removed, block boundaries as newlines). This is
        the text the analyzer saw.

    matches
        A list of (startchar, endchar) tuples into visible_text, merged so
        they never overlap.

    html
        The highlighted HTML fragment, rebuilt using the tag/attribute
        whitelist with all text and attribute values escaped.
    """

    def __init__(self, visible_text, matches, html):
        self.visible_text = visible_text
        self.matches = matches
        self.html = html

    def __str__(self):
        return self.html

    def __repr__(self):
        return "%s(%r, %r)" % (
            self.__class__.__name__,
            self.visible_text,
            self.matches,
        )


def _render_html_attrs(tag, attrs, attributes):
    allowed = set(attributes.get("*", ()))
    allowed |= set(attributes.get(tag, ()))
    parts = []
    seen = set()
    for name, value in attrs:
        if not name:
            continue
        name = name.lower()
        if name in seen or name not in allowed:
            continue
        seen.add(name)
        if value is None:
            value = ""
        if name in HTML_URL_ATTRIBUTES and not _is_safe_url(value):
            continue
        parts.append(' %s="%s"' % (name, htmlescape(value, quote=True)))
    return "".join(parts)


def _render_html_fragment(events, matches, tags, attributes, mark_open, mark_close):
    output = []
    stack = []
    visible_index = 0
    match_index = 0

    for event in events:
        kind = event[0]
        if kind == "text":
            run = event[1]
            run_start = visible_index
            run_end = run_start + len(run)
            visible_index = run_end

            # Advance past matches that end before this run
            while match_index < len(matches) and matches[match_index][1] <= run_start:
                match_index += 1

            cursor = 0
            index = match_index
            while index < len(matches) and matches[index][0] < run_end:
                start, end = matches[index]
                index += 1
                if end <= run_start:
                    continue
                local_start = max(start, run_start) - run_start
                local_end = min(end, run_end) - run_start
                if local_start > cursor:
                    output.append(htmlescape(run[cursor:local_start], quote=False))
                output.append(mark_open)
                output.append(htmlescape(run[local_start:local_end], quote=False))
                output.append(mark_close)
                cursor = local_end
            if cursor < len(run):
                output.append(htmlescape(run[cursor:], quote=False))
        elif kind == "start":
            tag = event[1]
            if tag in tags:
                output.append(
                    "<%s%s>" % (tag, _render_html_attrs(tag, event[2], attributes))
                )
                stack.append(tag)
        elif kind == "void":
            tag = event[1]
            if tag in tags:
                attr_string = _render_html_attrs(tag, event[2], attributes)
                if tag in HTML_VOID_TAGS:
                    output.append("<%s%s>" % (tag, attr_string))
                else:
                    output.append("<%s%s></%s>" % (tag, attr_string, tag))
        else:  # "end"
            tag = event[1]
            if tag in stack:
                # Close any tags still open inside this one to keep the
                # output well-formed, then close the tag itself
                while stack:
                    open_tag = stack.pop()
                    output.append("</%s>" % open_tag)
                    if open_tag == tag:
                        break

    while stack:
        output.append("</%s>" % stack.pop())

    return "".join(output)


def highlight_html(
    text,
    terms,
    analyzer=None,
    tags=None,
    attributes=None,
    mark_tag="mark",
    mark_class=None,
    mode="query",
):
    """Highlights query terms in a fragment of HTML.

    Only the visible text nodes of the fragment are analyzed: tags and
    attributes never participate in matching, and the content of script and
    style elements is excluded entirely. Entities are decoded for matching
    ("&amp;" is one visible character). A term may span adjacent inline tags
    ("co<strong>de</strong>" matches "code", producing one highlight element
    per text node) but never crosses a block-level boundary such as a
    paragraph.

    The output is rebuilt using only the whitelisted tags and attributes,
    with all text and attribute values escaped, so the input HTML is never
    trusted. URL attributes (href etc.) are dropped unless they use a safe
    scheme (http, https, mailto) or no scheme.

    :param text: the HTML fragment to highlight.
    :param terms: a collection of term strings to highlight (matched against
        the analyzed token texts, as with :func:highlight).
    :param analyzer: the analyzer to tokenize the visible text with. If None,
        a whoosh.analysis.StandardAnalyzer is used.
    :param tags: the set of tag names allowed in the output. Defaults to
        DEFAULT_HTML_FRAGMENT_TAGS.
    :param attributes: a dict mapping tag names to sets of allowed attribute
        names (the key "*" applies to all tags). Defaults to
        DEFAULT_HTML_FRAGMENT_ATTRIBUTES.
    :param mark_tag: the tag name to wrap matches with.
    :param mark_class: an optional class name to add to the match elements.
    :param mode: the analyzer mode (as in :func:highlight).
    :returns: an HtmlHighlightResult with the visible text, the merged match
        ranges in the visible text, and the highlighted HTML fragment.
    """

    if analyzer is None:
        analyzer = StandardAnalyzer()
    if tags is None:
        tags = DEFAULT_HTML_FRAGMENT_TAGS
    if attributes is None:
        attributes = DEFAULT_HTML_FRAGMENT_ATTRIBUTES
    if not _tag_name_pat.match(mark_tag):
        raise ValueError("Invalid mark tag name: %r" % mark_tag)

    parser = _HtmlFragmentParser()
    parser.feed(text)
    parser.close()

    visible_text = "".join(parser.visible)
    termset = frozenset(terms)
    tokens = analyzer(visible_text, chars=True, mode=mode, removestops=False)
    matches = merge_ranges(
        (t.startchar, t.endchar)
        for t in set_matched_filter(tokens, termset)
        if t.matched
    )

    if mark_class is not None:
        mark_open = '<%s class="%s">' % (
            mark_tag,
            htmlescape(mark_class, quote=True),
        )
    else:
        mark_open = "<%s>" % mark_tag
    mark_close = "</%s>" % mark_tag

    html = _render_html_fragment(
        parser.events, matches, tags, attributes, mark_open, mark_close
    )
    return HtmlHighlightResult(visible_text, matches, html)
