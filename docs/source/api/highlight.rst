====================
``highlight`` module
====================

.. automodule:: whoosh.highlight

See :doc:`how to highlight terms in search results </highlight>`.


Manual highlighting
===================

.. autoclass:: Highlighter
   :members:

.. autofunction:: highlight

.. autofunction:: highlight_html

.. autoclass:: HtmlHighlightResult
   :members:


Fragmenters
===========

.. autoclass:: Fragmenter
   :members:

.. autoclass:: WholeFragmenter
.. autoclass:: SentenceFragmenter
.. autoclass:: ContextFragmenter
.. autoclass:: PinpointFragmenter


Scorers
=======

.. autoclass:: FragmentScorer
.. autoclass:: BasicFragmentScorer


Formatters
==========

.. autoclass:: UppercaseFormatter
.. autoclass:: HtmlFormatter
.. autoclass:: GenshiFormatter
.. autoclass:: HtmlFragmentFormatter


Utility classes
===============

.. autoclass:: Fragment
    :members:
