"""
==============================================================================
MODULE 03: LEMMATIZATION
==============================================================================

WHAT IS LEMMATIZATION?
----------------------
Lemmatization reduces a word to its BASE DICTIONARY FORM called a "lemma".
Unlike stemming, it ALWAYS produces a real, valid English word.

Examples:
    "running"   -> "run"       (verb form)
    "better"    -> "good"      (adjective form - stemming can't do this!)
    "studies"   -> "study"     (noun/verb form)
    "was"       -> "be"        (irregular verb)
    "mice"      -> "mouse"     (irregular plural)
    "happily"   -> "happily"   (adverb stays as is)

STEMMING vs LEMMATIZATION - THE KEY DIFFERENCE:
------------------------------------------------
+------------------+-------------------+-------------------+
| Word             | Stemming          | Lemmatization     |
+------------------+-------------------+-------------------+
| "better"         | "better"          | "good"            |
| "studies"        | "studi"           | "study"           |
| "running"        | "run"             | "run"             |
| "was"            | "wa"              | "be"              |
| "mice"           | "mice"            | "mouse"           |
| "feet"           | "feet"            | "foot"            |
+------------------+-------------------+-------------------+

Stemming:       Just chops off endings (fast but crude)
Lemmatization:  Uses vocabulary + grammar rules (slower but accurate)

HOW DOES LEMMATIZATION WORK?
-----------------------------
1. It looks up the word in a dictionary/vocabulary
2. It considers the Part of Speech (POS) of the word
    - "running" as a VERB -> "run"  
    - "running" as a NOUN (as in "morning running") -> "running"
3. It applies morphological analysis (understanding word structure)

LIBRARIES USED:
    - SpaCy   -> Built-in lemmatization (recommended, easy to use)
    - NLTK    -> WordNet Lemmatizer (needs POS tag as input)

==============================================================================
"""