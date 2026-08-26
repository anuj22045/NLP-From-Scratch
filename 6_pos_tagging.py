"""
==============================================================================
MODULE 06: PART OF SPEECH (POS) TAGGING
==============================================================================

WHAT IS POS TAGGING?
--------------------
POS tagging assigns a grammatical label to each word in a sentence.
It tells us WHAT ROLE each word plays.

    "Apple CEO Tim Cook announced a new iPhone"
    PROPN  NOUN PROPN PROPN VERB  DET ADJ  PROPN

COMMON POS TAGS:
    NOUN  = Noun (dog, city, book)
    VERB  = Verb (run, eat, is)
    ADJ   = Adjective (big, red, beautiful)
    ADV   = Adverb (quickly, very, well)
    PROPN = Proper Noun (Tim, Apple, India)
    DET   = Determiner (the, a, an)
    ADP   = Preposition (in, on, at, from)
    PRON  = Pronoun (I, he, she, it)
    CONJ  = Conjunction (and, but, or)
    PUNCT = Punctuation (., !, ?)

SpaCy provides TWO levels:
    token.pos_  -> Coarse POS (NOUN, VERB, ADJ) - simple category
    token.tag_  -> Fine-grained POS (NN, NNS, VBD) - detailed tag
        NN  = singular noun, NNS = plural noun
        VBD = past tense verb, VBG = gerund (-ing form)
        JJ  = adjective, RB  = adverb, etc.

WHY IS POS TAGGING USEFUL?
- Helps lemmatization (knowing "running" is a verb vs noun)
- Information extraction (find all nouns = key topics)
- Grammar checking, language translation
- Disambiguation ("bank" = river bank or financial bank?)
==============================================================================
"""

import spacy
nlp = spacy.load('en_core_web_sm')

# ============================================================================
# SECTION 1: BASIC POS TAGGING
# ============================================================================

print("=" * 70)
print("POS TAGGING (Part of Speech)")
print("=" * 70)

text = "Apple CEO Tim Cook announced a new iPhone in California on Monday"
doc = nlp(text)

print(f"\nText: {text}\n")
print(f"{'Word':<15} {'POS':<8} {'Tag':<8} {'Explanation'}")
print("-" * 55)
for token in doc:
    print(f"{token.text:<15} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")

# ============================================================================
# SECTION 2: DETAILED POS WITH LEMMA
# ============================================================================

print("\n" + "=" * 70)
print("POS WITH LEMMA (Detailed View)")
print("=" * 70)

text2 = "I had a flight from New York to San Francisco on 24th January 2026"
doc2 = nlp(text2)

print(f"\n{'Word':<15} {'Lemma':<15} {'POS':<10} {'Tag':<10} {'Explanation'}")
print("=" * 70)
for token in doc2:
    explanation = spacy.explain(token.tag_) or "N/A"
    print(f"{token.text:<15} {token.lemma_:<15} {token.pos_:<10} {token.tag_:<10} {explanation}")

# ============================================================================
# SECTION 3: EXTRACTING SPECIFIC POS
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING WORDS BY POS")
print("=" * 70)

text3 = "The beautiful cat quickly jumped over the lazy brown dog near the old bridge"
doc3 = nlp(text3)

nouns = [t.text for t in doc3 if t.pos_ == "NOUN"]
adjs = [t.text for t in doc3 if t.pos_ == "ADJ"]
verbs = [t.text for t in doc3 if t.pos_ == "VERB"]
advs = [t.text for t in doc3 if t.pos_ == "ADV"]

print(f"\nText: {text3}")
print(f"\nNouns:      {nouns}")
print(f"Adjectives: {adjs}")
print(f"Verbs:      {verbs}")
print(f"Adverbs:    {advs}")

# This is useful for: finding key topics (nouns), 
# sentiments (adjectives), actions (verbs)

print("""
POS TAGGING SUMMARY:
- pos_  = coarse tag (NOUN, VERB, ADJ)
- tag_  = fine-grained tag (NN, VBD, JJ)
- spacy.explain() gives human-readable description
- Useful for: lemmatization, info extraction, grammar checking
""")