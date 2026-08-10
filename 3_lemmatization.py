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
# ============================================================================
# SECTION 1: LEMMATIZATION WITH SPACY
# ============================================================================

import spacy 
#load the english language model
#this model contain vocabulary, grammar rules, and vectors
nlp = spacy.load("en_core_web_sm")

print("="*65)
print("Lemmatization with Spacy")
print("="*65)


# -----example 1: Simpple Sentence----
text = "The mice were eating delicious meals while the children played happily in the parks"
doc = nlp(text)

print(f"\n Original text: {text}\n")
print(f"{'Word':<15} {'Lemma':<15} {'POS':<10} {'Explanation'}")
print("-"*50)

for token in doc:
    print(f"{token.text:<15} {token.lemma_:<15} {token.pos_:<10} {spacy.explain(token.pos_)}")


#-----example 2 : large text -------

print("\n" + "=" * 70)
print("LEMMATIZATION ON COMPLEX TEXT")
print("=" * 70)

text2 = """Based on coalescence of Mitochondrial DNA and Y Chromosome data, 
it is thought that the earliest extant lineages of anatomically modern humans 
had reached there from Africa between 80,000 and 50,000 years ago. Their long 
occupation, initially in varying forms of isolation as hunter-gatherers, has 
made the region highly diverse. These cultures gradually evolved into the 
Indus Valley Civilisation, which flourished during 2500-1900 BCE."""

doc2 = nlp(text2)

print(f"\nOriginal Text:\n{text2}\n")
print(f"{'Word':<20} -----> {'Lemma':<20}")
print("-" * 45)

for token in doc2:
    if token.text.lower() != token.lemma_.lower() and token.is_alpha:
        print(f"{token.text:<20} --------> {token.lemma_:<20}")


# ============================================================================
# SECTION 2: LEMMATIZATION WITH NLTK (WordNet Lemmatizer)
# ============================================================================

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("\n" + "=" * 70)
print("LEMMATIZATION WITH NLTK (WordNet Lemmatizer)")
print("=" * 70)

# IMPORTANT: NLTK's WordNetLemmatizer needs you to specify the POS (Part of Speech)
# If you don't specify, it assumes the word is a NOUN by default
# POS tags for WordNet:
#   'n' = noun, 'v' = verb, 'a' = adjective, 'r' = adverb

lemmatizer = WordNetLemmatizer()

print("\n without POS (default to noun)")
test_words = ['running', 'better', 'studies', 'geese', 'was', 'happily']
for word in test_words:
    result = lemmatizer.lemmatize(word)
    print(f"  {word:<15} -> {result:<15}")

# NOTICE: "running" stays "running" because as a NOUN, "running" IS the base form!
# "better" stays "better" because as a NOUN, it doesn't change!


#with correct POS

print("\nWith correct POS specified:")
word_pos_pairs = [
    ('running', 'v'),     # verb -> should give "run"
    ('better', 'a'),      # adjective -> should give "good"
    ('studies', 'n'),     # noun -> should give "study"
    ('studies', 'v'),     # verb -> should give "study"
    ('geese', 'n'),       # noun -> should give "goose"
    ('was', 'v'),         # verb -> should give "be" (may not work in NLTK)
    ('happily', 'r'),     # adverb -> should give "happily"
]

for word, pos in word_pos_pairs:
    result = lemmatizer.lemmatize(word, pos=pos)
    pos_name = {'n': 'noun', 'v': 'verb', 'a': 'adj', 'r': 'adverb'}[pos]
    print(f"  {word:<15} (as {pos_name:<8}) -> {result:<15}")


# ============================================================================
# SECTION 3: SPACY vs NLTK COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: SpaCy vs NLTK Lemmatization")
print("=" * 70)

comparison_words = ['running', 'better', 'studies', 'was', 'mice', 'feet', 'happily', 'went', 'children']

print(f"\n{'Word':<15} {'SpaCy':<15} {'NLTK (noun)':<15} {'NLTK (verb)':<15}")
print("-" * 60)

for word in comparison_words:
    # SpaCy lemmatization (automatic POS detection)
    doc = nlp(word)
    spacy_lemma = doc[0].lemma_

    # NLTK lemmatization (manual POS)
    nltk_noun = lemmatizer.lemmatize(word, pos='n')
    nltk_verb = lemmatizer.lemmatize(word, pos='v')

    print(f"{word:<15} {spacy_lemma:<15} {nltk_noun:<15} {nltk_verb:<15}")


# KEY INSIGHT:
# SpaCy is better for lemmatization because:
# 1. It automatically detects POS (you don't have to specify it)
# 2. It handles irregular words better ("better" -> "good", "was" -> "be")
# 3. It uses context to determine the correct lemma


# ============================================================================
# SECTION 4: PRACTICAL EXAMPLE - LEMMATIZING A FULL SENTENCE
# ============================================================================

print("\n" + "=" * 70)
print("PRACTICAL EXAMPLE: Lemmatization in Action")
print("=" * 70)

sentence = "The striped bats were hanging on their feet and ate best fishes"

# Using SpaCy for lemmatization
doc = nlp(sentence)
lemmatized_sentence = " ".join([token.lemma_ for token in doc])

print(f"\nOriginal:    {sentence}")
print(f"Lemmatized:  {lemmatized_sentence}")

# NOTICE:
# "striped"  -> "stripe"   (adjective/verb reduced)
# "bats"     -> "bat"      (plural -> singular)
# "were"     -> "be"       (irregular verb)
# "hanging"  -> "hang"     (verb reduced)
# "feet"     -> "foot"     (irregular plural!)
# "ate"      -> "eat"      (irregular past tense!)
# "best"     -> "good"     (superlative -> base form!)
# "fishes"   -> "fish"     (plural -> singular)