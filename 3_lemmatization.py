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
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

print("\n" + "=" * 70)
print("SUBWORD TOKENIZATION (Concept Demonstration)")
print("=" * 70)

# Let's demonstrate BPE concept manually with a simple example
text = "unhappiness"

# A hypothetical subword vocabulary might break this as:
subword_tokens = ["un", "##happi", "##ness"]
# The "##" prefix means "this continues the previous token"
# This is the WordPiece style (used in BERT)

print(f"Original Word: {text}")
print(f"Subword Tokens (WordPiece style): {subword_tokens}")
print(f"Explanation: 'un' = prefix, '##happi' = root, '##ness' = suffix")

# BPE style (used in GPT) - uses special spacing markers
bpe_tokens = ["un", "happ", "iness"]
print(f"\nSubword Tokens (BPE style): {bpe_tokens}")

# Let's show a practical example using Python's built-in approach
# to simulate subword tokenization
words_to_tokenize = ["playing", "unhappiness", "internationalization", "preprocessing"]

print(f"\nManual Subword Breakdown:")
print(f"{'Word':<25} {'Possible Subwords'}")
print("-" * 60)
# These are approximate subword splits for demonstration
subword_map = {
    "playing":               ["play", "##ing"],
    "unhappiness":           ["un", "##happi", "##ness"],
    "internationalization":  ["inter", "##national", "##ization"],
    "preprocessing":         ["pre", "##process", "##ing"]
}
for word in words_to_tokenize:
    print(f"{word:<25} {subword_map[word]}")

print("\nNote: Modern models like GPT and BERT use subword tokenization")
print("      to handle any word, even ones they have never seen before!")


# ============================================================================
# SECTION 5: COMPARISON OF ALL TOKENIZATION TYPES
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: ALL TOKENIZATION TYPES")
print("=" * 70)

sample = "I am learning NLP!"

print(f"\nOriginal Text: '{sample}'\n")

# Sentence
print(f"1. Sentence Tokens:   {sent_tokenize(sample)}")

# Word
print(f"2. Word Tokens:       {word_tokenize(sample)}")

# Character
print(f"3. Character Tokens:  {list(sample)}")

# Subword (conceptual)
print(f"4. Subword Tokens:    ['I', 'am', 'learn', '##ing', 'NL', '##P', '!']")