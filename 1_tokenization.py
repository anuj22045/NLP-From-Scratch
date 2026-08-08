"""MODULE 01: TOKENIZATION
WHAT IS TOKENIZATION?
---------------------
Tokenization is the very FIRST step in any NLP pipeline.
It means breaking a large piece of text into smaller pieces called "tokens".

Think of it like this:
    - You have a paragraph (a big chunk of text).
    - You want to break it into sentences, words, or even characters.
    - Each small piece = one "token".

WHY DO WE NEED IT?
------------------
Computers don't understand text the way humans do.
They need the text to be split into small, manageable pieces so they can
process each piece individually (count words, find patterns, etc.)

TYPES OF TOKENIZATION:
----------------------
1. SENTENCE Tokenization  -> Splits text into sentences
2. WORD Tokenization       -> Splits text into words
3. CHARACTER Tokenization  -> Splits text into individual characters
4. SUBWORD Tokenization    -> Splits text into subword units (used in modern AI models)

LIBRARIES USED:
    - NLTK  (Natural Language Toolkit) -> for sentence & word tokenization
    - SpaCy                            -> for sentence & word tokenization
    - (Manual Python)                  -> for character tokenization
    - (Manual / tiktoken)              -> for subword tokenization concept
    """
# -----------------------------------------------------------------------------------------------------------------------------


# --- Using NLTK ---

# import nltk

# Download the tokenizer model (run only once, it downloads data needed for splitting)
# 'punkt_tab' contains rules that NLTK uses to figure out where sentences end.
# For example: periods (.), question marks (?), exclamation marks (!)
# nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello! How are you doing? I am learning NLP. It's really amazing. Dr. Smith went to Washington."

# sent_tokenize() looks for sentence boundaries (like ., ?, !)
# It is smart enough to know "Dr." is NOT the end of a sentence.
sentences = sent_tokenize(text)

print("=" * 70)
print("SENTENCE TOKENIZATION (NLTK)")
print("=" * 70)
print(f"Original Text:\n{text}\n")
print(f"Number of sentences found: {len(sentences)}\n")
for i, sentence in enumerate(sentences, 1):
    print(f"  Sentence {i}: {sentence}")


# --- Using SpaCy ---

import spacy

# Load the English language model
# This model has vocabulary, grammar rules, and trained pipelines
# Run this once if not installed: spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)  # spaCy processes the entire text through its pipeline

print("\n" + "=" * 70)
print("SENTENCE TOKENIZATION (SpaCy)")
print("=" * 70)
print(f"Original Text:\n{text}\n")
for i, sent in enumerate(doc.sents, 1):
    print(f"  Sentence {i}: {sent.text}")

# KEY DIFFERENCE:
# NLTK uses rule-based approach (pattern matching)
# SpaCy uses a trained statistical model (more accurate for complex text)

# -----------------------------------------------------------------------------
# SECTION 2: WORD TOKENIZATION (Breaking text into words)
# -----------------------------------------------------------------------------

text = "Hello! How are you doing? I am learning NLP. It's amazing."

# --- Using NLTK ---
# word_tokenize() splits text into individual words and punctuation
words_nltk = word_tokenize(text)

print("\n" + "=" * 70)
print("WORD TOKENIZATION (NLTK)")
print("=" * 70)
print(f"Original Text:\n{text}\n")
print(f"Number of tokens: {len(words_nltk)}")
print(f"Tokens: {words_nltk}")

# NOTICE: "It's" becomes TWO tokens -> ["It", "'s"]
# This is because NLTK breaks contractions (It's = It + 's)
# Punctuation marks (!, ?, .) are also separate tokens

# --- Using SpaCy ---
doc = nlp(text)
words_spacy = [token.text for token in doc]

print("\n" + "=" * 70)
print("WORD TOKENIZATION (SpaCy)")
print("=" * 70)
print(f"Number of tokens: {len(words_spacy)}")
print(f"Tokens: {words_spacy}")

# SpaCy also splits "It's" into ["It", "'s"]
# Both libraries handle contractions similarly


# ============================================================================
# SECTION 3: CHARACTER TOKENIZATION (Breaking text into characters)
# ============================================================================
"""
CHARACTER TOKENIZATION
----------------------
This is the simplest form. Each character (letter, space, punctuation)
becomes its own token. No special library needed - just use Python's list()

WHEN IS IT USED?
- In some deep learning models (character-level models)
- For languages that don't use spaces between words (like Chinese, Japanese)
- For spelling correction systems
"""

text = "Hello NLP!"

# Method 1: Simple list conversion
char_tokens = list(text)

print("\n" + "=" * 70)
print("CHARACTER TOKENIZATION")
print("=" * 70)
print(f"Original Text: {text}")
print(f"Number of character tokens: {len(char_tokens)}")
print(f"Character Tokens: {char_tokens}")

# Method 2: Lowercase character tokenization (more common in practice)
# Converting to lowercase makes it case-insensitive
char_tokens_lower = list(text.lower())
print(f"\nLowercase Tokens: {char_tokens_lower}")

# Method 3: Only alphabetic characters (removing spaces and punctuation)
char_tokens_alpha = [ch for ch in text.lower() if ch.isalpha()]
print(f"Alpha-only Tokens: {char_tokens_alpha}")


# ============================================================================
# SECTION 4: SUBWORD TOKENIZATION (Breaking text into subword pieces)
# ============================================================================

# SUBWORD TOKENIZATION
# --------------------
"""
This is used in modern AI / Transformer models (GPT, BERT, etc.)

WHY SUBWORD?
- Word tokenization has a problem: what about rare or unknown words?
  Example: "unhappiness" might not be in the vocabulary
- Subword tokenization breaks it into: ["un", "happi", "ness"]
  Now the model can understand it from its parts!

COMMON SUBWORD METHODS:
  1. BPE (Byte Pair Encoding)     -> Used in GPT models
  2. WordPiece                    -> Used in BERT
  3. SentencePiece                -> Used in T5, ALBERT

HOW DOES BPE WORK? (Simplified)
--------------------------------
Step 1: Start with all individual characters
Step 2: Find the most common pair of adjacent characters
Step 3: Merge that pair into a single token
Step 4: Repeat steps 2-3 until you reach the desired vocabulary size

Example with "low", "lower", "newest", "widest":
  Start:  l, o, w, e, r, n, w, e, s, t, w, i, d, e, s, t
  After BPE merges: "low", "er", "new", "est", "wid", "est"
  Common parts like "est" get their own token!
  """

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


# WHEN TO USE WHAT:
# - Sentence -> When you need to analyze individual sentences
# - Word     -> Most common, used in Bag of Words, TF-IDF, etc.
# - Character -> Character-level models, languages without spaces
# - Subword  -> Modern Transformer models (GPT, BERT, T5)