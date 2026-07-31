# MODULE 01: TOKENIZATION
# WHAT IS TOKENIZATION?
# ---------------------
# Tokenization is the very FIRST step in any NLP pipeline.
# It means breaking a large piece of text into smaller pieces called "tokens".

# Think of it like this:
#     - You have a paragraph (a big chunk of text).
#     - You want to break it into sentences, words, or even characters.
#     - Each small piece = one "token".

# WHY DO WE NEED IT?
# ------------------
# Computers don't understand text the way humans do.
# They need the text to be split into small, manageable pieces so they can
# process each piece individually (count words, find patterns, etc.)

# TYPES OF TOKENIZATION:
# ----------------------
# 1. SENTENCE Tokenization  -> Splits text into sentences
# 2. WORD Tokenization       -> Splits text into words
# 3. CHARACTER Tokenization  -> Splits text into individual characters
# 4. SUBWORD Tokenization    -> Splits text into subword units (used in modern AI models)

# LIBRARIES USED:
#     - NLTK  (Natural Language Toolkit) -> for sentence & word tokenization
#     - SpaCy                            -> for sentence & word tokenization
#     - (Manual Python)                  -> for character tokenization
#     - (Manual / tiktoken)              -> for subword tokenization concept
# -----------------------------------------------------------------------------------------------------------------------------


# --- Using NLTK ---

import nltk

# Download the tokenizer model (run only once, it downloads data needed for splitting)
# 'punkt_tab' contains rules that NLTK uses to figure out where sentences end.
# For example: periods (.), question marks (?), exclamation marks (!)
nltk.download('punkt_tab', quiet=True)

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
