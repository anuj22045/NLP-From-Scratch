"""
==============================================================================
MODULE 04: STOPWORD REMOVAL
==============================================================================

WHAT ARE STOPWORDS?
-------------------
Stopwords are the most COMMON words in a language that carry very LITTLE
meaning. Examples: "the", "is", "in", "at", "a", "and", "or", "but"

WHY REMOVE THEM?
- Reduces data size (fewer words to process)
- Improves model accuracy (less noise)
- Speeds up processing

WHEN NOT TO REMOVE:
- Sentiment analysis: "not good" -> removing "not" changes meaning!
- Machine translation, Question answering

LIBRARIES: NLTK (predefined list), SpaCy (built-in detection)
==============================================================================
"""

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")

#==================================================================
#STOP WORDS WITH NLTK
#==================================================================

stop_words_nltk = set(stopwords.words('english'))

print("="*70)
print("STOP WORDS WITH NLTK")
print("="*70)

print("\n total english words in NLTK: ", {len(stop_words_nltk)})
print(f"First 20 (sorted): {sorted(list(stop_words_nltk))[:20]}")



text = "This is a great example to demonstrate basic NLP tasks using the NLTK library and also stop word removal."
words = word_tokenize(text.lower())

filtered = [w for w in words if w not in stop_words_nltk and w.isalpha()]

print(f"\n Original: {text}")
print(f"all tokens({len(words)}): {words}")
print(f"filtered tokens({len(filtered)}): {filtered}")


#==================================================================
#STOPWORDS REMOVAL WITH SPACY
#==================================================================

print("\n" + "=" * 70)
print("STOPWORD REMOVAL WITH SPACY")
print("=" * 70)

# SpaCy has built-in: token.is_stop (True if stopword)
doc = nlp(text)

print(f"\n {'word':<15} {'Is Stopword?'}")
print("-"*30)

for token in doc:
    print(f"{token.text:<15} {token.is_stop}")

    filtered_spacy = [t.text for t in doc if not t.is_stop and not t.is_punct]
    print(f"\n filtered: {filtered_spacy}")

    print(f"Spacy total Stopwords: {len(nlp.Defaults.stop_words)}")

    print(f"SpaCy total stopwords: {len(nlp.Defaults.stop_words)}")