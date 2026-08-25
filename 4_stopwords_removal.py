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


# ============================================================================
# SECTION 3: CUSTOMIZING STOPWORDS
# ============================================================================

print("\n" + "=" * 70)
print("CUSTOMIZING STOPWORDS")
print("=" * 70)

#Adding custom stopwords

custom_stops = stop_words_nltk.copy()
custom_stops.add("example")
custom_stops.add("demonstrate")

words = word_tokenize(text.lower())
filtered_custom = [w for w in words if w not in custom_stops and w.isalpha()]
print(f"\n With custom stops added: {filtered_custom}")

#keeping negation words for sentiment analysis

sentiment_stops = stop_words_nltk.copy()
for w in ["not", "no", "never"]:
    sentiment_stops.discard(w)


text2 = "This movie is not good and I will never watch it again"

words2 = word_tokenize(text2.lower())
print(f"\n Original: (text2)")
print(f"Default Removal: {[w for w in words2 if w not in stop_words_nltk and w.isalpha()]}")
print(f"Sentiment removal: {[w for w in words2 if w not in sentiment_stops and w.isalpha()]}")
print(" -> 'not' and 'never' KEPT because they change meaning")



# ============================================================================
# SECTION 4: MULTI-LANGUAGE STOPWORDS
# ============================================================================


print("\n" + "=" * 70)
print("STOPWORDS IN OTHER LANGUAGES")
print("=" * 70)
print(f"Available: {stopwords.fileids()}")
for lang in ['spanish', 'french', 'german']:
    print(f"{lang.capitalize()} (first 8): {stopwords.words(lang)[:8]}")


# ============================================================================
# SECTION 5: PRACTICAL PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("PIPELINE: Tokenize -> Stopword Removal -> Lemmatize")
print("=" * 70)

text = "The quick brown foxes were jumping over the lazy dogs in the beautiful gardens"

doc = nlp(text)

result = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]

print(f"\n Original: {text}")
print(f"processed: {result}")
print("""
SUMMARY:
- Stopwords = common filler words (the, is, a, and...)
- NLTK: manual list filtering | SpaCy: token.is_stop attribute
- Customizable: add domain words, keep negations for sentiment
- Always consider your use case before removing stopwords!
""")