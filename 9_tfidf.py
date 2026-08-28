"""
==============================================================================
MODULE 09: TF-IDF (Term Frequency - Inverse Document Frequency)
==============================================================================

WHAT IS TF-IDF?
---------------
TF-IDF is an IMPROVED version of Bag of Words.
Instead of just counting words, it measures how IMPORTANT a word is
to a document within a collection (corpus).

THE PROBLEM WITH BAG OF WORDS:
    In BoW, common words like "the", "is", "and" get HIGH counts.
    But these words aren't important! TF-IDF solves this.

TF-IDF HAS TWO PARTS:
-----------------------

1. TF (Term Frequency): How often does the word appear in THIS document?
    TF = (Number of times word appears in document) / (Total words in document)

    Example: "the cat sat on the mat" -> TF("the") = 2/6 = 0.33

2. IDF (Inverse Document Frequency): How RARE is the word across ALL documents?
    IDF = log(Total documents / Documents containing this word)

    If a word appears in EVERY document, IDF is LOW (it's common, not special)
    If a word appears in only ONE document, IDF is HIGH (it's rare, important!)

3. TF-IDF = TF x IDF
    High TF-IDF = word is frequent in this doc BUT rare in other docs (IMPORTANT!)
    Low TF-IDF  = word is either rare in this doc OR common in all docs

EXAMPLE:
    Doc 1: "the cat sat on the mat"
    Doc 2: "the dog sat on the log"
    Doc 3: "cats and dogs are animals"

    "the" -> appears in Doc 1, Doc 2 (common) -> LOW TF-IDF
    "cat" -> appears only in Doc 1 (rare)     -> HIGH TF-IDF for Doc 1
    "animals" -> only in Doc 3 (rare)         -> HIGH TF-IDF for Doc 3

LIBRARY: scikit-learn (sklearn) -> TfidfVectorizer
==============================================================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# ============================================================================
# SECTION 1: BASIC TF-IDF
# ============================================================================

print("=" * 70)
print("TF-IDF - BASIC EXAMPLE")
print("=" * 70)

documents = [
    "the cat sat on the mat",    # Doc 1
    "the dog sat on the log",    # Doc 2
    "cats and dogs are animals"  # Doc 3
]

# TfidfVectorizer handles everything:
# 1. Tokenizes text
# 2. Calculates TF for each word in each document
# 3. Calculates IDF for each word across all documents
# 4. Multiplies TF x IDF to get the final score
tfidf_vec = TfidfVectorizer()

matrix = tfidf_vec.fit_transform(documents)

# Display as a nice DataFrame table
df = pd.DataFrame(
    matrix.toarray(),
    columns=tfidf_vec.get_feature_names_out(),
    index=['Doc1', 'Doc2', 'Doc3']
)

print(f"\nDocuments:")
for i, doc in enumerate(documents, 1):
    print(f"  Doc {i}: {doc}")

print(f"\nTF-IDF Matrix:")
print(df.round(3))  # Round to 3 decimal places for readability

# READING THE OUTPUT:
# "the" has LOW scores everywhere (common word, appears in Doc1 and Doc2)
# "cat" has HIGH score in Doc1 (only appears there -> important for Doc1!)
# "animals" has HIGH score in Doc3 (unique to Doc3 -> very important!)
# "sat" has equal scores in Doc1 and Doc2 (appears in both)

# ============================================================================
# SECTION 2: TF-IDF FOR A LARGER CORPUS
# ============================================================================

print("\n" + "=" * 70)
print("TF-IDF ON EMAIL CORPUS")
print("=" * 70)

corpus = [
    "Win a free iPhone now click here for your prize",
    "Dear user please reset your password for security",
    "Get free tickets to the concert buy now",
    "Your meeting is scheduled for tomorrow morning",
    "Free offer buy one get one free limited time",
]

labels = ["spam", "ham", "spam", "ham", "spam"]

tfidf = TfidfVectorizer(stop_words='english', max_features=15)
matrix = tfidf.fit_transform(corpus)

df = pd.DataFrame(
    matrix.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=[f"Doc{i+1}({l})" for i, l in enumerate(labels)]
)
print(f"\nTF-IDF Matrix (top 15 features, stopwords removed):")
print(df.round(2))

# Notice: "free" has high scores in spam emails but 0 in ham
# This is exactly what helps ML models classify spam vs ham!

# ============================================================================
# SECTION 3: BAG OF WORDS vs TF-IDF COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: Bag of Words vs TF-IDF")
print("=" * 70)

from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "the cat sat on the mat the",
    "the dog sat on the log",
]

# Bag of Words (raw counts)
bow = CountVectorizer()
bow_matrix = bow.fit_transform(docs)

# TF-IDF (weighted)
tfidf2 = TfidfVectorizer()
tfidf_matrix = tfidf2.fit_transform(docs)

print(f"\nDocuments:")
for i, d in enumerate(docs):
    print(f"  Doc {i+1}: {d}")

print(f"\nBag of Words (raw counts):")
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names_out(), index=['Doc1', 'Doc2'])
print(df_bow)

print(f"\nTF-IDF (weighted importance):")
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf2.get_feature_names_out(), index=['Doc1', 'Doc2'])
print(df_tfidf.round(3))

print("""
KEY DIFFERENCES:
    BoW:    "the" = 3 (highest count in Doc1) -> misleadingly important
    TF-IDF: "the" = low score -> correctly identified as unimportant
    TF-IDF: "cat" = high in Doc1, "dog" = high in Doc2 -> correctly important!
""")

print("""
TF-IDF SUMMARY:
+---------------------+-------------------------------------------+
| Concept             | Details                                   |
+---------------------+-------------------------------------------+
| TF                  | How often a word appears in ONE document  |
| IDF                 | How rare a word is across ALL documents   |
| TF-IDF              | TF x IDF = importance score               |
| High TF-IDF         | Word is frequent here, rare elsewhere     |
| Low TF-IDF          | Word is common everywhere (less useful)   |
| vs Bag of Words     | BoW = raw counts, TF-IDF = smart weights  |
| Library             | sklearn.TfidfVectorizer                   |
| Used for            | Search ranking, text classification, etc. |
+---------------------+-------------------------------------------+

NLP PIPELINE SO FAR:
1. Tokenization (split text into tokens)
2. Stopword Removal (remove common words)
3. Stemming/Lemmatization (reduce to root form)
4. Feature Extraction (BoW or TF-IDF to convert text -> numbers)
5. Feed into ML model for classification, clustering, etc.
""")