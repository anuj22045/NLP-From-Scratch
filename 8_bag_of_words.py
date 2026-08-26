"""
==============================================================================
MODULE 08: BAG OF WORDS (BoW)
==============================================================================

WHAT IS BAG OF WORDS?
---------------------
Bag of Words is a way to convert TEXT into NUMBERS so that machine learning
models can understand it. Computers can't read text, they need numbers!

HOW DOES IT WORK? (Step by step)
---------------------------------
Step 1: Collect all unique words from all documents -> this is the VOCABULARY
Step 2: For each document, count how many times each vocabulary word appears
Step 3: Create a matrix where rows = documents, columns = words, values = counts

Example:
    Doc 1: "the cat sat"
    Doc 2: "the dog sat"

    Vocabulary: [cat, dog, sat, the]

    Matrix:
    cat  dog  sat  the
    Doc 1:  [  1,   0,   1,   1 ]   <- "cat" appears 1 time, "dog" 0 times
    Doc 2:  [  0,   1,   1,   1 ]   <- "cat" appears 0 times, "dog" 1 time

WHY "BAG"?
    Because we throw all words into a "bag" and just COUNT them.
    We IGNORE the order of words!
    "cat sat on mat" and "mat on sat cat" produce the SAME result.

    This is the main LIMITATION of BoW - word order is lost.

USE CASES:
    - Spam detection (spam emails have certain words more frequently)
    - Text classification
    - Document similarity
    - Simple search engines

LIBRARY: scikit-learn (sklearn) -> CountVectorizer
==============================================================================
"""

from sklearn.feature_extraction.text import CountVectorizer

# ============================================================================
# SECTION 1: BASIC BAG OF WORDS
# ============================================================================

print("=" * 70)
print("BAG OF WORDS - BASIC EXAMPLE")
print("=" * 70)

# Simple corpus (collection of documents/sentences)
simple_corpus = [
    "the cat sat on the mat",    # Doc 0
    "the dog sat on the log",    # Doc 1
    "cats and dogs are friends"  # Doc 2
]

# CountVectorizer does everything automatically:
# 1. Tokenizes text into words
# 2. Builds vocabulary (all unique words)
# 3. Counts word occurrences in each document
vectorizer = CountVectorizer()

# fit_transform: learn vocabulary FROM data AND transform to matrix
matrix = vectorizer.fit_transform(simple_corpus)

# Get the vocabulary (all unique words, sorted alphabetically)
vocab = vectorizer.get_feature_names_out()
print(f"\nVocabulary ({len(vocab)} words): {list(vocab)}")

# Convert sparse matrix to regular array for display
print(f"\nBoW Matrix:")
print(f"{'':>15}", end="")
for word in vocab:
    print(f"{word:>8}", end="")
print()
print("-" * (15 + 8 * len(vocab)))

for i, doc in enumerate(simple_corpus):
    row = matrix.toarray()[i]
    print(f"Doc {i} ({doc[:20]+'...':<20})", end="")
    for count in row:
        print(f"{count:>8}", end="")
    print()

# READING THE MATRIX:
# "the" appears 2 times in Doc 0 (because "the cat sat on THE mat" has "the" twice)
# "cat" appears 1 time in Doc 0, 0 times in Doc 1
# "friends" only appears in Doc 2

# ============================================================================
# SECTION 2: PRACTICAL EXAMPLE - SPAM DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("PRACTICAL EXAMPLE: SPAM vs HAM (Legitimate) EMAILS")
print("=" * 70)

# Sample email corpus
corpus = [
    "Congratulations! you have won a free lottery ticket. Click here to claim your prize",
    "Dear user, your account has been compromised. Please reset your password immediately",
    "Limited Time Offer! Buy one get one free on all products. Don't miss out!",
    "Hello friend, just wanted to check in and see how you are doing",
    "Reminder: your appointment is scheduled for tomorrow at 10 AM"
]

# Labels (what we want to predict using ML)
labels = ["spam", "spam", "spam", "ham", "ham"]
# Create BoW model
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(corpus)

print(f"\nVocabulary size: {len(vectorizer.get_feature_names_out())} unique words")
print(f"Matrix shape: {matrix.shape} (5 documents x {matrix.shape[1]} words)")

# Show word frequencies for first email
print(f"\nFirst Email (spam):")
print(f'  "{corpus[0]}"')
print(f"\nWord Frequencies:")
word_freq = dict(zip(vectorizer.get_feature_names_out(), matrix.toarray()[0]))
# Show only words that appear (count > 0)
for word, count in sorted(word_freq.items()):
    if count > 0:
        print(f"  {word:<20} : {count}")


#  ============================================================================
# SECTION 3: BOW OPTIONS
# ============================================================================

print("\n" + "=" * 70)
print("COUNTERVECTORIZER OPTIONS")
print("=" * 70)

docs = ["The cat sat on the mat", "The dog sat on the log"]

# Option 1: max_features - limit vocabulary size
v1 = CountVectorizer(max_features=4)  # Keep only top 4 words
m1 = v1.fit_transform(docs)
print(f"\nmax_features=4: {list(v1.get_feature_names_out())}")

# Option 2: stop_words - remove common English words
v2 = CountVectorizer(stop_words='english')
m2 = v2.fit_transform(docs)
print(f"stop_words='english': {list(v2.get_feature_names_out())}")

# Option 3: ngram_range - capture word pairs (bigrams)
v3 = CountVectorizer(ngram_range=(1, 2), max_features=10)
m3 = v3.fit_transform(docs)
print(f"bigrams (1,2): {list(v3.get_feature_names_out())}")

# BIGRAMS capture word ORDER (partially solving BoW's limitation)
# "not good" as a bigram is different from "good" alone!

print("""
BAG OF WORDS SUMMARY:
- Converts text to numbers by counting word frequencies
- Uses CountVectorizer from sklearn
- Creates a document-term matrix (rows=docs, cols=words)
- IGNORES word order (main limitation)
- Options: max_features, stop_words, ngram_range
- Used for: spam detection, text classification, document similarity
- Next step: TF-IDF (improves on BoW by weighting word importance)
""")