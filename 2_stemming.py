# ==============================================================================
# MODULE 02: STEMMING
# ==============================================================================

# WHAT IS STEMMING?
# -----------------
# Stemming is the process of reducing a word to its ROOT FORM (called "stem").
# It chops off the ending of words using simple rules.

# Example:
#     "running"  -> "run"       (removed "ning")
#     "studies"  -> "studi"     (removed "es", changed "y" to "i")
#     "played"   -> "play"      (removed "ed")

# WHY DO WE NEED STEMMING?
# -------------------------
# In NLP, we often want to treat different forms of the same word as ONE word.
#     - "run", "running", "runs", "ran" -> all mean the same concept: "run"
#     - Without stemming, the computer treats them as 4 DIFFERENT words!
#     - With stemming, they all become "run" (or something close to it)

# This helps in:
#     - Search engines (searching "running" also finds "run", "runs")
#     - Text classification (reduces vocabulary size)
#     - Sentiment analysis (different forms = same meaning)

# IMPORTANT NOTE:
#     Stemming is FAST but NOT ALWAYS ACCURATE.
#     It uses simple rules to chop off word endings, so sometimes
#     the result is NOT a real word:
#         "studies" -> "studi" (not a real word, but that's ok!)
#         "fairly"  -> "fairli" (not a real word either)
    
#     If you need real dictionary words, use LEMMATIZATION (Module 03).

# THREE TYPES OF STEMMERS IN NLTK:
# ---------------------------------
# 1. PORTER STEMMER    -> Most popular, gentle, widely used
# 2. SNOWBALL STEMMER  -> Improved Porter, supports multiple languages
# 3. LANCASTER STEMMER -> Most aggressive, chops the most


# ==============================================================================

from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# Sample words to test stemming on
# These words have different forms of the same root
words = [
    'running', 'runner', 'ran',           # forms of "run"
    'easily', 'fairly', 'fairness',       # adverbs and nouns
    'studies', 'studying', 'studied',     # forms of "study"
    'happiness', 'happily', 'happy',      # forms of "happy"
    'connection', 'connected', 'connecting',  # forms of "connect"
    'generalization', 'generalize'        # forms of "general"
]


# ============================================================================
# SECTION 1: PORTER STEMMER
# ============================================================================
# HOW IT WORKS (Simplified):
# Step 1: Handle plurals and past tenses
#         "cats" -> "cat", "agreed" -> "agree"
# Step 2: Handle double suffixes
#         "relational" -> "relate"
# Step 3: Handle more suffixes
#         "electrical" -> "electric"
# Step 4: Handle yet more suffixes
#         "allowance" -> "allow"
# Step 5: Clean up endings
#         Remove final 'e' if the stem is long enough

porter = PorterStemmer()

print("=" * 70)
print("1. PORTER STEMMER (Gentle, Most Popular)")
print("=" * 70)
print(f"\n{'Original Word':<20} {'Stemmed Word':<20}")
print("-" * 40)
for word in words:
    print(f"{word:<20} {porter.stem(word):<20}")


# ============================================================================
# SECTION 2: SNOWBALL STEMMER
# ============================================================================

# SNOWBALL STEMMER
# -----------------------------------------
# - Created by the same person (Martin Porter) as an IMPROVEMENT
# - Fixes some issues in the original Porter stemmer
# - Supports MULTIPLE LANGUAGES (not just English!)
#   Supported languages: Arabic, Danish, Dutch, English, Finnish, French,
#   German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian,
#   Spanish, Swedish, Tamil, Turkish
#
# Example of improvement over Porter:
#   Porter:   "generalization" -> "gener"
#   Snowball: "generalization" -> "general"  (better!)

snowball = SnowballStemmer('english')  # specify the language

print("\n" + "=" * 70)
print("2. SNOWBALL STEMMER (Improved Porter, Multi-language)")
print("=" * 70)
print(f"\n{'Original Word':<20} {'Stemmed Word':<20}")
print("-" * 40)
for word in words:
    print(f"{word:<20} {snowball.stem(word):<20}")

# Show supported languages
print(f"\nSupported Languages: {SnowballStemmer.languages}")


# ============================================================================
# SECTION 3: LANCASTER STEMMER
# ============================================================================

# LANCASTER STEMMER (1990, by Paice/Husk)
# ----------------------------------------
# - The most AGGRESSIVE stemmer
# - Chops off MORE letters than Porter or Snowball
# - Results are often very short and may not be recognizable
# - FAST because of simple iterative rules
#
# Example:
#   Porter:     "running" -> "run"
#   Snowball:   "running" -> "run"
#   Lancaster:  "running" -> "run"  (same here)
#
#   Porter:     "generalization" -> "gener"
#   Snowball:   "generalization" -> "general"
#   Lancaster:  "generalization" -> "gen"  (too aggressive!)
#
# WHEN TO USE LANCASTER?
# - When you want maximum compression of vocabulary
# - When exact word form doesn't matter
# - Generally: NOT recommended for most tasks (too aggressive)

lancaster = LancasterStemmer()

print("\n" + "=" * 70)
print("3. LANCASTER STEMMER (Most Aggressive)")
print("=" * 70)
print(f"\n{'Original Word':<20} {'Stemmed Word':<20}")
print("-" * 40)
for word in words:
    print(f"{word:<20} {lancaster.stem(word):<20}")


# ============================================================================
# SECTION 5: STEMMING IN ACTION - A PRACTICAL EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("PRACTICAL EXAMPLE: Stemming a Sentence")
print("=" * 70)

sentence = "The children were happily playing in the gardens while their dogs were running around"

# Step 1: Split the sentence into words (basic tokenization)
words_in_sentence = sentence.lower().split()

# Step 2: Apply Porter Stemmer to each word
stemmed_words = [porter.stem(word) for word in words_in_sentence]

print(f"\nOriginal Sentence:")
print(f"  {sentence}")
print(f"\nAfter Stemming (Porter):")
print(f"  {' '.join(stemmed_words)}")

# NOTICE: "children" -> "children" (Porter doesn't handle irregular words)
#         "happily"  -> "happili" (not a real word, but captures the root)
#         "playing"  -> "play" (correctly reduced)
#         "gardens"  -> "garden" (correctly reduced)
#         "running"  -> "run" (correctly reduced)

print("""
SUMMARY:
+-------------------+---------------+------------------+--------------------+
| Feature           | Porter        | Snowball         | Lancaster          |
+-------------------+---------------+------------------+--------------------+
| Aggressiveness    | Low           | Medium           | High               |
| Speed             | Fast          | Fast             | Very Fast          |
| Accuracy          | Good          | Better           | Lower              |
| Multi-language    | English only  | 15+ languages    | English only       |
| Real words?       | Sometimes     | More often       | Rarely             |
| Best for          | General use   | Multi-lang apps  | Vocab compression  |
+-------------------+---------------+------------------+--------------------+

KEY TAKEAWAY:
- Use SNOWBALL for most tasks (best balance of speed and accuracy)
- Use PORTER if you want the classic, well-tested approach
- Avoid LANCASTER unless you specifically need aggressive stemming
- If you need REAL dictionary words, use LEMMATIZATION (next module!)
""")