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
