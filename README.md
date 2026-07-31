# Natural Language Processing (NLP) – Complete Course

A structured, module-based NLP course built using the **NLTK** and **SpaCy** libraries. Every module includes detailed theoretical explanations through comments along with practical, executable code examples.

## Modules

### Module 01: Tokenization (`module_01_tokenization.py`)

* **Sentence Tokenization** – Splitting text into individual sentences (NLTK + SpaCy)
* **Word Tokenization** – Dividing text into separate words (NLTK + SpaCy)
* **Character Tokenization** – Converting text into individual characters
* **Subword Tokenization** – Splitting words into smaller subword units (BPE, WordPiece concepts)

### Module 02: Stemming (`module_02_stemming.py`)

* **Porter Stemmer** – Widely used and less aggressive stemming technique
* **Snowball Stemmer** – Enhanced version of Porter with multilingual support
* **Lancaster Stemmer** – Highly aggressive stemming algorithm
* Side-by-side comparison of all three stemming methods

### Module 03: Lemmatization (`module_03_lemmatization.py`)

* Lemmatization using **SpaCy** (automatic POS recognition)
* Lemmatization using **NLTK WordNet** (manual POS selection)
* Comparison of **Stemming vs. Lemmatization**
* End-to-end preprocessing pipeline example

### Module 04: Stopword Removal (`module_04_stopword_removal.py`)

* Stopword removal using **NLTK** (predefined stopword lists)
* Stopword removal using **SpaCy** (`is_stop` attribute)
* Adding and removing custom stopwords
* Working with multilingual stopwords
* Complete text preprocessing workflow

### Module 05: Named Entity Recognition (`module_05_ner.py`)

* Named Entity Recognition with **SpaCy**
* Recognition of entities such as **PERSON, ORG, GPE, DATE, MONEY**, and more
* Understanding **IOB (Inside-Outside-Beginning)** tagging
* Detecting multiple entity types from complex text

### Module 06: POS Tagging (`module_06_pos_tagging.py`)

* Part-of-Speech tagging using **SpaCy**
* Difference between **Coarse POS (`pos_`)** and **Fine-Grained POS (`tag_`)**
* Filtering words based on their POS categories (nouns, verbs, adjectives, etc.)

### Module 07: Dependency Parsing (`module_07_dependency_parsing.py`)

* Understanding dependency tree structures
* Extracting **Subject–Verb–Object (SVO)** relationships
* Visualizing dependency trees through HTML output

### Module 08: Bag of Words (`module_08_bag_of_words.py`)

* Understanding the Bag of Words concept
* Implementation using **CountVectorizer**
* Practical spam classification example
* Exploring **CountVectorizer** parameters (`max_features`, `stop_words`, `ngrams`)

### Module 09: TF-IDF (`module_09_tfidf.py`)

* Explanation of **TF**, **IDF**, and **TF-IDF** with formulas
* Implementation using **TfidfVectorizer**
* Comparison between **Bag of Words** and **TF-IDF**
* Final summary of the complete NLP preprocessing pipeline

---

# Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# Running the Modules

Execute each module separately:

```bash
python module_01_tokenization.py
python module_02_stemming.py
# Continue similarly for the remaining modules
```

# Recommended Learning Sequence

Study the modules in the given order (**01 → 09**) to build a strong foundation. Each module introduces concepts that are used and expanded upon in the following modules.
