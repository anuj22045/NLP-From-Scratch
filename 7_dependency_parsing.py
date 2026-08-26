"""
==============================================================================
MODULE 07: DEPENDENCY PARSING
==============================================================================

WHAT IS DEPENDENCY PARSING?
---------------------------
Dependency parsing shows the GRAMMATICAL RELATIONSHIPS between words.
It creates a tree structure where each word is connected to its "head" word.

Example: "I ate dinner"
    "I"      -> nsubj (nominal subject) of "ate"
    "ate"    -> ROOT (main verb of the sentence)
    "dinner" -> dobj (direct object) of "ate"

Think of it like a family tree, but for grammar:
    ate (ROOT)
    ├── I (subject - WHO ate?)
    └── dinner (object - WHAT was eaten?)

COMMON DEPENDENCY LABELS:
    ROOT   = The main verb of the sentence
    nsubj  = Nominal subject (who/what does the action)
    dobj   = Direct object (what receives the action)
    amod   = Adjective modifier (describes a noun)
    prep   = Prepositional modifier
    pobj   = Object of preposition
    det    = Determiner (the, a, an)
    conj   = Conjunction
    cc     = Coordinating conjunction (and, but, or)
    advmod = Adverbial modifier
    aux    = Auxiliary verb (is, was, have, will)

WHY IS IT USEFUL?
- Understanding sentence structure
- Question answering (who did what to whom?)
- Information extraction
- Machine translation
- Grammar checking
==============================================================================
"""

import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
# ============================================================================
# SECTION 1: BASIC DEPENDENCY PARSING
# ============================================================================

print("=" * 70)
print("DEPENDENCY PARSING")
print("=" * 70)

text = "I ate and then slept after heavy dinner."
doc = nlp(text)

print(f"\nText: {text}\n")
print(f"{'Word':<15} {'Dep Relation':<15} {'Head Word':<15} {'Explanation'}")
print("-" * 70)
for token in doc:
    print(f"{token.text:<15} {token.dep_:<15} {token.head.text:<15} {spacy.explain(token.dep_)}")

# READING THE OUTPUT:
# "I"      -> nsubj of "ate"     (I is the SUBJECT of ate)
# "ate"    -> ROOT               (main action of the sentence)
# "and"    -> cc of "ate"        (conjunction connecting two verbs)
# "slept"  -> conj of "ate"      (second verb connected by "and")
# "dinner" -> pobj of "after"    (object of the preposition "after")
# "heavy"  -> amod of "dinner"   (adjective modifying dinner)

# ============================================================================
# SECTION 2: UNDERSTANDING THE TREE STRUCTURE
# ============================================================================

print("\n" + "=" * 70)
print("DEPENDENCY TREE STRUCTURE")
print("=" * 70)

text2 = "The quick brown fox jumped over the lazy dog"
doc2 = nlp(text2)

print(f"\nText: {text2}\n")

# Show tree with depth
for token in doc2:
    # token.ancestors gives all parent tokens up to ROOT
    depth = len(list(token.ancestors))
    indent = "  " * depth
    print(f"{indent}├── {token.text} ({token.dep_}) -> head: {token.head.text}")


# ============================================================================
# SECTION 3: FINDING SUBJECT-VERB-OBJECT
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING SUBJECT-VERB-OBJECT (SVO)")
print("=" * 70)

sentences = [
    "The cat ate the fish",
    "John bought a new car",
    "She teaches math at school"
]

for sent in sentences:
    doc = nlp(sent)
    subject = [t.text for t in doc if t.dep_ == "nsubj"]
    verb = [t.text for t in doc if t.dep_ == "ROOT"]
    obj = [t.text for t in doc if t.dep_ in ("dobj", "attr")]

    print(f"\nSentence: {sent}")
    print(f"  Subject: {subject} | Verb: {verb} | Object: {obj}")


# ============================================================================
# SECTION 4: VISUALIZING THE DEPENDENCY TREE (HTML)
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZING DEPENDENCY TREE")
print("=" * 70)

text3 = "I ate and then slept after heavy dinner."
doc3 = nlp(text3)

# displacy.render creates an HTML visualization
html = displacy.render(doc3, style='dep')

from pathlib import Path

output_path = "dependency_tree.html"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nDependency tree saved to: {output_path}")
print("Open this file in your browser to see the visual tree!")

# For Jupyter Notebook use: displacy.render(doc3, style='dep', jupyter=True)

print("""
DEPENDENCY PARSING SUMMARY:
- Shows grammatical relationships between words
- Each word has a HEAD (parent) and a DEP (relationship label)
- ROOT = main verb, nsubj = subject, dobj = object
- displacy.render() creates beautiful HTML visualizations
- Used in: QA systems, info extraction, grammar checking
""")