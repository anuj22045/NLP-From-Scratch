"""
==============================================================================
MODULE 05: NAMED ENTITY RECOGNITION (NER)
==============================================================================

WHAT IS NER?
------------
NER identifies and classifies REAL-WORLD ENTITIES in text into categories:
    - PERSON:  People's names        -> "Tim Cook", "Elon Musk"
    - ORG:     Organizations         -> "Apple", "Google", "NASA"
    - GPE:     Countries/Cities      -> "India", "California", "New York"
    - DATE:    Dates/Time            -> "Monday", "January 2026"
    - MONEY:   Monetary values       -> "$50 million"
    - PRODUCT: Products              -> "iPhone", "MacBook"

WHY IS NER IMPORTANT?
- Extracting information from news articles, resumes, medical records
- Powering search engines, chatbots, recommendation systems
- Building knowledge graphs

LIBRARY: SpaCy (pre-trained models with NER pipeline built-in)
==============================================================================
"""

import spacy
nlp = spacy.load('en_core_web_sm')

# ============================================================================
# SECTION 1: BASIC NER
# ============================================================================

print("=" * 70)
print("NAMED ENTITY RECOGNITION (NER)")
print("=" * 70)

text = "Apple CEO Tim Cook announced a new iphone in California on Monday"

doc = nlp(text)
print(f"\n text: {text}\n")
print(f"{'Entity':<20} {'Label':<10} {'Description'}")
print("-"*60)

for ent in doc.ents:
    print(f"{ent.text:<20} {ent.label_:<10} {spacy.explain(ent.label_)}")

# ============================================================================
# SECTION 2: IOB TAGGING (B-I-O Format)
# ============================================================================

# IOB = Inside, Outside, Beginning
# B = Beginning of an entity
# I = Inside (continuation of) an entity
# O = Outside (not an entity)
# Example: "Tim Cook" -> Tim=B-PERSON, Cook=I-PERSON


print("\n" + "=" *60)
print("IOB TAGGING (B-I-O Format)")
print("="*60)
print(f"\n{'word':<15}{'IOB':<5} {'Entity Type'}")
print("-"*35)
for token in doc:
    print(f"{token.text:<15} {token.ent_iob_:<5} {token.ent_type_}")

# ============================================================================
# SECTION 3: NER ON COMPLEX TEXT
# ============================================================================

print("\n" + "=" * 70)
print("NER ON COMPLEX TEXT")
print("=" * 70)

text2 = "I had a Flight New York to San Francisco on 24th January 2026. The ticket cost $450 from United Airlines."
doc2 = nlp(text2)
print(f"\n text: {text2}\n")
print(f"{'Entity':<25} {'Label':<12} {'Description'}")
print("-"*65)

print(f"\n Text: {text2}\n")
print(f"{'Entity':<25} {'Label':<12} {'Description'}")
print("-" * 65)

for ent in doc2.ents:
    print(f"{ent.text:<25} {ent.label_:<12} {spacy.explain(ent.label_)}")


# ENTITY LABELS EXPLAINED:
# PERSON   = People (Tim Cook, Barack Obama)
# ORG      = Organizations (Apple, Google, United Airlines)
# GPE      = Countries, Cities, States (India, California)
# DATE     = Dates (Monday, January 2026, yesterday)
# MONEY    = Money amounts ($450, 50 million dollars)
# CARDINAL = Numbers that aren't other types (100, three)
# ORDINAL  = Ordinal numbers (first, 24th)
# LOC      = Non-GPE locations (the Pacific Ocean, Mount Everest)
# NORP     = Nationalities, religious/political groups (Indian, Republican)

print("""
NER SUMMARY:
- Finds real-world entities (people, places, organizations, dates, money)
- SpaCy provides entity labels (PERSON, ORG, GPE, DATE, MONEY, etc.)
- IOB tagging marks Beginning(B), Inside(I), Outside(O) of entities
- Used in: information extraction, search engines, chatbots
""")