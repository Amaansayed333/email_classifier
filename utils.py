import re
import spacy

# Load spaCy model once at import time
nlp = spacy.load("en_core_web_sm")

def mask_pii(text):
    entities = []
    doc = nlp(text)
    masked_text = text
    
    # To handle offsets properly when replacing text, we build masked_text gradually
    masked_text_builder = []
    last_idx = 0
    
    # Collect entities to mask for PERSON and email regex matches
    pii_spacy = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            pii_spacy.append((ent.start_char, ent.end_char, "[full_name]", "full_name", ent.text))
    
    # Email regex pattern
    email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
    pii_regex = []
    for match in re.finditer(email_pattern, text):
        pii_regex.append((match.start(), match.end(), "[email]", "email", match.group()))
    
    # Combine all entities for masking, sort by start position
    pii_all = pii_spacy + pii_regex
    pii_all.sort(key=lambda x: x[0])
    
    current_pos = 0
    masked_text = ""
    for start, end, mask_str, classification, entity_text in pii_all:
        # Add text before entity
        masked_text += text[current_pos:start]
        # Add mask
        masked_text += mask_str
        # Record entity info (with updated positions in masked_text)
        entities.append({
            "position": [len(masked_text) - len(mask_str), len(masked_text)],
            "classification": classification,
            "entity": entity_text
        })
        current_pos = end
    # Add remaining text after last entity
    masked_text += text[current_pos:]
    
    return masked_text, entities
