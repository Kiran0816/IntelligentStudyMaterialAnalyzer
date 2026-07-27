import random
import re
import logging
from modules.preprocessing import sentence_segmentation, extract_entities, ensure_nltk_resources

logger = logging.getLogger(__name__)

# Predefined category fallbacks for distractor generation
CATEGORY_FALLBACKS = {
    'PERSON': ['Albert Einstein', 'Isaac Newton', 'Marie Curie', 'Nikola Tesla', 'Alan Turing', 'Charles Darwin', 'Ada Lovelace'],
    'ORG': ['Google', 'Microsoft', 'IBM', 'Apple', 'Amazon', 'Meta', 'Intel', 'Oracle', 'Samsung'],
    'GPE': ['London', 'New York', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Rome', 'Toronto', 'Singapore'],
    'LOC': ['Mount Everest', 'Amazon River', 'Pacific Ocean', 'Sahara Desert', 'Grand Canyon', 'Alps'],
    'DATE': ['1995', '2001', '2010', '2015', '2020', '2025', '1989', '1945'],
    'CARDINAL': ['10', '50', '100', '500', '1000', '5', '12', '24']
}

def get_wordnet_cohyponyms(word):
    """
    Finds coordinate terms (sister terms) of a word using WordNet.
    E.g., for "dog", it might return "cat", "horse", "pig", etc.
    """
    ensure_nltk_resources()
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word.lower().replace(' ', '_'))
        if not synsets:
            return []
            
        synset = synsets[0]
        hypernyms = synset.hypernyms()
        if not hypernyms:
            return []
            
        cohyponyms = []
        for hypernym in hypernyms:
            for hyponym in hypernym.hyponyms():
                for lemma in hyponym.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != word.lower() and name not in cohyponyms:
                        cohyponyms.append(name.title())
        return cohyponyms
    except Exception as e:
        logger.warning(f"WordNet cohyponym lookup failed: {e}")
        return []

def generate_numeric_distractors(number_str):
    """
    Generates numeric distractors by adding/subtracting/scaling the target number.
    """
    # Try to extract the number
    num_match = re.search(r'[-+]?\d*\.\d+|\d+', number_str)
    if not num_match:
        return [number_str + " (A)", number_str + " (B)", number_str + " (C)"]
        
    num_val = float(num_match.group()) if '.' in num_match.group() else int(num_match.group())
    
    distractors = set()
    attempts = 0
    while len(distractors) < 3 and attempts < 15:
        attempts += 1
        # Apply scaling or random arithmetic
        offset_type = random.choice(['add', 'sub', 'mul', 'div'])
        if offset_type == 'add':
            val = num_val + random.randint(1, 10)
        elif offset_type == 'sub':
            val = num_val - random.randint(1, 10)
        elif offset_type == 'mul':
            val = num_val * random.choice([2, 5, 10])
        else:
            val = num_val / 2 if num_val % 2 == 0 else num_val + 2
            
        # Format back similarly
        if isinstance(num_val, int):
            val = int(val)
        val_str = str(val)
        formatted = number_str.replace(num_match.group(), val_str)
        if formatted != number_str:
            distractors.add(formatted)
            
    # If failed to generate 3, add fallback numeric values
    while len(distractors) < 3:
        distractors.add(str(random.randint(1, 100)))
        
    return list(distractors)

def generate_mcqs(text, count=5):
    """
    Generates multiple-choice questions from the text.
    1. Extract named entities.
    2. Match sentences containing these entities.
    3. Blank out the entity.
    4. Generate distractors from:
       - Same-type entities in text.
       - WordNet coordinate terms.
       - Category-specific default lists.
    """
    if not text or not text.strip():
        return []
        
    # Segment sentences
    sentences = sentence_segmentation(text)
    if len(sentences) < 2:
        return []
        
    # Extract entities
    entities = extract_entities(text)
    if not entities:
        # If no NER entities found, fall back to YAKE keywords or noun phrases
        logger.info("No NER entities found for MCQ. Using YAKE keywords.")
        from modules.keyword_extractor import extract_keywords
        kws = extract_keywords(text, top_n=10)
        entities = [{"name": kw[0], "label": "KEYWORD"} for kw in kws]
        
    if not entities:
        return []
        
    # Group entities by label/type for contextual distractors
    entities_by_label = {}
    for ent in entities:
        lbl = ent['label']
        name = ent['name'].strip()
        if len(name) < 2:
            continue
        if lbl not in entities_by_label:
            entities_by_label[lbl] = set()
        entities_by_label[lbl].add(name)
        
    generated_mcqs = []
    used_questions = set()
    
    # Shuffle sentences to sample from different parts of the document
    random.seed(42) # set seed for consistency within a document
    shuffled_sentences = sentences.copy()
    random.shuffle(shuffled_sentences)
    
    for sentence in shuffled_sentences:
        # Check sentence length (prefer medium sentences 8-30 words)
        words = sentence.split()
        if len(words) < 8 or len(words) > 30:
            continue
            
        # Find which entities are in this sentence
        for ent in entities:
            ent_name = ent['name']
            ent_label = ent['label']
            
            # Match exact word boundaries of entity to avoid matching substrings (e.g. "BERT" in "Robert")
            pattern = r'\b' + re.escape(ent_name) + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE):
                # We found a candidate sentence and entity!
                # Create the question by replacing entity with blank
                # Case-insensitive replacement
                question_text = re.sub(pattern, "_______", sentence, flags=re.IGNORECASE)
                question_text = question_text.strip()
                
                # Check if we already generated a similar question
                if question_text in used_questions:
                    continue
                    
                # Generate distractors
                distractors = set()
                
                # 1. Try other entities of the same type from the text
                same_type_ents = entities_by_label.get(ent_label, set())
                for other in same_type_ents:
                    if other.lower() != ent_name.lower() and len(distractors) < 3:
                        distractors.add(other.title())
                        
                # 2. Try WordNet cohyponyms (works best for singular nouns/words)
                if len(distractors) < 3 and ' ' not in ent_name:
                    cohyponyms = get_wordnet_cohyponyms(ent_name)
                    for co in cohyponyms:
                        if len(distractors) < 3:
                            distractors.add(co)
                            
                # 3. Numeric handling for numbers/dates
                if len(distractors) < 3 and (ent_label in ('DATE', 'CARDINAL') or re.search(r'\d+', ent_name)):
                    num_dist = generate_numeric_distractors(ent_name)
                    for nd in num_dist:
                        if len(distractors) < 3:
                            distractors.add(nd)
                            
                # 4. Fall back to category predefined lists
                if len(distractors) < 3:
                    fallback_list = CATEGORY_FALLBACKS.get(ent_label, ['Context A', 'Context B', 'Context C', 'Context D'])
                    random.shuffle(fallback_list)
                    for item in fallback_list:
                        if item.lower() != ent_name.lower() and len(distractors) < 3:
                            distractors.add(item)
                            
                # 5. Last resort: select random words from text
                if len(distractors) < 3:
                    logger.info("Using last resort random word distractors.")
                    words_clean = [w.strip(".,?!();:\"'") for w in words if len(w) > 4 and w.lower() != ent_name.lower()]
                    random.shuffle(words_clean)
                    for w in words_clean:
                        if len(distractors) < 3:
                            distractors.add(w.title())
                            
                # Check if we have 3 distractors
                if len(distractors) >= 3:
                    distractor_list = list(distractors)[:3]
                    options = distractor_list + [ent_name.title()]
                    # Shuffle options
                    random.shuffle(options)
                    
                    mcq = {
                        'question': question_text,
                        'options': options,
                        'correct_answer': ent_name.title()
                    }
                    
                    generated_mcqs.append(mcq)
                    used_questions.add(question_text)
                    break # Only make one question per sentence
                    
        if len(generated_mcqs) >= count:
            break
            
    # Reset seed to random
    random.seed(None)
    
    return generated_mcqs
