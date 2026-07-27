import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import logging
import re

logger = logging.getLogger(__name__)

# Flag to check if NLTK resources are initialized
_nltk_initialized = False

def ensure_nltk_resources():
    global _nltk_initialized
    if not _nltk_initialized:
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet.zip', 'wordnet'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
            ('corpora/words', 'words')
        ]
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
        _nltk_initialized = True

# Lazy loader for spaCy model
_spacy_nlp = None

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                logger.info("Loading spaCy model 'en_core_web_sm'...")
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("spaCy model 'en_core_web_sm' not found. Downloading...")
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load spaCy: {e}. Fallback to NLTK will be used for NER.")
    return _spacy_nlp

def clean_text(text):
    """
    Cleans raw text by removing duplicate spaces, special characters,
    but keeping basic sentence structure.
    """
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control/non-printable characters
    text = "".join(ch for ch in text if ch.isprintable() or ch == '\n' or ch == '\t')
    return text.strip()

def sentence_segmentation(text):
    """Splits text into sentences."""
    ensure_nltk_resources()
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return sent_tokenize(cleaned)

def preprocess_text(text):
    """
    Full preprocessing pipeline:
    1. Tokenization
    2. Lowercasing
    3. Stopword removal
    4. POS Tagging
    5. Lemmatization
    Returns: Dict containing processed details and clean_text.
    """
    ensure_nltk_resources()
    cleaned = clean_text(text)
    if not cleaned:
        return {"tokens": [], "clean_text": "", "pos_tags": [], "lemmatized": []}
        
    sentences = sent_tokenize(cleaned)
    words = word_tokenize(cleaned.lower())
    
    # Stopwords & punctuation removal
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # POS Tagging
    tagged_words = pos_tag(filtered_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word, tag in tagged_words:
        # Map Penn Treebank tag to WordNet POS tag
        wn_tag = 'n'
        if tag.startswith('V'):
            wn_tag = 'v'
        elif tag.startswith('J'):
            wn_tag = 'a'
        elif tag.startswith('R'):
            wn_tag = 'r'
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wn_tag))
        
    return {
        "tokens": words,
        "clean_text": cleaned,
        "pos_tags": tagged_words,
        "lemmatized": lemmatized_words
    }

def extract_entities(text):
    """
    Extracts named entities from the text.
    First tries spaCy, falls back to NLTK ne_chunk.
    Returns: List of dicts {"name": "...", "label": "..."}
    """
    nlp = get_spacy_nlp()
    entities = []
    
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append({
                    "name": ent.text,
                    "label": ent.label_
                })
            return entities
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}. Falling back to NLTK.")
            
    # NLTK Fallback
    try:
        ensure_nltk_resources()
        sentences = sent_tokenize(text)
        for sent in sentences:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            chunked = nltk.ne_chunk(tagged, binary=False)
            
            for subtree in chunked:
                if hasattr(subtree, 'label'):
                    entity_name = " ".join([token for token, pos in subtree.leaves()])
                    entities.append({
                        "name": entity_name,
                        "label": subtree.label()
                    })
    except Exception as e:
        logger.error(f"NLTK entity extraction fallback failed: {e}")
        
    return entities
