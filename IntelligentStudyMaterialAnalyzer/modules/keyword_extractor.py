import logging
from collections import Counter

logger = logging.getLogger(__name__)

def fallback_keyword_extractor(text, top_n=10):
    """
    Fallback keyword extractor based on word frequency.
    Extracts words, lemmatizes them, and returns most frequent ones.
    """
    logger.info("Running fallback keyword extractor (Word Frequency).")
    try:
        from modules.preprocessing import preprocess_text
        res = preprocess_text(text)
        words = [w for w in res['lemmatized'] if len(w) > 3] # ignore short tokens
        
        counter = Counter(words)
        most_common = counter.most_common(top_n)
        if not most_common:
            return []
            
        max_freq = most_common[0][1]
        # YAKE scores are smaller for more important words (usually 0 is best).
        # We simulate this score by taking: 1.0 - (frequency / max_frequency)
        return [(word, round(1.0 - freq/max_freq, 4)) for word, freq in most_common]
    except Exception as e:
        logger.error(f"Fallback keyword extraction failed: {e}")
        return []

def extract_keywords(text, top_n=10):
    """
    Extracts keywords using YAKE (Yet Another Keyword Extractor).
    Falls back to a word frequency extractor if yake fails or is not installed.
    """
    if not text or not text.strip():
        return []
        
    try:
        import yake
        logger.info("Extracting keywords using YAKE.")
        # n is the max ngram size
        kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=2, 
            dedupLim=0.9, 
            top=top_n, 
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        
        # YAKE returns (keyword, score)
        if keywords:
            return [(kw, round(score, 4)) for kw, score in keywords]
    except Exception as e:
        logger.warning(f"YAKE keyword extraction failed or not installed: {e}")
        
    return fallback_keyword_extractor(text, top_n)
