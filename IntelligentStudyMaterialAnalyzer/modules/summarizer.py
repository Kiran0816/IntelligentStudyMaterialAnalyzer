import logging
from modules.preprocessing import sentence_segmentation, preprocess_text

logger = logging.getLogger(__name__)

# Lazy loading of Transformers pipeline
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            logger.info("Loading Hugging Face Transformers Summarization pipeline (t5-small)...")
            # Set device=-1 to force CPU execution since local systems might not have CUDA configured
            _summarizer = pipeline("summarization", model="t5-small", device=-1)
            logger.info("T5 Summarizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}. Summarizer will use extractive fallback.")
            raise e
    return _summarizer

def chunk_text(text, max_words=350):
    """
    Splits text into chunks of maximum words to respect transformer token limits (e.g., 512 tokens).
    Keeps sentence boundaries intact.
    """
    sentences = sentence_segmentation(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if current_word_count + words_in_sentence > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence
        else:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extractive_fallback_summary(text, ratio=0.25, min_sentences=3, max_sentences=6):
    """
    A lightweight extractive summarization algorithm:
    1. Scores sentences based on frequency of non-stopword tokens.
    2. Selects top-scoring sentences.
    3. Re-orders them as they originally appeared in the text.
    """
    logger.info("Running extractive fallback summarizer.")
    try:
        sentences = sentence_segmentation(text)
        if len(sentences) <= min_sentences:
            return text
            
        # Get word frequencies
        prep = preprocess_text(text)
        lemmas = prep["lemmatized"]
        from collections import Counter
        word_freqs = Counter(lemmas)
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            for word in words:
                # Add word freq to sentence score
                if word in word_freqs:
                    score += word_freqs[word]
            sentence_scores.append((i, score, sentence))
            
        # Sort sentences by score descending
        sorted_by_score = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Decide how many sentences to include
        num_sentences = max(min_sentences, min(max_sentences, int(len(sentences) * ratio)))
        top_sentences = sorted_by_score[:num_sentences]
        
        # Re-order by original index
        top_sentences_sorted = sorted(top_sentences, key=lambda x: x[0])
        
        summary = " ".join([item[2] for item in top_sentences_sorted])
        return summary
    except Exception as e:
        logger.error(f"Extractive fallback failed: {e}")
        # Simplest possible fallback: take the first few lines
        lines = text.split('.')
        return ". ".join(lines[:4]) + "."

def generate_summary(text):
    """
    Generates a summary of the provided text.
    Uses abstractive T5-Small model with text chunking.
    Falls back to frequency-based extractive summarization if T5 fails or is unavailable.
    """
    if not text or not text.strip():
        return ""
        
    # Check if text is too short to summarize
    word_count = len(text.split())
    if word_count < 50:
        return text
        
    try:
        summarizer_pipeline = get_summarizer()
        chunks = chunk_text(text)
        summaries = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
            # Format query for T5
            prompt_text = "summarize: " + chunk
            
            # Predict
            # Adjust max/min length dynamically based on chunk size
            words = len(chunk.split())
            max_len = min(150, max(30, int(words * 0.4)))
            min_len = min(30, int(max_len * 0.5))
            
            result = summarizer_pipeline(
                prompt_text, 
                max_length=max_len, 
                min_length=min_len, 
                do_sample=False
            )
            summaries.append(result[0]['summary_text'])
            
        final_summary = " ".join(summaries)
        logger.info("Abstractive summary generated successfully via T5-Small.")
        return final_summary
        
    except Exception as e:
        logger.warning(f"T5 summarization failed: {e}. Falling back to extractive summarizer.")
        return extractive_fallback_summary(text)
